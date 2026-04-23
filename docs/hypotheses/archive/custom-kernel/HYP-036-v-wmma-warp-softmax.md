# HYP-036: WMMA for V accumulate + warp-reduced softmax

## Hypothesis

The HYP-032 addendum sweep extended to seq=32k showed v5_paged regrows
with context length (110 μs @ 8k → 314 μs @ 32k). I assumed the cause was
compute-bound scalar-FMA dequant and scoped a full Marlin pipelining HYP-036.

Then I actually profiled it. nsys confirmed the v5 kernel is 91.7% of GPU time
at seq=16384, but the assumed bottleneck was wrong. An in-kernel clock64()
phase probe (bypassing DCGM-locked ncu) gave this per-phase breakdown:

```
phase                            ~ns    share
1_K_cp_async_submit              114     5.1%
2_K_norms_precompute              80     3.6%
3_K_wait_group                     6     0.3%
4_K_dequant                      112     5.0%
5_WMMA_QK                        924    41.2%
6_softmax_update                 327    14.6%   ← scalar FMA
7_V_cp_async_submit              114     5.1%
8_V_norms_precompute              80     3.6%
9_V_wait_group                     6     0.3%
10_V_dequant                     112     5.0%
11_V_accumulate                  370    16.5%   ← scalar FMA, NOT WMMA
TOTAL                           2245   100.0%
```

**Dequant is 10% combined** — HYP-032's shuffle LUT already cleared that bar.

The two biggest *addressable* phases are both scalar, both 17/15% of total:
- **V accumulate** (`O += P · V`) uses scalar FMA across head_dim × tile_n
  even though the QK phase right next to it uses WMMA. Inconsistency.
- **Softmax update** does 16 scalar exp/FMA per row sequentially. A warp
  shuffle reduction over the 16 tokens would replace 16 serial ops with a
  log₂(16)=4-step butterfly.

WMMA_QK at 41% is the biggest phase but mostly dictated by smem load
latency (8 k-tiles × 2 `load_matrix_sync` + 8 `mma_sync` ≈ 680 ns minimum at
Ampere, vs 924 ns measured). Closing that gap needs FlashAttention-style
multi-stage pipelining — full kernel rewrite, out of scope for this hyp.

## Prediction

A100, same rig as HYP-035/032 (bs=1, kv_heads=8, qo_heads=32, bdy=4, head_dim=128):

1. **V accumulate → WMMA**: cut phase from 370 ns to ~200 ns. Savings
   ~7.5% of per-tile kernel time.
2. **Warp-reduced softmax**: cut from 327 ns to ~150 ns. Savings ~8%.
3. **Combined kernel speedup at seq=4096**: ~15% (weighted across all tiles).
   Target latency: 64 μs → **~54 μs**. Closes FlashInfer gap from 1.47×
   to **~1.25×**.
4. **Correctness**: cosine(new, HYP-035) = 1.0000. WMMA V is bit-equivalent
   to scalar FMA for this sum (deterministic accumulator); warp softmax uses
   the same online-softmax math in a different reduction order.

Non-goals for this hypothesis:
- Addressing WMMA_QK smem load latency (41%). Full FA-style pipelining
  pending, future HYP if needed.
- Any compute-layout change to the paged cache or the split-KV grid.

## Method

### 1. V-WMMA (single file change to v5 kernel header)

Current V accumulate is scalar: for each output dim d, sum `P[q, j] * V[j, d]`
across j. Convert to:

```cpp
// P = [16, 16] attn weights fp16 — already in scores_smem after softmax.
// V = [16, head_dim] fp16 — already in kv_smem after Phase 4 dequant (reused).
// O_part = P · V = [16, head_dim] fp16 via k_tiles=8 WMMA mma_sync calls.

::nvcuda::wmma::fragment<matrix_a, 16, 16, 16, __half, row_major> p_frag;
::nvcuda::wmma::fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
::nvcuda::wmma::fragment<accumulator, 16, 16, 16, float> o_frag_new;
// For each head_dim tile (head_dim / 16 = 8 tiles):
//   load p_frag from scores_smem (same matrix every tile; could hoist)
//   load v_frag from kv_smem + tile_offset
//   mma_sync(o_frag_new, p_frag, v_frag, o_frag_new)
// Scale by 1/d_curr (online softmax denominator) AFTER the full loop.
```

The accumulator `o_accs[bdy][odims_per_thread]` stays the same — we just
swap the inner loop (which currently does 16 scalar FMAs per output dim
per tile) for one `mma_sync` call per head_dim-tile.

Cross-tile online softmax rescale already handles the scale_old factor on
the accumulator; no change there. V is already in fp16 in `kv_smem` after
Phase 4, so no extra dequant cost.

### 2. Warp-reduced softmax

Current softmax_update is a scalar loop over tile_n=16 tokens per query head:
```cpp
float m_old = m_vals[h], d_old = d_vals[h];
float m_new = m_old;
for (int j = 0; j < 16; j++) {
    m_new = fmaxf(m_new, scores_smem[h * 16 + j] * scale);
}
float d_new = 0.f;
for (int j = 0; j < 16; j++) {
    d_new += __expf(scores_smem[h * 16 + j] * scale - m_new);
}
// Rescale o_accs[h] by expf(m_old - m_new), etc.
```

Replace with warp-parallel reduction. Each of 32 lanes is assigned one
score (lane `tx` handles token `tx % 16`); two lanes per token (lanes in
[0,15] and [16,31] hold duplicate values for mutual-exclusion-free reduction).
Butterfly reduce:

```cpp
float s = scores_smem[h * 16 + (tx % 16)] * scale;  // all 32 lanes read
float m = s;
#pragma unroll
for (int delta = 8; delta > 0; delta >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, delta));
}
// m is now max across 16 tokens, replicated on all lanes of each half-warp
float d_j = __expf(s - m);
float d = d_j;
#pragma unroll
for (int delta = 8; delta > 0; delta >>= 1) {
    d += __shfl_xor_sync(0xFFFFFFFF, d, delta);
}
// d is the denominator. All lanes have m, d, d_j (this lane's contribution).
// Write back rescaled scores (for V-WMMA consumption).
scores_smem[h * 16 + (tx % 16)] = __float2half(d_j / d);
```

log₂(16)=4 butterfly steps instead of 16 serial iterations.

### 3. Benchmark

Re-run the HYP-032 long-context sweep (seq ∈ {256, 512, 1024, 2048, 4096,
8192, 16384, 32768}) in a single Forge job via notebook SSH (cheaper than
5× batch jobs). Compare:
- Pre: current kernel numbers (from `results/hyp032/` and `results/hyp032_long_profile/`)
- Post: after V-WMMA + warp softmax

Re-run the phase probe too — the per-phase breakdown after these changes
validates the prediction at the phase level, not just end-to-end.

### 4. Correctness

- `tests/test_v5_graph.py` cosine gate (≥ 0.9999 vs eager) catches any drift.
- `tests/bench_v5_graph.py` runs a cos vs eager check per call.

## Status: rejected (warp-butterfly softmax)

## Results (Forge A100-SXM4-40GB, 2026-04-17)

Implemented the warp-butterfly softmax first, left V-WMMA for after if softmax
landed. Ran correctness (cosine=0.999996 ✓) and the long-context sweep.

**End-to-end: slower by 7–33% across seq ∈ {256..8192}**. Example at seq=4096:
64.1 μs (pre) → 76.1 μs (post), +19%. Curve:

| seq   | pre (HYP-032) | post (HYP-036 softmax) | Δ      |
|-------|--------------:|-----------------------:|-------:|
|  512  |       36.0 μs |                47.8 μs | +33%   |
| 1024  |       43.8 μs |                54.6 μs | +25%   |
| 2048  |       46.1 μs |                58.2 μs | +26%   |
| 4096  |       64.1 μs |                76.1 μs | +19%   |
| 8192  |      103.0 μs |               110.4 μs |  +7%   |

Phase probe confirms the regression is localized to softmax:

| phase        | pre   | post   | Δ     |
|--------------|------:|-------:|------:|
| softmax_update | 327 ns | **1510 ns** | **+362%** |
| (all other phases unchanged) | | | |

## Analysis

The warp-butterfly reduction is ~5× slower than the scalar unroll. Why:

**Old scalar unroll (fast)** — all 32 lanes execute the same code:
```cpp
for (int j = 0; j < 16; j++) {
    local_scores[j] = exp2f(local_scores[j] - m_vals[h]);
    d_vals[h] += local_scores[j];
}
```
The 16 iterations are *independent per lane*. The compiler unrolls, and the
A100's 16 SFUs can issue exp2f at 16-lanes-per-cycle throughput. The 16
serial dependencies exist only within each lane's accumulator chain, but the
warp's SFU pipeline hides them. Effective latency ≈ 16 × ~1 cycle = 16 cycles
for the exp2 wave across the warp.

**New warp-butterfly (slow)** — serial shuffle dependency chain:
```cpp
float lane_max = s;
for (int delta = 8; delta > 0; delta >>= 1) {
    lane_max = max(lane_max, __shfl_xor_sync(0xFFFFFFFF, lane_max, delta));
}
```
Each `__shfl_xor_sync` has ~6-cycle latency and the output feeds the next
shuffle. 4 shuffles × 6 cycles = 24 cycles minimum per reduction. Two
reductions (max + sum) = 48 cycles per bdy iteration. × 4 bdy = 192 cycles
of pure serial shuffle latency. Plus another 32 cycles (from the ~1490 ns
overhead) for the other overhead I can't immediately explain — likely extra
register pressure from the shuffle state or the per-lane guard branch.

**The core mistake:** I assumed "16 serial ops → 4 parallel ops" was a win.
But the 16 ops weren't actually serial across the warp — they were
serial *within each lane*, and the compiler + SFU pipeline already
parallelized across lanes. Shuffle reductions only help when the data is
already distributed across lanes (which it isn't here — all lanes
broadcast-read the same smem addresses).

## Prediction verdicts

| Prediction | Target | Result | Verdict |
|-----------|--------|--------|---------|
| Correctness (cos ≥ 0.9999) | ≥ 0.9999 | 0.999996 | ✓ confirmed |
| Softmax phase speedup | 327 → ~150 ns | 327 → **1510 ns** (4.6× slower) | ✗ **rejected** |
| End-to-end seq=4096 | 64 → 54 μs | 64 → 76 μs (+19%) | ✗ **rejected** |

## Decision

**Reverted the softmax change.** v5_paged stays on the scalar-unroll softmax.
The V-WMMA half of this hypothesis is NOT implemented (avoided the wasted
effort given softmax result invalidated the framing).

## Lessons

1. Warp-shuffle reductions are fast when data is distributed; slow when all
   lanes hold the same value. The access pattern in v5's softmax is
   broadcast-read — shuffle reductions add serial latency without saving
   compute.
2. "Log₂(N) parallel steps vs N serial steps" is only a win if those N
   steps are sequentially dependent in the original code. Here they were
   parallel-independent across lanes; the compiler already got the win.
3. Profile *after* changes, not just before. The phase probe made the
   regression instantly obvious (327 → 1510 ns in Phase 6 alone).

The actual big lever remains WMMA_QK (41% of kernel, 924 ns measured). That
requires multi-stage cp_async pipelining (full FA-style kernel rewrite).
Deferred to a future hypothesis when either the long-context latency gap
starts blocking production or we have kernel-rewrite time. For now, v5_paged
stays at its post-HYP-032 shape — within 1.47× of FlashInfer at seq=4096,
within 0.87× at seq=512 (beating FI).

## References

- HYP-031 (tensor-core dequant v5 kernel) — the kernel we're modifying.
- HYP-032 (warp-shuffle codebook LUT) — confirmed. Showed that
  `__shfl_sync` can replace serialized scalar ops cheaply; same trick
  applied to softmax here.
- HYP-035 (paged-native v5) — confirmed. No interaction; this HYP modifies
  post-load compute only.
- Profile data: `results/profile_v5_paged/seq-16384.md` — justification for
  this framing.
- NVIDIA CUTLASS online-softmax + WMMA sample: reference for the
  P·V WMMA layout (same as FlashAttention V accumulate).
- Phase 13c in `docs/ROADMAP.md` — this is the pragmatic continuation
  after HYP-032 closed the easy dequant wins. Remaining WMMA_QK latency
  (41% of kernel) is deferred to a separate FA-style-pipelining HYP if
  needed after this lands.
