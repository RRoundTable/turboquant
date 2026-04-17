# HYP-037: Split WMMA_QK accumulator to break serial c_frag dependency

## Hypothesis

Phase probe split the WMMA_QK phase into load-only vs mma-only and showed the
bottleneck is **mma serial dependency**, not smem loads:

| phase | cycles | ns |
|---|---:|---:|
| 5a_load_only (16 load_matrix_sync) | 163 | 116 |
| 5b_mma_only (8 mma_sync, single c_frag) | **1586** | **1125** |

The current kernel accumulates the 8 k-tiles of QK through a single accumulator
fragment:

```cpp
fragment<accumulator, 16, 16, 16, float> c_frag;
fill_fragment(c_frag, 0.0f);
for (kt = 0; kt < 8; kt++) {
    load_matrix_sync(a_frag, q_smem + kt*16, head_dim);
    load_matrix_sync(b_frag, kv_smem + kt*16, head_dim);
    mma_sync(c_frag, a_frag, b_frag, c_frag);  // ← c_frag serial dep
}
```

Every `mma_sync` reads c_frag as input and writes it as output, so the 8
mma_syncs form a single serial chain. Per-mma latency measured at ~200
cycles (6× higher than the documented ~16 cycle throughput), likely due to
scoreboard/register-dependency stalls on c_frag.

Using **N independent accumulator fragments** (one per chunk of k_tiles)
breaks the chain into N shorter chains running in parallel. At end, reduce.

```cpp
fragment c_frag[N];
for (n = 0; n < N; n++) fill_fragment(c_frag[n], 0.0f);
for (kt = 0; kt < 8; kt++) {
    // k-tile kt accumulates into c_frag[kt % N]
    mma_sync(c_frag[kt % N], a_frag, b_frag, c_frag[kt % N]);
}
// Merge: c_frag[0] += c_frag[1] + ... + c_frag[N-1]
```

For N=2: 2 chains of 4 mma_syncs each, plus 1 merge. For N=4: 4 chains of 2.
For N=8: 8 independent single-mma chains (no dep at all), plus 7 merges.

## Prediction

A100, same rig as HYP-032 probe (bs=1, kv_heads=8, bdy=4, head_dim=128):

**Phase 5 (WMMA_QK) cycle reduction:**
- N=1 (current): 1586 cycles (baseline)
- N=2: **~900 cycles** (4 mma latency + merge overhead)
- N=4: **~500 cycles** (2 mma latency + 3 merges)
- N=8: **~350 cycles** (1 mma latency + 7 merges; register-heavy)

**Best-case total kernel speedup (at N=4):**
- Phase 5: 924 → ~350 ns (save ~575 ns per tile)
- All other phases unchanged (~1320 ns)
- Per-tile total: 2245 → ~1670 ns (~26% reduction)
- End-to-end at seq=4096: 64 → **~48 μs** (matches FlashInfer's 44 μs)
- End-to-end at seq=16384: 173 → **~130 μs** (vs FI 68 μs, gap shrinks to 1.9×)
- End-to-end at seq=32768: 314 → **~230 μs** (vs FI 123 μs, gap shrinks to 1.9×)

**Register pressure risk:**
- N=2: +1 c_frag = 8 fp32 regs × 32 lanes = 256 regs extra. Minimal.
- N=4: +3 c_frag = 768 regs extra. Watch for spills.
- N=8: +7 c_frag = 1792 regs extra. Likely hits spill limit at `-O3`.

Start with N=4, measure register count (`-Xptxas -v`), fall back to N=2 if
spills appear.

**Correctness:** bitwise-equivalent sum rearrangement (different order but
same fp32 additions, which ARE associative-ish at this scale — cosine
should still be ≥ 0.9999).

## Method

### 1. Kernel change (single file)

`csrc/include/flashinfer_decode_turboquant_v5_tc.cuh`, Phase 3 (WMMA QK block
around line 413):

```cpp
constexpr uint32_t kAccN = 4;  // number of parallel accumulators
nvcuda::wmma::fragment<accumulator, 16, 16, 16, float> c_frag[kAccN];
#pragma unroll
for (uint32_t n = 0; n < kAccN; n++) fill_fragment(c_frag[n], 0.0f);

#pragma unroll
for (uint32_t kt = 0; kt < k_tiles; kt++) {
    load_matrix_sync(a_frag, q_smem + kt * V5_WMMA_K, head_dim);
    load_matrix_sync(b_frag, kv_smem + kt * V5_WMMA_K, head_dim);
    mma_sync(c_frag[kt % kAccN], a_frag, b_frag, c_frag[kt % kAccN]);
}

// Merge: c_frag[0] += c_frag[1..kAccN-1]
#pragma unroll
for (uint32_t n = 1; n < kAccN; n++) {
    #pragma unroll
    for (uint32_t i = 0; i < c_frag[0].num_elements; i++) {
        c_frag[0].x[i] += c_frag[n].x[i];
    }
}

store_matrix_sync(scores_smem, c_frag[0], V5_WMMA_N, mem_row_major);
```

No change to the load phase, no change to smem layout. Only the accumulator
chain is restructured.

### 2. Register pressure check

Add `-Xptxas -v` to the JIT compile flags temporarily, inspect the "Used N
registers, M bytes spill" line. Target: 0 bytes spill at N=4. If spill > 0,
try N=2.

### 3. Benchmark

Re-run the HYP-032 long-context sweep in the notebook (same process model):
seq ∈ {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}. Compare
`tq_v5_paged_split_graph` latency pre/post.

Run the phase probe against the new kernel to verify phase 5 cycle count
drops as predicted (baseline 1586 → target ~500 at N=4).

### 4. Correctness

`tests/test_v5_graph.py` cosine gate (≥ 0.9999) + per-call cosine in
bench_v5_graph.py must stay at 1.0 or very close. fp32 additions reordered;
float accumulation non-associativity should be negligible at this
magnitude.

## Status: rejected (measurement artifact in prior probe)

## Results (Forge A100-SXM4-40GB, 2026-04-17)

Implemented N=4 parallel c_frag accumulators in the real v5 kernel.
Correctness: cos ≥ 0.999996 at seq ∈ {256, 1024, 4096, 16384}. A/B bench
on the same Forge pod:

| seq   | N=1 baseline | N=4 (HYP-037) | Δ     |
|-------|-------------:|--------------:|------:|
|  1024 |      55.8 μs |       55.7 μs |  0%   |
|  4096 |      76.6 μs |       77.1 μs | +0.6% |
| 16384 |     184.5 μs |      185.1 μs | +0.3% |

Flat — no measurable benefit or regression.

## Why the prior probe misled me

The initial phase probe measured **mma-only** (preloaded a_frag, b_frag,
no per-iter load) and showed 192 cycles (N=1) → 65 cycles (N=4), a 3×
speedup. I took that as evidence that parallel c_frag would help Phase 5.

It didn't, because the interleaved pattern (what the real kernel actually
does) behaves completely differently. Extended probe:

| variant | cycles | ns |
|---------|-------:|---:|
| 5a load_matrix_sync only (16 loads) | 163 | 116 |
| 5b mma_sync only, N=1 (preloaded) | 193 | 137 |
| 5d mma_sync only, N=4 (preloaded) | 65 | 46 |
| **5g interleave N=1** (load → load → mma, 8×) | **1044** | **740** |
| **5f interleave N=4** | **1056** | **749** |

With real interleaved load+mma, **N=4 is identical to N=1 (within noise)**.
The critical path is the data dependency `load a_frag → load b_frag → mma
(uses a_frag, b_frag)`. Each mma can't issue until its a_frag/b_frag are
loaded from smem (~90 cycles latency × 8 iterations). Splitting the c_frag
chain doesn't help because mmas aren't stalling on c — they're stalling
on a and b.

## Prediction verdicts

| Prediction | Target | Result | Verdict |
|-----------|--------|--------|---------|
| Correctness (cos ≥ 0.9999) | ≥ 0.9999 | 0.999996 | ✓ confirmed |
| 0 register spill at N=4 | 0 bytes spill | 0 bytes | ✓ confirmed |
| N=4 mma-only speedup over N=1 | 3× | 3× | ✓ confirmed (but wrong target) |
| N=4 real-kernel speedup | ~26% at seq=4096 | 0% | ✗ **rejected** |
| End-to-end seq=4096 | 64 → ~48 μs | no change | ✗ rejected |

## Lessons

1. **Probe what the real kernel does, not simplified analogs.** My mma-only
   probe removed the load→mma dependency by preloading fragments. That
   dependency IS the bottleneck; removing it for measurement pointed at
   the wrong target.
2. A `load_only` probe + `mma_only` probe can't predict `load+mma_interleaved`
   latency. The dependency between them is the interesting thing.
3. Smem load→compute dependency chain is the real bottleneck: 1044 cycles
   = ~163 load + ~193 mma + ~688 stall waiting for data. The stall is the
   ~75% of Phase 5 time.

## Decision

**Reverted the c_frag fanout change.** Kernel stays on single c_frag.

The real fix for Phase 5 WMMA_QK is **software-pipelined loads** — issue
tile kt+1's load_matrix_sync during tile kt's mma_sync to hide the load
latency. That's HYP-038. Requires 2 pairs of a/b fragment variables
(double-buffer) and explicit loop unrolling.

This is now the second "looked good in isolation, did nothing in the real
kernel" rejection (HYP-036 was warp-butterfly softmax). Common thread: the
interplay between phases matters more than each phase in isolation.

## References

## References

- HYP-031 (v5 tensor-core kernel) — the kernel being modified.
- HYP-032 (warp-shuffle codebook LUT) — confirmed. Unrelated phase.
- HYP-036 (warp-butterfly softmax) — REJECTED. Lesson: profile after
  changes, not just predict. This hypothesis is narrowly scoped to the
  one measured bottleneck (mma dependency chain) and has a cheap probe
  gate before full commit.
- Profile data: `results/profile_v5_paged/seq-16384.md` — original phase
  breakdown. This hypothesis addresses the 41%-of-kernel WMMA_QK phase.
- Split WMMA probe data (this doc, above): isolates mma as the dominant
  cost within phase 5.
- NVIDIA Ampere WMMA programming guide: suggests multi-accumulator pattern
  for k-dim accumulation to expose tensor core throughput.
- FlashAttention kernel: uses multiple accumulator fragments per warp
  (one per query-row chunk) — same general principle, different motivation
  there (query parallelism, not k-dim).
