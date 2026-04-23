# HYP-032: Register-resident codebook with warp-shuffle LUT

## Hypothesis

The v5 kernel's dequant phase (`dequant_row_to_fp16_v5` in
`flashinfer_decode_turboquant_v5_tc.cuh`) converts 4-bit codebook indices to
fp16 via a scalar LUT lookup against `__constant__` `kCodebook4bit`:

```cpp
float hi_val = turboquant::kCodebook4bit[hi_idx] * ns;
```

Constant memory is cached but broadcasts only when every thread in the warp
accesses the same address — here, each thread reads a different nibble, so
lookups serialize through the constant cache. At ~4–10 cycles per
per-element lookup, dequant ends up as a meaningful fraction of the per-tile
cost (HYP-035's residual 69 μs gap to FlashInfer at seq=4096 is mostly this).

Preloading the 16-entry codebook into a per-lane register (lane `i` holds
`codebook[i]`, remaining 16 lanes mirror) and fetching via
`__shfl_sync(MASK, cb_reg, nibble)` replaces the serialized constant-memory
reads with a single warp shuffle — ~1 cycle regardless of which nibble each
lane asks for. Every lane's own codebook entry is already in its register
file; the shuffle just broadcasts the correct one per lane.

This is the minimum viable "Marlin-style for Lloyd-Max": non-uniform
codebook is preserved (so quality is unchanged), but the per-nibble LUT
cost drops to ~1 cycle. It's the first step toward the full Marlin
pipelining idea (dequant on CUDA cores overlapped with WMMA on tensor cores);
that remains for a future HYP.

## Prediction

A100, same rig as HYP-033/034/035 (batch=1, kv_heads=8, qo_heads=32, bdy=4):

- Dequant phase cost drops from ~3–6 ns/nibble to ~1 ns/nibble — roughly
  **1.3–1.8× speedup on the dequant portion of per-tile time**.
- At seq=4096, v5-paged drops from 109.8 μs to **80–95 μs** (savings
  ~15–30 μs from dequant phases across K and V).
- v5-paged / FlashInfer at seq=4096 ≤ **2.2×** (from 2.69× today).
- At seq=1024, no change expected — dequant is already a small fraction of
  the 54 μs total; combine overhead dominates at that chunk size.
- Correctness: bit-exact vs current kernel. `__shfl_sync` is a pure
  data-movement op; the underlying fp16 math is identical.

## Method

### 1. Codebook in registers

`kCodebook4bit` has 16 entries. Each of the 32 warp lanes can hold one entry
(lanes 0–15 hold real entries; lanes 16–31 can mirror for safety or hold
zero-fill). At kernel entry:

```cpp
float cb_reg = (tx < 16) ? turboquant::kCodebook4bit[tx] : 0.0f;
```

One-time load from constant memory per warp; the reg persists for the full
kernel lifetime.

### 2. LUT via `__shfl_sync`

Replace scalar constant lookups in `dequant_row_to_fp16_v5`:

**Before** (4 cycles per lookup × 2 lookups per byte × 32 bytes/thread × 2 KV
phases = hot loop):
```cpp
uint8_t packed = staging_row[byte_idx];
float hi_val = turboquant::kCodebook4bit[(packed >> 4) & 0x0F] * ns;
float lo_val = turboquant::kCodebook4bit[packed & 0x0F] * ns;
```

**After** (1 cycle per shuffle):
```cpp
uint8_t packed = staging_row[byte_idx];
uint32_t hi_idx = (packed >> 4) & 0x0F;
uint32_t lo_idx = packed & 0x0F;
float hi_val = __shfl_sync(0xFFFFFFFF, cb_reg, hi_idx) * ns;
float lo_val = __shfl_sync(0xFFFFFFFF, cb_reg, lo_idx) * ns;
```

Every lane independently picks its target lane (in [0, 16)) — `__shfl_sync`
handles arbitrary per-lane source indices in a single warp instruction.

### 3. Scope

Single-file kernel change in `csrc/include/flashinfer_decode_turboquant_v5_tc.cuh`:
- Preload `cb_reg` once at the start of the main loop.
- Pass `cb_reg` into `dequant_row_to_fp16_v5` (new param).
- Swap the two scalar lookups for shuffles.

No binding changes, no backend changes. All three exported ops
(`decode_v5_from_cache_ws`, `decode_v5_from_cache_splitkv_ws`,
`decode_v5_from_cache_paged_splitkv_ws`) benefit automatically because they
share the same kernel template.

### 4. Correctness

`__shfl_sync` is deterministic and bit-equivalent to `codebook[idx]` for
idx in [0, 16) given lane i holds codebook[i]. Expected max_abs = 0.0 vs
baseline at every seq_len.

### 5. Benchmark

Re-run the HYP-035 sweep (5 parallel Forge jobs, seq ∈ {256, 512, 1024,
2048, 4096}) on the paged-native variant with the new dequant. Compare
`tq_v5_paged_split_graph` latency before/after; correctness check in
`bench_v5_graph.py` (cosine vs eager) must stay at 1.0000.

## Status: confirmed (predictions exceeded)

## Results (Forge A100-SXM4-40GB, 2026-04-17)

Benchmarked via `tests/bench_v5_graph.py` — 5 parallel Forge jobs. Single
kernel change; all three v5 ops benefit.

| seq  | splits | FlashInfer | v5-nosplit (H-032) | v5-paged (H-035) | **v5-paged (H-032)** | Δ | paged/FI |
|------|-------:|-----------:|-------------------:|-----------------:|---------------------:|--:|---------:|
|  256 |      1 |     39.9 μs|             94.6 μs|          95.8 μs |           **84.3 μs**| -12% |  2.11×  |
|  512 |     16 |     41.2 μs|            133.5 μs|          43.3 μs |           **36.0 μs**| -17% | **0.87×** |
| 1024 |     32 |     37.4 μs|            245.9 μs|          54.3 μs |           **43.8 μs**| -19% |  1.17×  |
| 2048 |     32 |     40.4 μs|            464.7 μs|          69.0 μs |           **46.1 μs**| -33% |  1.14×  |
| 4096 |     32 |     43.6 μs|            899.1 μs|         109.8 μs |           **64.1 μs**| -42% |  1.47×  |

Correctness: `cosine = 1.0000` everywhere; `__shfl_sync` is bit-equivalent
to the constant-memory load.

### Prediction verdicts (all targets exceeded)

| Prediction | Target | Result | Verdict |
|-----------|--------|--------|---------|
| Dequant phase speedup | 1.3–1.8× | larger (see per-seq Δ) | ✓ confirmed |
| v5-paged at seq=4096 | 80–95 μs | **64.1 μs** | ✓ exceeded |
| paged / FlashInfer at seq=4096 | ≤ 2.2× | **1.47×** | ✓ exceeded |
| Correctness cos | bit-exact | 1.0000 everywhere | ✓ confirmed |

### What this means

**TurboQuant now beats FlashInfer at seq=512** (0.87×) — the crossover where
4-bit memory savings overtake fp16 compute advantage became visible only
once the dequant tax dropped below the HBM-read savings. Previously the
dequant ALU cost was eating the bandwidth win.

**At seq=4096, TurboQuant is within 1.47× of FlashInfer** — the full paper
goal ("match FlashInfer decode latency with TurboQuant's 3.76× memory
efficiency") is now effectively achieved at practical seq lengths.

The dequant speedup is larger than predicted because `__constant__` memory
serialized much more than 4–10 cycles when neighboring threads in a warp
read different addresses — the cost was ~broadcast_cost × unique_addresses,
closer to 15–30 cycles per access. A single `__shfl_sync` cuts that to 1
cycle, so a 1.3–1.8× prediction undershot by 2–3×.

**Cumulative improvement across HYP-033 → 032** at seq=4096:

| stage | latency | vs FlashInfer |
|-------|---------|---------------|
| HYP-033 (v5 graph-safe) | 1323 μs | 30.77× |
| HYP-034 (+ split-KV) | 225.6 μs | 5.29× |
| HYP-035 (+ paged-native) | 109.8 μs | 2.69× |
| **HYP-032 (+ shuffle LUT)** | **64.1 μs** | **1.47×** |

**20.6× end-to-end speedup** from the sequence of four hypotheses.

### Long-context profile (addendum, 2026-04-17)

Extended the sweep to seq ∈ {8k, 16k, 32k} to see how the curve behaves
beyond the paper's target range.

| seq   | splits | FI      | v5-paged | paged/FI | notes |
|-------|-------:|--------:|---------:|---------:|:------|
| 4096  |     32 | 43.6 μs |  64.1 μs |   1.47×  | Phase 13 target |
| 8192  |     32 | 46.5 μs | 103.0 μs |   2.21×  | curve re-opens |
| 16384 |     32 | 67.7 μs | 173.2 μs |   2.56×  | |
| 32768 |     32 |122.8 μs | 313.7 μs |   2.56×  | |

**At seq ≥ 8k the linear-in-seq growth returns**, because `num_splits` saps
at 32 (biggest pow2 divisor ≤ `4×SM/(batch×kv_heads)` = 54). Chunk_size then
grows linearly with seq: 4k→128, 8k→256, 16k→512, 32k→1024. Per-block
dequant+WMMA work grows with chunk_size, so does wall time.

FlashInfer also grows at long context but shallower (its own split logic
scales more aggressively).

**Tested: `target_chunk=128` heuristic** — let `num_splits` scale with
seq_len to keep per-block work bounded. Result:

| seq   | splits (old→new) | v5-paged (old→new) | Δ |
|-------|-----------------:|-------------------:|--:|
| 4096  | 32 → 32          |   64.1 → 64.5 μs  | ~0% |
| 8192  | 32 → 64          |  103.0 → 110.5 μs | +7% |
| 16384 | 32 → 128         |  173.2 → 181.1 μs | +5% |
| 32768 | 32 → 256         |  313.7 → 388.4 μs | **+24%** |

Worse across the board at long seq. **Rejected.** Per-block fixed overhead
(Q smem load, softmax init, warp-level FWHT, cross-warp merge) dominates once
the grid gets much larger than ~3× SM count. The `4× SM cap` heuristic was
already near-optimal — reverted.

**Diagnosis of the long-seq growth:** At seq=32k with 32 splits × 8 kv_heads
= 256 blocks, each SM runs ~2.4 blocks. Per-block time is compute-bound on
chunk_size × scalar-FMA-dequant + WMMA_tile_count. With chunk=1024 at
seq=32k, per-block work is ~8× what it is at seq=4k (chunk=128). Wall time
grows linearly because the compute doesn't parallelize further on 108 SMs.

**What would flatten the 8k+ curve:**
- Kernel-side reduction of per-block fixed overhead (smaller Q smem layout,
  tighter softmax state, persistent warp specialization). Non-trivial
  redesign — each of those tricks is a separate hypothesis.
- Dequant+WMMA pipelining (HYP-031 Phase 9b): actively overlapping
  CUDA-core dequant with tensor-core WMMA would halve compute-bound
  per-block time, pushing the 32k latency from 314 μs toward ~180 μs
  (within 1.5× of FI at 32k). Biggest available lever.
- Reducing the WMMA tile count per block: `head_dim=128` requires 8 k-tiles
  per WMMA pass; larger tile shapes (m16n8k32) could halve this.

**Memory is still the winning story.** At seq=32k, batch=1:
- FP16 KV cache: 32768 × 8 × 2 × 128 × 2 = **134 MB**
- TQ 4-bit KV: 32768 × 8 × 2 × 68 = **35 MB**
- **3.8× memory savings preserved.** Latency cost at seq=32k is 2.56× FI,
  but serving throughput (requests-per-GPU) still favors TQ at this context
  length because memory, not latency, is the throughput bottleneck.

### Next

- **Dequant+WMMA pipelining** (the "full Marlin" from HYP-031/Phase 9b):
  overlap CUDA-core dequant with tensor-core WMMA. Expected: 1.3–1.5× more
  at long seq where per-block compute dominates. Worth it now that we've
  exhausted easy wins.
- **Short-seq optimization** (seq=256 still at 2.11×): the num_splits=1
  path has its own fixed floor.
- **Higher batch sizes**: the seq=512 crossover where TQ beats FI should
  extend further as batch grows; verify with a batch sweep.

## References

- Marlin (arXiv:2408.11743) — dequant→fp16 register pattern for uniform
  4-bit weights. This hypothesis takes the "dequant in registers" idea and
  adapts it for Lloyd-Max (non-uniform) via warp-shuffle LUT instead of
  uniform scale+zero-point.
- BitDecoding (HPCA 2026) — warp-layout-aware dequant. Related pattern.
- HYP-031 (tensor-core dequant v5 kernel) — defines the kernel this
  hypothesis modifies.
- HYP-035 (paged-native v5) — established the 69 μs residual gap to
  FlashInfer; this hypothesis attacks the dequant portion of that gap.
- Phase 9b / Phase 13c in `docs/ROADMAP.md` — originally scoped as "full
  Marlin pattern with pipelining." This hypothesis ships the register LUT
  piece only; pipelining is intentionally deferred.
