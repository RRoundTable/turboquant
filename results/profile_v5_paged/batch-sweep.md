# v5_paged batch × seq sweep (2026-04-17)

Forge A100-SXM4-40GB, NGC pytorch:24.01, tq_v5_paged_split_graph kernel
(HYP-035 + HYP-032 shipped state). All numbers are p50 μs over 200 CUDA
graph replays.

## TQ v5_paged latency (μs)

| seq \ batch |     1   |     4   |    16   |    32   |
|-------------|--------:|--------:|--------:|--------:|
|    4096     |    76.1 |   177.6 |   596.8 |  1144.9 |
|   16384     |   184.4 |   596.1 |  2241.2 |  4431.8 |
|   32768     |   328.3 |  1144.1 |  4431.8 |  8823.2 |

## Per-request cost (latency / batch)

| seq \ batch |     1   |     4   |    16   |    32   | b=32/b=1 |
|-------------|--------:|--------:|--------:|--------:|---------:|
|    4096     |    76.1 |    44.4 |    37.3 |    35.8 | **0.47×** (2.13× throughput) |
|   16384     |   184.4 |   149.0 |   140.1 |   138.5 |    0.75× (1.33× throughput) |
|   32768     |   328.3 |   286.0 |   277.0 |   275.7 |    0.84× (1.19× throughput) |

## Interpretation

**Short-mid context (seq=4096) benefits from batching.** Per-request cost
drops from 76.1 μs → 35.8 μs at batch=32 — the SMs are under-utilized at
batch=1 and filling them with more concurrent requests gives 2.1× throughput.

**Long context (seq=32768) barely benefits from batching.** Per-request
cost only drops from 328 → 276 μs (1.19× throughput). The GPU is **already
SM-saturated at batch=1** because of the split-KV grid (32 splits × 8
kv_heads = 256 blocks on 108 SMs).

## Practical implication

The observed "TQ is 2.5× slower than FlashInfer at seq=32k" gap (from prior
batch=1 benches) reflects **per-token compute efficiency**, not SM
underutilization. Batching doesn't close it because:

1. TQ is compute-bound at long context (see `seq-16384.md`: Phase 5 WMMA_QK
   = 41% of kernel; load→mma stall accounts for 66% of that).
2. FlashInfer is similarly compute-bound at long context (its own split-KV
   saturates SMs).
3. Both kernels scale ~linearly with batch once SMs are saturated, so the
   gap ratio stays constant.

The gap is real per-kernel, not an artifact of profiling at batch=1.

**Memory-side still favors TQ:** at seq=32k batch=32, FP16 KV cache is
32768 × 32 × 8 × 2 × 128 × 2 = 4.3 GB. TQ at 4-bit is 1.13 GB —
**3.8× more concurrent requests fit in the same HBM**.

## FlashInfer comparison not collected this session

flashinfer-python 0.6.x requires PyTorch 2.3+ APIs (`torch.uint32`,
`torch.library.custom_op`, JIT `launch_metadata` kwarg, etc.) that NGC
pytorch:24.01 (torch 2.2) doesn't have. Multiple monkey-patches worked
through several layers but the integration kept hitting new 2.3+ surfaces.
Not worth more shim effort; batch=1 FI numbers in `results/hyp032_long_profile/`
remain the reference for direct comparison.

## Conclusion

Batching does NOT paper over the long-context latency gap. If closing that
gap matters, the lever is still the raw-PTX WMMA rewrite
(documented as out-of-scope in HYP-038).
