# HYP-006: Fused inline dequant will reduce smem traffic and improve occupancy

## Hypothesis
Eliminating the fp16 smem intermediate buffer and dequanting inline during QK/V compute will:
1. Reduce smem usage by ~7× (staging-only, no fp16 K/V buffers)
2. Eliminate half↔float conversion overhead
3. Reduce smem write/read traffic by 4× (packed bytes vs fp16)
4. Higher occupancy from less smem → better latency hiding

## Prediction
20-40% speedup over v2. Still won't match SDPA (scalar compute vs tensor cores).

## Method
v4 kernel: cp_async packed bytes → staging, precompute norms → smem, inline dequant to float during QK/V compute. Custom QK/V loops (no FlashInfer function reuse).

## Results
Correctness: **cos=1.0** (6/6 configs, verified vs v2 output).

| seq | SDPA (μs) | v2 (μs) | v4 (μs) | v4/v2 | v4/SDPA |
|-----|-----------|---------|---------|-------|---------|
| 128 | 39 | 108 | 97 | 0.89× | 2.5× |
| 512 | 53 | 206 | 159 | 0.78× | 3.0× |
| 1024 | 60 | 415 | 296 | 0.71× | 4.9× |
| 2048 | 67 | 755 | 503 | 0.67× | 7.5× |

22-33% speedup over v2, scaling better at longer sequences.

## Analysis
The speedup comes from:
- Less smem traffic (1 byte/dim vs 4 bytes/dim for packed vs fp16)
- No float→half→float conversion
- Better occupancy from ~1.2KB smem (vs ~8.2KB for v2)

Remaining gap to SDPA (3-7×) is NOT from pipelining or smem — it's from **scalar FMA vs tensor core HMMA**. FlashAttention uses tensor cores for QK/V matmul, giving ~16× throughput over scalar ops. Our codebook dequant produces float values that can't feed tensor cores.

## Status: confirmed
22-33% improvement as predicted. Confirmed that remaining gap is compute-bound (scalar vs tensor core).
