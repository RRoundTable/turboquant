# HYP-025: Async Write Overlap for TPOT Reduction

## Hypothesis
Launching the quantize-write kernel on a separate CUDA stream will hide
the 2.6ms write overhead (0.094ms × 28 layers) by overlapping with MLP
compute, reducing TPOT from 5.75ms to ~3.5ms.

## Prediction
Write kernel overlaps with MLP compute on a separate stream.
TPOT drops from 5.75ms (1.67×) to ~3.5ms (~1.0× vs FP16).

## Method

### Trial 1: Async stream in do_kv_cache_update, sync in forward
- Write launched on `torch.cuda.Stream()` in `do_kv_cache_update`
- Sync via `wait_stream` at start of `forward()`
- Result: **6.37ms (1.85×) — WORSE**
- Cause: write and decode both memory-bandwidth bound, concurrent streams
  cause L2 cache thrashing

### Trial 2: Async stream with corrected sync point
- Write launched async in `do_kv_cache_update`
- Sync at start of `forward()` (hoping MLP ran between layers)
- Result: **6.19ms (1.80×) — WORSE + quality degradation**
- Cause: vLLM calls `do_kv_cache_update` then `forward` for the SAME
  layer back-to-back. No MLP between them. No overlap window exists.
- Quality: repeated words ("capital capital"), wrong math ("2+11=33")
  from race condition — decode reads cache before write completes.

## Profiling data (standalone kernel timing)

| Component | Time per call | × 28 layers | Notes |
|-----------|--------------|-------------|-------|
| Write kernel (K+V) | 0.094ms | 2.6ms | normalize + FWHT + quantize + pack |
| Decode kernel | 0.106ms | 3.0ms | attention with inline dequant |
| fp16→fp32 cast | 0.018ms | 0.5ms | for write kernel input |
| permute+view | 0.015ms | 0.4ms | NHD→HND cache access |
| Page table | 0.924ms | 0.9ms | (cached, only once per step) |

## Why overlap is impossible in the attention backend

```
vLLM per-layer execution order:
  [QKV projection] → [do_kv_cache_update] → [forward/attention] → [output projection] → [MLP]
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Our code runs here — no MLP between update and forward
```

The attention backend only controls `do_kv_cache_update` and `forward`.
Both are called synchronously for the same layer. The write must complete
before forward reads the cache. There is no compute to overlap with.

To overlap write with MLP, the write must be launched AFTER forward
returns and overlap with the output projection + MLP of the same layer.
This requires vLLM model-level changes (not achievable as a plugin).

## Status: rejected

Async write within the attention backend adds overhead (stream sync,
L2 contention) without any overlap benefit. The 2.3ms quantization
cost is irreducible at the plugin level.

**Final TPOT: 5.75ms (1.67× vs FP16 3.44ms).**
