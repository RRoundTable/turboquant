# TurboQuant Kernel Optimization Plan

## Current State (2026-03-31)

FlashInfer-style fused decode kernel: **correct** (cosine=1.0) and **18× slower** than SDPA after bdz=16 optimization.

| Kernel | Latency (1024 tok) | vs SDPA | Threads | Notes |
|--------|-------------------|---------|---------|-------|
| SDPA (FlashAttention) | 20.6 μs | 1.0× | 128-512 | Tensor cores, pipelined |
| TQ FlashInfer-style (bdz=16) | **373 μs** | **18×** | 256 | Correct, all tests pass |
| TQ FlashInfer-style (bdz=1) | 1739 μs | 84× | 16 | Correct, baseline |
| TQ standalone (bdz=16) | 142 μs | 6.9× | 256 | Different kernel, simpler |

## Optimization Steps Completed

| Step | What | Before → After | Result |
|------|------|---------------|--------|
| **7a. bdz=16 merge** | Multi-tz parallelism with softmax merge | 1739 → 373 μs | **4.7× speedup** |
| 7b. Precompute page offsets | Cache divmod + __ldg results in smem | 373 → 416 μs | **Negative** (smem pressure), reverted |

### 7a Bug Fixes
1. **tx merge overwrite**: all tx threads wrote o_acc to same smem offset → each tx stores at tx-dependent offset
2. **GQA smem addressing**: each ty read only its own token subset → all ty threads read ALL tokens_per_tz
3. **NaN softmax**: exp2(-inf - (-inf)) = NaN → skip when all scores -inf
4. **NaN merge**: merge with tz that has m=-inf → skip those tz groups

## Remaining Steps

| Step | What | Expected | Effort | Status |
|------|------|----------|--------|--------|
| **7c** | In-kernel FWHT | -203 μs Python overhead | Medium | Not started |
| **7d** | Inject into FlashInfer source | ~25-30 μs | High | Not started |

### 7c. In-Kernel FWHT

Current: Python FWHT for Q rotation (before kernel) + output un-rotation (after kernel) = 203 μs.

Moving both into the kernel eliminates this overhead entirely:
- Q rotation: shared-memory FWHT on Q per-head, done once before main loop
- Output un-rotation: shared-memory FWHT on o_acc after merge, before global write

For 128-dim FWHT with 8 threads (bdx=8, vec_size=8):
- 3 butterfly stages within each thread's 8 values (no communication)
- 3 butterfly stages across threads (need shared memory or warp shuffle)
- Run per 64-dim chunk (2 chunks for head_dim=128)

### 7d. Inject into FlashInfer Source

Replace `cp_async` KV load in FlashInfer's `decode.cuh` with our `dequant_load_to_smem`. This gives:
- FlashInfer's tensor cores for QK/V computation
- FlashInfer's cp_async pipeline (even though we load synchronously, the pipeline scheduling helps)
- FlashInfer's optimized tile sizes and warp scheduling

The dequant-load is ~20 lines of code. Everything else is FlashInfer's production kernel.

Theoretical lower bound: ~18 μs (3.76× less VRAM bandwidth than fp16 SDPA).

## Performance Timeline

```
Phase 3a (standalone, bdz=1):     856 μs  (41.6× vs SDPA)
Phase 6a (standalone, bdz=16):    142 μs  (6.9×)
Phase 7  (FlashInfer-style, bdz=1): 1739 μs  (84×)
Phase 7a (FlashInfer-style, bdz=16): 373 μs  (18×)    ← current
Phase 7c (+ in-kernel FWHT):       ~373 μs kernel + 0 Python  (18×, saves 203μs total)
Phase 7d (inject into FlashInfer):  ~25-30 μs  (1.2-1.5×)   ← target
Theoretical minimum:                ~18 μs  (potentially faster than SDPA)
```

## Test Results Summary

All tests pass at cosine=1.000000 with bdz=16:

| Config | Cosine |
|--------|--------|
| 1 head, 1 token | 1.000000 |
| 1 head, 4 tokens | 1.000000 |
| 1 head, 16 tokens | 1.000000 |
| 2 heads, 1 token | 1.000000 |
| 8 heads, 1 token | 1.000000 |
| 8 heads, 16 tokens | 1.000000 |
| 8kv 16qo, 1 token (GQA) | 1.000000 |
| 8kv 16qo, 16 tokens (GQA) | 1.000000 |
| 8kv 16qo, 64 tokens (GQA) | 1.000000 |
| bdz=1 vs bdz=2 (8 tokens, no GQA) | 1.000000 |
