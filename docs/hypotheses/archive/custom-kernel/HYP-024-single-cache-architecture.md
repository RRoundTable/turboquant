# HYP-024: Single Quantized Cache Architecture

## Hypothesis
Replacing dual storage (fp16 + uint8) with a single uint8 KV cache will reduce memory
from 1.27× to 0.53× of FP16 baseline (2× savings) while maintaining decode quality.
Prefill uses FlashAttention with fresh fp16 K,V (no cache read needed for new tokens).

## Prediction
- Memory: 1.27× → 0.53× (2× savings)
- Prefill TTFT: same as FP16 (FA with fresh K,V)
- Decode TPOT: same as current (5.59ms, v4 kernel unchanged)
- Quality: 5/5 correct

## Method
1. `kv_cache_dtype="fp8"` for uint8 allocation
2. `do_kv_cache_update`: quantize to uint8 only, skip fp16 write
3. `forward` prefill: call `flash_attn_varlen_func` with fresh fp16 K,V
4. `forward` decode: parse uint8 cache, call v4 kernel
5. Remove all legacy dual-storage code

## Status: pending
