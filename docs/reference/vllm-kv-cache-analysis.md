# vLLM KV Cache Architecture Analysis

Analysis of vLLM's KV cache system for TurboQuant integration (v1 engine).

Source: `~/workdir/vllm` on `mlsys-dgx-spark` (cloned 2026-03-26).

---

## 1. Inference Pipeline (Single Decode Step)

```
Scheduler
  │  produces: slot_mapping (where to write), block_table (where to read)
  ▼
GPUModelRunner.execute_model()
  ├─ prepare inputs (slot_mapping, block_table, attention metadata)
  ├─ set_forward_context (thread-local metadata for attention layers)
  └─ model forward
       │
       For each decoder layer:
       ├─ QKV projection → Q, K, V tensors
       ├─ RoPE applied to Q, K
       └─ Attention.forward(query, key, value)
            │
            ├─ WRITE: do_kv_cache_update(K, V, cache, slot_mapping)
            │    → quantize K/V (e.g., FP8 scale+cast)
            │    → scatter into paged cache at slot positions
            │
            └─ READ + COMPUTE: forward(Q, K, V, cache, block_table)
                 → attention kernel reads K/V from paged cache
                 → computes softmax(QK/√d)V
                 → writes output
```

Key: the **write** and **read** steps are cleanly separated. Both FlashAttention and FlashInfer backends set `forward_includes_kv_cache_update = False`, meaning KV write is a distinct operation before attention compute.

## 2. Eager Mode vs Compiled Mode

| Mode | Description |
|------|-------------|
| **Eager** (`CompilationMode.NONE`) | Standard PyTorch execution, no torch.compile, no CUDA graphs |
| **Piecewise** (`VLLM_COMPILE` + CUDA graphs) | Model compiled in sections; attention ops still execute in eager PyTorch between graph boundaries |
| **Full compile** | Entire model traced by torch.compile |

Even in compiled mode, attention ops are registered as opaque custom ops — torch.compile does not trace inside them. This means a custom attention backend works the same regardless of compilation mode.

## 3. Three Integration Hooks

### Hook 1: KV Cache Spec

Controls how much memory each KV cache page uses.

- Subclass `FullAttentionSpec` in `vllm/v1/kv_cache_interface.py`
- Override `page_size_bytes` to reflect compressed size
- This is how vLLM calculates `num_blocks` from the GPU memory budget

For TurboQuant: a 3-bit page is ~5x smaller than fp16, so ~5x more blocks fit in memory.

### Hook 2: Attention Backend

Controls how K/V are written to and read from cache.

- Subclass `AttentionBackend` + `AttentionImpl` in `vllm/v1/attention/backend.py`
- `get_kv_cache_shape()` — return the quantized tensor shape
- `do_kv_cache_update(key, value, kv_cache, slot_mapping)` — quantize K/V before writing to cache
- `forward(query, key, value, kv_cache, ...)` — dequantize from cache before (or during) attention
- `supported_kv_cache_dtypes` — register the new cache dtype
- Register in `AttentionBackendEnum` in `vllm/v1/attention/backends/registry.py`

### Hook 3: Scale/Parameter Management

Controls how quantization parameters are stored on the model.

- Subclass `BaseKVCacheMethod` in `vllm/model_executor/layers/quantization/kv_cache.py`
- `create_weights()` — register codebook, Hadamard signs, norms as layer parameters
- `process_weights_after_loading()` — move to GPU, store on the attention layer

## 4. FP8 KV Cache (Reference Pattern)

FP8 is the closest existing quantization. The flow:

1. **Config:** `--kv-cache-dtype fp8`
2. **Allocation:** Cache allocated as int8 buffer, reinterpreted as `torch.float8_e4m3fn`
3. **Write:** `do_kv_cache_update()` → divide by `k_scale`/`v_scale` → FP8 cast on store
4. **Read:** Attention kernel receives `k_descale`/`v_descale` → dequant inside kernel

**Key difference from TurboQuant:** FP8 is a simple scale+cast. TurboQuant needs codebook lookup + inverse Hadamard rotation + norm rescaling — cannot use the same cast mechanism.

## 5. TurboQuant Eager-Mode Integration

### What "eager mode" means

The dequantization runs as a **separate GPU kernel** before the attention kernel. The paged KV cache stores quantized data (real memory savings), but there's extra kernel launch overhead vs kernel fusion.

### Write path (quantize K/V → store in cache)

Intercept `do_kv_cache_update()`:

```
Attention layer produces K, V (fp16)
  → TurboQuant quantize: K, V → indices (uint8) + norms (float32)
  → scatter quantized data into paged cache via slot_mapping
```

This replaces the standard `reshape_and_cache_flash` call. The quantized cache stores uint8 indices + float32 norms per vector instead of fp16 values.

### Read path (dequantize from cache → attention)

Intercept `forward()` before calling the attention kernel:

```
Read quantized pages from cache using block_table
  → TurboQuant dequantize: indices + norms → fp16 K, V
  → pass dequantized fp16 tensors to standard FlashAttention/FlashInfer kernel
```

Two options for the dequantized output:
1. **Contiguous buffer:** Dequant all active KV into a flat `[total_tokens, heads, head_dim]` tensor → use non-paged attention interface
2. **Page-aligned buffer:** Dequant into a temporary paged fp16 cache → reuse standard paged attention

### Memory layout

```
Standard fp16:    [num_blocks, block_size, num_kv_heads, head_dim] × fp16
TurboQuant:       [num_blocks, block_size, num_kv_heads, padded_dim] × uint8  (indices)
                + [num_blocks, block_size, num_kv_heads]             × float32 (norms)
```

At 3 bits: each coordinate stored as uint8 (1 byte) vs fp16 (2 bytes), plus a float32 norm per vector. Effective compression depends on `head_dim`:
- head_dim=128: `128 × 1 + 4 = 132 bytes` vs `128 × 2 = 256 bytes` → ~1.9x
- True 3-bit packing (4 values per 12 bits): would yield ~5x but requires bit-packing kernels

### What eager mode proves

1. Correctness — quantized KV cache produces correct generation output
2. Quality — perplexity impact is within acceptable bounds
3. Memory savings — more sequences fit in GPU memory
4. Functional integration — works with vLLM's scheduler, page manager, prefix caching

What it does NOT prove: throughput. The extra dequant kernel launch adds latency. FlashInfer kernel fusion (Roadmap item 2) eliminates this.

## 6. Key Files

| File | Purpose |
|------|---------|
| `vllm/v1/kv_cache_interface.py` | KVCacheSpec — page size calculation |
| `vllm/v1/core/block_pool.py` | Block pool — page allocation/free |
| `vllm/v1/worker/gpu/attn_utils.py` | Physical tensor allocation + reshape |
| `vllm/model_executor/layers/attention/attention.py` | Attention layer — forward dispatch |
| `vllm/v1/attention/backend.py` | AttentionBackend + AttentionImpl interfaces |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention backend (reference) |
| `vllm/v1/attention/backends/flashinfer.py` | FlashInfer backend (reference) |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | KV cache write kernel |
| `vllm/model_executor/layers/quantization/kv_cache.py` | FP8 scale management (reference) |
| `vllm/config/cache.py` | CacheDType config |
| `vllm/v1/attention/backends/registry.py` | Backend registration |
