# HYP-026: Packed 4-bit Cache (68 bytes/head, 3.76× savings)

## Hypothesis
Override `get_kv_cache_shape` to return 68 bytes/head (instead of 128)
and patch `AttentionSpec.real_page_size_bytes` to match. vLLM allocates
smaller cache → 3.76× memory savings vs fp16.

## Prediction
KV cache: ~280K tokens (vs 74K fp16 = 3.76× more).
TPOT: same or better (less memory = better cache utilization).

## Method
1. `get_kv_cache_shape` returns `(2, blocks, block_size, kv_heads, 68)`
2. Plugin patches `AttentionSpec.real_page_size_bytes` to use 68B/head
3. `entry_byte_stride=68` in kernel (tight packing, no padding)

## Result

**FAILED** — shape mismatch crash:
```
RuntimeError: shape '[2, 9250, 8, 16, 68]' is invalid for input of size 303104000
```

### Root cause

vLLM V1 has TWO independent code paths for cache sizing:

1. **Allocation path**: `_reshape_kv_cache_tensors` in `gpu_model_runner.py`
   - Computes flat buffer size from `num_blocks × block_size × kv_heads × head_size × dtype_size`
   - Uses `head_size=128` from `AttentionSpec` (model config)
   - Result: 303,104,000 bytes (128 bytes/head)

2. **Reshape path**: `get_kv_cache_shape` from the backend
   - Returns `[2, 9250, 8, 16, 68]` (68 bytes/head)
   - Result: 160,729,600 elements
   - **Does not match allocation** → crash

The `real_page_size_bytes` patch was applied but **not used** by the
allocation path. vLLM V1 sizes the raw buffer from element-level
calculations, not from `page_size_bytes`.

### What we tried

| Approach | Result |
|----------|--------|
| Override `get_kv_cache_shape` → 68B | Crash: shape mismatch |
| Patch `real_page_size_bytes` | Not used by allocation path |
| Patch `CacheDType` Literal | Works but doesn't affect allocation |
| Patch `STR_DTYPE_TO_TORCH_DTYPE` | Works but doesn't affect allocation |

### Plugin ceiling

| Metric | Plugin (fp8) | Upstream PR (custom) |
|--------|-------------|---------------------|
| Memory savings | 2× | 3.76× |
| Bytes/head | 128 (60B waste) | 68 (zero waste) |
| Cache tokens | 148K | ~280K |

### What the upstream PR needs

One change in vLLM V1's `gpu_model_runner.py`:

```python
# Current: flat buffer sized from head_size
size = num_blocks * block_size * kv_heads * head_size * dtype_size

# Fix: flat buffer sized from get_kv_cache_shape
shape = backend.get_kv_cache_shape(num_blocks, block_size, kv_heads, head_size, cache_dtype)
size = math.prod(shape) * dtype_size
```

This makes the allocation path consistent with the reshape path.

## Status: rejected (plugin limitation)

The 3.76× savings is technically correct but can't be achieved as a
vLLM plugin. Requires upstream vLLM changes to align allocation with
the backend's `get_kv_cache_shape`.
