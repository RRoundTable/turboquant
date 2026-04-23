# FlashInfer Decode Kernel — TurboQuant Injection Plan

Analysis of where and how to inject TurboQuant dequantization into FlashInfer's paged decode attention kernel.

---

## Current Data Flow

```
Global Memory (DTypeKV)
  ──[cp_async (raw byte copy)]──►
Shared Memory (DTypeKV)
  ──[cast_load (type conversion)]──►
Registers (float)
  ──[compute_qk / update_local_state]──►
Output
```

`cp_async` is a hardware async copy — no transformation possible. Type conversion (FP8→float, FP16→float) only happens at the smem→register boundary via `vec_t::cast_load()`.

## Paged KV Layout

```
k_data / v_data: flat tensor [max_num_pages, num_heads, page_size, head_dim]  (HND)
                                    or [max_num_pages, page_size, num_heads, head_dim]  (NHD)

Page lookup: indices[indptr[batch_idx] .. indptr[batch_idx+1]] = physical page IDs
Element offset: page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx
```

`protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx)` does `__ldg(indices + page_iter)` to get the physical page, then computes the element offset. Out-of-bounds reads return offset 0 (harmless).

## Thread/Tile Parameters (HEAD_DIM=128, DTypeKV=fp16)

```
vec_size = 8          (elements per thread per load)
bdx = 16              (threads covering head_dim: 16 × 8 = 128)
bdy = GROUP_SIZE      (GQA group, typically 1-8)
bdz = 128/(bdx*bdy)  (KV chunk parallelism, e.g. 8 for GQA=1)
tile_size_per_bdx = 4 (KV rows per bdx group per iteration)
num_stages_smem = 2   (double buffering on SM80+)
```

Each `cp_async` loads 128 bits (16 bytes) = 8 fp16 elements per thread per call.

## KV Loading Code (The Injection Points)

### Preload phase (lines 492-517)

```cpp
// Each thread loads tile_size_per_bdx rows of K and V
for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    kv_offset[j] = kv_offset_smem[...] + tx * vec_size;  // element offset
}
for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    cp_async::pred_load<vec_bits, kPrefetch, kNoFill>(
        k_smem + <target>,
        paged_kv.k_data + kv_offset[j],   ← LOAD K FROM GLOBAL
        <predicate>);
}
cp_async::commit_group();
// Same for V...
```

### Main loop (lines 548-583)

Same pattern repeated in the pipelined loop body.

## Injection Plan: Replace cp_async with Load-Dequant-Store

### What changes

Replace every `cp_async::pred_load` for K and V with:

```cpp
// Instead of: cp_async::pred_load(..., paged_kv.k_data + kv_offset[j], ...)
// Do:
if (predicate) {
    // 1. Load packed quantized bytes from global memory
    uint8_t packed[QUANT_BYTES_PER_VEC];
    load_global(packed, quant_kv.k_data + quant_offset[j]);

    // 2. Unpack indices
    uint8_t indices[vec_size];
    unpack_4bit_or_3bit(packed, indices);

    // 3. Codebook lookup → float
    DTypeKV dequantized[vec_size];
    for (int i = 0; i < vec_size; i++)
        dequantized[i] = __float2half(codebook[indices[i]] * norm);

    // 4. Store to shared memory (same location cp_async would have written)
    store_shared(k_smem + <target>, dequantized);
}
__syncthreads();  // replaces cp_async::wait_group
```

### What stays unchanged

- `compute_qk()` — reads fp16 from shared memory, unchanged
- `update_local_state()` — reads fp16 from shared memory, unchanged
- `sync_state()` — float register operations, unchanged
- Output path — unchanged
- Page table structure (indices, indptr) — same logical structure

### New data structures needed

```cpp
struct paged_kv_turbo_t {
    uint_fastdiv page_size;
    uint32_t num_heads, head_dim, batch_size;
    uint32_t stride_page, stride_n, stride_h;  // adjusted for quantized element size

    uint8_t* k_quant_data;   // packed 4-bit/3-bit codebook indices
    uint8_t* v_quant_data;
    half* k_norms;            // FP16 L2 norm per token per head
    half* v_norms;

    IdType* indices;          // same page table
    IdType* indptr;
    IdType* last_page_len;

    // Codebook in constant memory (16 × float for 4-bit, 8 × float for 3-bit)
};
```

### Pipeline impact

`cp_async` enables async global→shared memory transfer overlapped with compute. Replacing with synchronous loads means the load-dequant-store blocks until complete. Mitigations:

1. **Software pipelining**: Manually double-buffer using `__syncthreads()` instead of cp_async groups
2. **Less data loaded**: TurboQuant loads 480 bytes per 16×64 tile vs 2048 bytes for fp16 — 4.27× less global memory bandwidth
3. **Register-heavy dequant**: Codebook lookup is register-only (16 floats in registers), no extra smem needed

### Build approach

1. Fork `decode.cuh` → `decode_turboquant.cuh` (keep original unchanged)
2. Fork `paged_kv_t` → `paged_kv_turbo_t` (quantized addressing)
3. Add to FlashInfer JIT: new `gen_batch_decode_turboquant_module()`
4. Python wrapper: `BatchDecodeWithTurboQuantKVCacheWrapper`

### Files to create/modify in `~/workdir/flashinfer`

| Action | File |
|--------|------|
| Create | `include/flashinfer/attention/decode_turboquant.cuh` |
| Create | `include/flashinfer/page_turboquant.cuh` |
| Create | `csrc/batch_decode_turboquant.cu` |
| Create | `csrc/batch_decode_turboquant_jit_binding.cu` |
| Create | `flashinfer/jit/attention/turboquant.py` |
| Create | `flashinfer/decode_turboquant.py` |
| Create | `tests/attention/test_decode_turboquant.py` |
