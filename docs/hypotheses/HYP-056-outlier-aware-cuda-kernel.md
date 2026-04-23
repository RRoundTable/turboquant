# HYP-056: Outlier-aware CUDA kernel for production serving (A_35_prime first)

## Context

HYP-055c/d established that outlier-aware mixed-precision quantization closes
most of the fp16 gap for our KV cache at 3.5-bit average — `B_35_prime` with
offline mask achieves −4.5 pp on Llama QA subset, vs paper's claim of 0 pp gap.
But the Python-hook bench (even with HYP-055e vectorization + HYP-055g
batching) runs at ~2 s/sample on B-methods vs ~0.3 s/sample on vLLM. For
sustained iteration and eventually shipping, we need CUDA kernels.

Our existing production kernel `flashinfer_decode_turboquant_v5_tc.cuh` only
supports **uniform 4-bit MSE**. HYP-055 research variants (A_35', B_35',
B_35_paper) are impossible to run through vLLM today.

HYP-056 ports the simplest research variant into production CUDA kernels
first, validates it reaches fp16 parity, then extends to the more complex
variants only if they prove out. Start with **A_35_prime**: no QJL, just a
two-tier MSE codebook. Success here unblocks 10× faster iteration on all
subsequent variant evals.

## Scope: A_35_prime only (defer QJL and paper variants)

A_35_prime layout (head_dim=128):
- **64 outlier dims** × 4-bit MSE (Lloyd-Max 16 levels)
- **64 regular dims** × 3-bit MSE (Lloyd-Max 8 levels)
- Avg: 3.5 bits/dim
- Per-head tile: 64×4/8 + 64×3/8 + 2 fp16 norms = **32 + 24 + 4 = 60 B raw,
  aligned to 64 B**
- Compression vs fp16: 256 / 64 = **4.0×**

Compared to the shipped uniform 4-bit kernel (80 B/head, 3.2× compression),
this **saves memory AND improves quality** if HYP-053 findings hold.

Deferred (new hypotheses when needed):
- HYP-057: QJL residual kernel path (B_35_prime)
- HYP-058: 5-bit codebook + QJL-on-both-tiers (B_35_paper)

## Hypothesis

A CUDA port of A_35_prime achieves:

1. **Correctness**: decoded attention output cosine ≥ 0.998 vs the Python
   reference `_outlier_aware_roundtrip` with mode=A_35_prime, on a common
   real Llama K/V capture.
2. **Latency**: decode kernel at least as fast as v5 uniform-4-bit
   (~64 μs at seq=4096 on A100) — adding a second codebook tier shouldn't
   cost >20% since both tiers dequant in parallel smem ops.
3. **Eval throughput via vLLM**: ≥ 8× faster than Python-hook HF bench on
   100-sample LongBench runs.
4. **Task quality**: matches Python-hook A_35_prime score within ±1 pp on
   the 13-task subset. Validates kernel semantics match reference.

## Design

### KV cache layout (per token per head)

```
byte 0            31 32             55 56      59 60      63
     ┌──────────────┬──────────────────┬──────────┬──────────┐
     │ outlier-tier │ regular-tier     │ norms    │ padding  │
     │  64 dims ×   │  64 dims × 3b    │  2×fp16  │  (align) │
     │  4b = 32 B   │  GGML-packed     │  4 B     │          │
     │  nibble-pack │  = 24 B          │          │          │
     └──────────────┴──────────────────┴──────────┴──────────┘
                       total = 60 B raw, 64 B aligned
```

- **Outlier tier (4-bit)**: same packing as existing v5 (2 nibbles per byte).
  Reuses `kCodebook4bit[16]` from `page_turbo.cuh:18`.
- **Regular tier (3-bit)**: GGML-style packing (8 indices in 3 bytes = 24 bits).
  Reuses `kCodebook3bit[8]` from `page_turbo.cuh:26`.
- **Norms**: two fp16 values, one per tier (outlier norm, regular norm).
- **Alignment**: 64-byte tile → cp.async.ca 128-bit loads work unchanged.

### Outlier mask (critical design choice)

**Mask is per-head per-layer, offline-calibrated, static during serving.**

Storage options:
1. **Per-layer `__constant__` memory** — but `__constant__` is 64 KB per device;
   at 32 layers × 8 heads × 128 dims × 1 bit = 32 KB, fits but leaves little
   for codebooks.
2. **Global memory tensor** passed as kernel arg — preferred. `[num_layers,
   num_heads, head_dim]` bool, loaded into smem at block start. 4 KB per
   layer × block fits easily in shared memory.

Layout: two index arrays per head, precomputed from the mask:
- `outlier_idx[H, 64]` — column indices of outlier dims (offline-computed)
- `regular_idx[H, 64]` — column indices of regular dims

During write kernel, split input K/V along mask into outlier and regular
sub-vectors, quantize each tier independently, pack into the tile.

During decode, the reverse: unpack outlier nibbles + regular 3-bit indices,
dequant to fp16 smem. Then scatter back to `[head_dim]` using the same
index arrays before feeding WMMA.

### Write kernel (prefill path)

File: `csrc/include/turboquant/quantize_write_kernel_a35.cuh`

Per head:
1. Load 64 tokens × 128 dims fp16 K/V from slot_mapping (~8 KB per token
   batch per head → fits in smem).
2. L2-normalize per sub-tier (outlier-64 norm, regular-64 norm, both fp16).
3. FWHT-rotate each sub-tier independently (use existing `hadamard.cuh`
   with `padded_dim = 64`).
4. Codebook-quantize each tier:
   - Outlier: `bucketize(normalized_x, kCodebook4bit_boundaries)` → 4-bit index
   - Regular: same with `kCodebook3bit_boundaries`
5. Nibble-pack outliers to 32 B, GGML-pack regulars to 24 B, append norms.
6. Write 64 B to `kv_cache[slot].data`.

Expected ~20-30 μs per 1 k tokens × 8 KV heads on A100, same order as current
uniform-4-bit write.

### Decode kernel (hot path)

File: `csrc/include/flashinfer_decode_turboquant_v6_a35.cuh` (new)

Structure mirrors v5_tc.cuh with two tweaks:
1. **Two-tier dequant** in smem: outlier rows dequant from 4-bit table, regular
   rows dequant from 3-bit table. Both run in parallel using different
   register-resident LUTs (`__shfl_sync` broadcasts from lanes 0-15 for 4-bit,
   lanes 0-7 for 3-bit).
2. **Index-scatter before WMMA**: after dequanting both tiers into fp16 smem,
   re-interleave using `outlier_idx[h]` / `regular_idx[h]` into canonical
   `[head_dim=128]` column order. Then apply inverse FWHT on each 64-dim
   sub-tier.
3. **WMMA QK** unchanged from v5 (same tile sizes).
4. **V accumulate** unchanged from v5.

Per-token smem cost:
- 2 staging tiers: 32+24 = 56 B
- fp16 dequant buffer: 128 × 2 = 256 B
- Total per tile-N (16 tokens): 56 × 16 + 256 × 16 = **5 KB/token block**
- 2-stage double buffer: 10 KB smem

Fits in A100 100 KB smem budget easily.

### Python binding

File: `csrc/src/decode_v6_a35_binding.cu`

Signature parallels `decode_v5_from_cache_paged_splitkv_ws`:
```cpp
Tensor decode_a35_from_cache_paged_splitkv_ws(
    Tensor q, Tensor kv_cache,
    Tensor indices, Tensor indptr, Tensor last_page_len, Tensor seq_lens,
    int num_kv_heads, int page_size, int head_dim,
    float sm_scale,
    Tensor outlier_idx,   // [num_layers, num_heads, 64] int16 — NEW
    Tensor regular_idx,   // [num_layers, num_heads, 64] int16 — NEW
    Tensor hadamard_signs, int qbytes, int nbytes,
    /* workspaces & split-KV args as v5 */
) -> Tensor
```

`qbytes = 56` (32 outlier + 24 regular), `nbytes = 4` (two fp16 norms).

### vLLM plugin integration

File: `turboquant/vllm_backend_fused.py`

Add a new quant mode `"A_35_prime"` (distinct from `"fp8"` which triggers the
4-bit backend). Dispatch in `get_kv_cache_shape`:

```python
if cache_dtype_str == "A_35_prime":
    return (2, num_blocks, block_size, num_kv_heads, 64)  # bytes_per_head=64
```

Dispatch in attention forward to call
`torch.ops.turboquant_a35.decode_a35_from_cache_paged_splitkv_ws` instead of
the v5 op.

Require user to pass the offline outlier mask path via env var
`TQ_OUTLIER_MASK_PATH`; the backend loads it once, converts to int16 index
tensors, moves to GPU, caches.

## Method — phased implementation with verification gates

### Phase 1 (this doc) — design + skeleton (no CUDA code yet)

Committed by this hypothesis doc. No executable changes.

### Phase 2 — write kernel + Python-reference correctness test

Implement `quantize_write_kernel_a35.cuh` + binding. Write a test that:
1. Generates synthetic K [2048, 8, 128] and a fixed mask.
2. Applies `_outlier_aware_roundtrip(mode=A_35_prime)` (Python ref).
3. Applies CUDA kernel: write to packed buffer, then decode full tokens
   back to fp16.
4. Checks reconstruction cosine ≥ 0.9998 (tighter than 0.998 because this
   is lossless path — both quantize the same way).

**Gate:** cosine fails → design mismatch; fix before Phase 3.

### Phase 3 — decode kernel

Implement `flashinfer_decode_turboquant_v6_a35.cuh` + binding. Test:
1. Write K/V cache via Phase 2 kernel.
2. Run `decode_a35_from_cache_paged_splitkv_ws` on synthetic Q.
3. Compare attention output to Python reference (fp16 Q, packed KV cache,
   full decode through hook). Target cosine ≥ 0.998.

**Gate:** cosine fails → fix kernel. Benchmark latency vs v5 uniform (target
within 20%).

### Phase 4 — vLLM plugin + LongBench eval

Extend `vllm_backend_fused.py`. Run `tests/bench_longbench_vllm.py --mode
A_35_prime` on 13 tasks × 100 samples.

**Gate:** aggregate score within ±1 pp of Python-hook A_35_prime bench.

## Kill criteria

- Phase 2 or 3 correctness cosine < 0.998 after 3 iterations of debugging.
- Phase 3 latency > 2× v5 uniform (would make A_35_prime slower than just
  using uniform 4-bit).
- Phase 4 task quality delta > 1.5 pp vs Python-hook (kernel has a bug).

## Status: pending

Phase 1 design committed. Phase 2 implementation in progress.
