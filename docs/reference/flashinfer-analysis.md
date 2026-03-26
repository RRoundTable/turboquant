# FlashInfer Architecture Analysis

Analysis of FlashInfer's internals relevant to TurboQuant KV cache quantization integration.

Source: `~/workdir/flashinfer` on `mlsys-dgx-spark` (cloned 2026-03-26).

---

## 1. Project Structure

```
flashinfer/
├── include/flashinfer/           # Header-only CUDA kernels (framework-agnostic, raw pointers)
│   ├── attention/
│   │   ├── decode.cuh            # Decode attention kernel (single-query per batch)
│   │   ├── prefill.cuh           # Prefill attention kernel (multi-query)
│   │   ├── variants.cuh          # DefaultAttention variant (masking, sliding window, softcap)
│   │   ├── variant_helper.cuh    # AttentionVariantBase + macros (REGISTER_LOGITS_TRANSFORM, etc.)
│   │   ├── state.cuh             # Online softmax state tracking
│   │   ├── cascade.cuh           # Cascading attention (merge partial results)
│   │   ├── hopper/               # SM90-specific kernels (FlashAttention-3)
│   │   └── blackwell/            # SM100+ kernels
│   ├── vec_dtypes.cuh            # vec_t<T,N> — vectorized load/store with cast_load/cast_store
│   ├── math.cuh                  # Fast math (exp2, log2, rcp, tanh)
│   └── pos_enc.cuh               # RoPE positional encoding
│
├── csrc/                          # TVM-FFI bindings (PyTorch tensor → raw pointer)
│   ├── batch_prefill.cu
│   ├── batch_prefill_jit_binding.cu
│   ├── batch_decode.cu
│   ├── batch_decode_jit_binding.cu
│   ├── *.jinja                    # Jinja templates for type specialization
│   └── page.cu                    # Paged KV cache append kernels
│
├── flashinfer/                    # Python package
│   ├── attention.py               # BatchAttention (plan → run pattern)
│   ├── prefill.py                 # BatchPrefillWithPagedKVCacheWrapper
│   ├── decode.py                  # BatchDecodeWithPagedKVCacheWrapper
│   ├── page.py                    # append_paged_kv_cache, get_batch_indices_positions
│   ├── quantization/              # FP4, FP8 quantization
│   ├── jit/
│   │   ├── core.py                # JitSpec, gen_jit_spec, build system
│   │   ├── env.py                 # FLASHINFER_GEN_SRC_DIR, FLASHINFER_CSRC_DIR
│   │   ├── attention/
│   │   │   ├── modules.py         # gen_batch_prefill_module, gen_customize_batch_prefill_module
│   │   │   └── variants.py        # AttentionSink variant declarations (C++ as Python strings)
│   │   └── ...
│   └── utils.py                   # TensorLayout, MaskMode, dtype helpers
│
└── 3rdparty/
    ├── cutlass/                   # NVIDIA CUTLASS (tensor core ops)
    └── spdlog/                    # Logging
```

## 2. JIT Compilation System

FlashInfer JIT-compiles CUDA kernels at runtime. No reinstall needed after editing `.cuh` files.

### Three Layers

1. **JitSpec** (`flashinfer/jit/core.py`) — Compilation metadata: source files, nvcc flags, URI hash
2. **Code Generation** (`flashinfer/jit/attention/modules.py`) — `gen_*_module()` functions render Jinja templates and copy sources to a writable gen directory
3. **Build & Load** — Generates `build.ninja`, compiles with nvcc, loads `.so` via TVM-FFI

### Key Rule

- `include/` — Read-only, framework-agnostic CUDA headers (no torch.h)
- `csrc/` — TVM-FFI bindings (torch tensors → raw pointers), read-only templates
- `FLASHINFER_GEN_SRC_DIR` — Writable dir for generated sources
- `FLASHINFER_JIT_DIR` — Writable dir for compiled `.so` files

### Adding a New Kernel

1. Write kernel in `include/flashinfer/my_op.cuh` (raw pointers, no torch)
2. Write launcher in `csrc/my_op.cu` (TVM-FFI bindings)
3. Write JIT binding in `csrc/my_op_jit_binding.cu`
4. Write JIT generator in `flashinfer/jit/my_op.py`
5. Write Python API in `flashinfer/my_op.py` with `@functools.cache`
6. Write tests, register in `flashinfer/aot.py`, export in `__init__.py`

Reference: `.claude/skills/add-cuda-kernel/SKILL.md` in flashinfer repo.

## 3. Attention Kernel Architecture

### Decode Attention (`include/flashinfer/attention/decode.cuh`)

The decode kernel is the hot path for autoregressive generation (one new query token, many cached KV tokens).

**Data flow:**
```
For each KV tile:
  1. Load K tile from global memory → shared memory (cp_async / vectorized load)
  2. Compute QK dot product: s[j] = sum(q_vec[i] * k_vec[i])
  3. Apply variant transforms: LogitsTransform, LogitsMask
  4. Online softmax update (state_t tracks running max/sum)
  5. Load V tile → shared memory
  6. Accumulate attention output: o += softmax(s) * V
```

**KV loading** uses `vec_t<DTypeKV, vec_size>::cast_load()` — this handles FP16→float, BF16→float, and FP8→float conversion automatically at the vector level.

### Prefill Attention (`flashinfer/prefill.py`)

Two backends:
- **FA2** (FlashAttention-2) — SM80+, default for most GPUs
- **FA3** (FlashAttention-3) — SM90+ (Hopper), uses TMA and warp-specialization

Both follow a **plan → run** pattern:
1. `plan()` — Analyzes batch layout, allocates workspace, selects tile sizes
2. `run()` — Executes attention kernel with planned configuration

### Paged KV Cache

**Storage format:**
```
paged_k_cache: [max_num_pages, page_size, num_kv_heads, head_dim]  # NHD layout
paged_v_cache: same shape
```

**Page management:**
- `kv_indices: [total_pages]` — Maps logical pages to physical page slots
- `kv_indptr: [batch_size + 1]` — CSR-format indptr for per-request page lists
- `kv_last_page_len: [batch_size]` — How many entries are used in the last page

**Append path:** `append_paged_kv_cache()` writes new KV entries into allocated pages.

## 4. Attention Variant System

The **most important extension point** for TurboQuant. FlashInfer's attention behavior is customizable via C++ structs that inherit from `AttentionVariantBase`.

### AttentionVariantBase (`include/flashinfer/attention/variant_helper.cuh`)

```cpp
struct AttentionVariantBase {
  static constexpr bool use_softmax = true;

  // Override these via REGISTER_* macros:
  LogitsTransform(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx)
  LogitsMask(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx)
  OutputTransform(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale)

  // v_scale support built in:
  get_v_scale(params)  // returns params.v_scale if exists, else 1.0f
};
```

### Customizable Hooks (REGISTER_* Macros)

| Macro | Purpose | When Called |
|-------|---------|------------|
| `REGISTER_LOGITS_TRANSFORM` | Transform attention logits (e.g., softcap, ALiBi) | After QK dot product |
| `REGISTER_LOGITS_MASK` | Mask out positions (causal, sliding window) | After logits transform |
| `REGISTER_M_D_UPDATE` | Custom softmax state update | Per KV tile |
| `REGISTER_OUTPUT_TRANSFORM` | Transform final output (e.g., rescale) | After all tiles |
| `REGISTER_QUERY_TRANSFORM` | Transform query before dot product | Before QK |
| `REGISTER_KEY_TRANSFORM` | Transform key before dot product | Before QK |

### Custom Variant Registration (Python Side)

```python
# Define variant as a C++ string in Python
my_variant_decl = r"""
struct MyVariant : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ MyVariant(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, ..., { return logits; })
  REGISTER_LOGITS_MASK(params, ..., { return true; })
};
"""

# Pass to JIT module generator
module = gen_customize_batch_prefill_module(
    backend="fa2",
    uri="my_custom_attention",
    dtype_q=torch.float16, dtype_kv=torch.float16, dtype_o=torch.float16,
    idtype=torch.int32,
    head_dim_qk=128, head_dim_vo=128,
    additional_tensor_names=["my_tensor"],      # Extra tensors passed to kernel
    additional_tensor_dtypes=["float"],
    additional_scalar_names=["my_scalar"],       # Extra scalars passed to kernel
    additional_scalar_dtypes=["double"],
    variant_name="MyVariant",
    variant_decl=my_variant_decl,
)
```

### Existing Example: AttentionSink

`flashinfer/jit/attention/variants.py` contains `AttentionSink` — a custom variant that adds a learned "sink" token to the softmax denominator. It demonstrates:
- Custom init from params
- Custom logits masking
- Custom m/d update (modifying softmax state)
- Custom output transform (rescaling by modified denominator)

## 5. FP8 KV Cache Support (Existing Quantization Pattern)

FlashInfer already supports FP8 KV cache — this is the closest existing pattern to TurboQuant integration.

### How FP8 KV Works

1. **Storage:** KV cache tensors use `torch.float8_e4m3fn` dtype instead of FP16/BF16
2. **Dequantization:** Happens automatically in the kernel via `vec_t::cast_load()` — FP8 values are cast to float during vectorized loads from shared memory
3. **Scale factors:** `k_scale` and `v_scale` are passed as scalar floats to rescale after dequantization
4. **Block scales:** `key_block_scales` and `value_block_scales` support per-block (finer-grained) scaling

**Key insight:** FP8 dequantization is trivially handled by hardware type casting. TurboQuant requires a multi-step dequantization (codebook lookup + inverse rotation + norm rescaling) which cannot use the same `cast_load` mechanism.

## 6. Integration Strategy for TurboQuant

### Challenge

TurboQuant's dequantization pipeline:
```
indices (uint8) → codebook_lookup → inverse_hadamard_rotation → norm_rescale → float vector
```

This is fundamentally different from FP8 (simple type cast). The dequantization involves:
1. **Codebook lookup:** Map uint8 index → float centroid (table lookup)
2. **Inverse Hadamard rotation:** `x = unpad(signs * FWHT(y))` — O(d log d) per vector
3. **Norm rescaling:** `x_hat = x * norm`

### Possible Approaches

#### Approach A: Fused Dequantize-Attention Kernel

Write a custom attention kernel that reads quantized KV directly and dequantizes inline during the attention computation.

**Where it hooks in:** Replace the KV loading path in `decode.cuh` / `prefill.cuh`. Instead of `k_vec.cast_load(smem + ...)`, load quantized indices, look up centroids, apply inverse rotation, then compute QK.

**Pros:** Maximum fusion, minimal memory bandwidth
**Cons:** Requires modifying core attention kernel templates; Hadamard rotation is non-trivial in shared memory

#### Approach B: Custom KV Cache Page Layout

Define a new paged KV layout that stores `(indices: uint8, norms: float32)` per page instead of `(keys: fp16, values: fp16)`. Write custom `append_paged_kv_cache` that quantizes on write, and a custom attention kernel that dequantizes on read.

**Where it hooks in:**
- Write path: Custom `append_paged_kv_cache` → quantize → store indices+norms
- Read path: Custom attention variant or kernel → load indices+norms → dequantize → attention

**Pros:** Clean separation, works with FlashInfer's page table management
**Cons:** Requires custom page format; dequantization still needs kernel modification

#### Approach C: Separate Dequantize Kernel + Standard Attention

Run TurboQuant dequantization as a separate CUDA kernel before calling FlashInfer's standard attention.

**Where it hooks in:**
- Store quantized KV in custom format
- Before attention: run dequant kernel → write full FP16 KV to a staging buffer
- Call standard FlashInfer attention on the staging buffer

**Pros:** Simplest to implement; no FlashInfer kernel modifications
**Cons:** Loses memory savings (staging buffer is full precision); extra kernel launch overhead; defeats the purpose of compression during attention

#### Approach D: Attention Variant + Pre-dequantized KV Tiles (Recommended Start)

Use Triton or a custom CUDA kernel to dequantize KV tiles on-the-fly and pass them as "additional tensors" through the variant system.

1. Store quantized KV in separate tensors (indices, norms, codebook)
2. Write a dequantize kernel that operates per-page-tile
3. Fuse with attention using FlashInfer's custom variant mechanism

**Why this may work:** The variant system lets you pass additional tensors to the kernel. A custom variant's `__device__` init can load and dequantize quantized KV from these tensors during kernel execution.

### DGX Spark Considerations

- **GPU:** NVIDIA GB10, compute capability 12.1 (SM121, Blackwell family)
- **SM121 limitation:** Does NOT support `tcgen05` MMA instructions required by FlashInfer's CUTLASS FMHA path. Must use `backend='fa2'` (FlashAttention-2 style kernels)
- **Architecture flags:** Use `supported_major_versions=[12]` or broader as needed

## 7. Key Files for Integration Work

| File | Purpose |
|------|---------|
| `include/flashinfer/attention/decode.cuh` | Decode kernel — KV loading + QK computation hot loop |
| `include/flashinfer/attention/variant_helper.cuh` | AttentionVariantBase + REGISTER_* macros |
| `include/flashinfer/vec_dtypes.cuh` | vec_t cast_load/cast_store (KV dtype handling) |
| `flashinfer/jit/attention/modules.py` | `gen_customize_batch_prefill_module` — custom variant JIT |
| `flashinfer/jit/attention/variants.py` | AttentionSink example variant |
| `flashinfer/prefill.py` | BatchPrefillWithPagedKVCacheWrapper (plan→run) |
| `flashinfer/page.py` | append_paged_kv_cache (KV write path) |
| `flashinfer/quantization/` | Existing FP4/FP8 quantization (reference pattern) |
| `.claude/skills/add-cuda-kernel/SKILL.md` | Step-by-step guide for adding new kernels |

## 8. FlashInfer API Patterns

### Plan-Run Pattern

```python
wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim)
output = wrapper.run(q, paged_kv_cache)
```

### Module Caching

```python
@functools.cache
def get_module(*args):
    return gen_some_module(*args).build_and_load()
```

Two-level cache: Python `@functools.cache` + file-level `~/.cache/flashinfer/`.

### TVM-FFI Binding

```cpp
// csrc/my_op_jit_binding.cu
#include "my_op.cu"
#include "tvm_ffi_utils.h"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, my_launcher);
```
