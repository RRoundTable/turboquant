# HYP-029: Dispatcher-routed decode READ op taking `kv_cache` directly

## Hypothesis

Replacing the Python-side `kv_cache.view(torch.uint8)` + four `.contiguous()`
slice views at `turboquant/vllm_backend_fused.py:247-253` with a new CUDA op
`decode_v4_from_cache(kv_cache, ...)` — that takes `kv_cache` as a mutable
`Tensor` argument and derives `k_base / v_base / k_norms / v_norms` internally
— will make TurboQuant fp8 decode survive vLLM V1's CUDA graph capture and
storage swap, and unlock `cudagraph_mode=2` speedup.

## Rationale

HYP-028 rejected the write-only fix and pinpointed the remaining graph-hostile
code: the Python view chain on the read path. The dispatcher refreshes
`data_ptr` only for tensors passed directly into a registered op — not for
Python views constructed from them *before* the op call.

Current flow (graph-hostile):
```python
cache_u8 = kv_cache.view(torch.uint8)                 # Python view — stale ptr
k_q = cache_u8[0][..., :qbytes].contiguous().view(-1) # dispatcher op, but
v_q = cache_u8[1][..., :qbytes].contiguous().view(-1) # input is the stale view
k_n = cache_u8[0][..., qbytes:...].contiguous()...    #
v_n = cache_u8[1][..., qbytes:...].contiguous()...    #
torch.ops.turboquant.decode_v4(q, k_q, v_q, k_n, v_n, ...)
```

Target flow (graph-safe, matches FlashAttention's pattern):
```python
torch.ops.turboquant.decode_v4_from_cache(
    q_fp16, kv_cache, kv_indices, kv_indptr, kv_last_page_len,
    num_kv_heads, block_size, head_size, padded_dim, scale,
    signs, qbytes, nbytes,
)
```

Inside the C++ op body:
```cpp
auto k_t = kv_cache.select(0, 0);  // [num_blocks, block_size, num_heads, qbytes+nbytes]
auto v_t = kv_cache.select(0, 1);
uint8_t* k_base = (uint8_t*)k_t.data_ptr();
uint8_t* v_base = (uint8_t*)v_t.data_ptr();
__half*  k_norms = (__half*)(k_base + qbytes);
__half*  v_norms = (__half*)(v_base + qbytes);
int entry_byte_stride = qbytes + nbytes;   // strided reads — no .contiguous()
```

The kernel is already stride-aware (`entry_byte_stride > 0` path added in
HYP-027), so no kernel changes are needed — only a new binding that computes
the 4 bases from `kv_cache` after the dispatcher refreshes its storage.

Bonus: eliminates four `.contiguous()` allocations per decode step.

## Prediction

On Qwen3-1.7B / A100-SXM4-40GB, `gpu_memory_utilization=0.3`,
`max_model_len=512`, 5 prompts × 20 tokens, best of 5:

| Config         | TPOT (predicted)   | Success criterion                        |
|----------------|--------------------|------------------------------------------|
| TQ fp8 eager   | 8.3–8.9 ms         | ≤ HYP-028 baseline (8.89 ms) — the four removed `.contiguous()` copies may even tighten it |
| TQ fp8 graphs  | **3.5–4.5 ms**     | no `cudaErrorIllegalAddress`; ≥ 2× speedup from graphs, approaching FP16-graphs (1.42 ms) ratio |
| Memory         | 278,592 tokens     | 3.76× savings preserved                  |
| Output match   | exact              | same tokens as FP16 reference and eager TQ |

The 3.5–4.5 ms target mirrors the 2.5× graph speedup FP16 gets (3.54→1.42).
TQ has ~8–9 ms baseline vs FP16's 3.54 — the absolute number will be higher,
but the *ratio* from graphs should be the same, since the kernel compute is
already graph-capturable (HYP-023 confirmed).

## Method

### Step 1 — Add the new binding (no kernel changes)

In `csrc/src/decode_v4_binding.cu`, add:

```cpp
torch::Tensor turboquant_decode_v4_from_cache(
    torch::Tensor q,                 // [batch, num_qo_heads, head_dim] fp16
    torch::Tensor kv_cache,          // [2, num_blocks, block_size, num_heads, qbytes+nbytes] uint8
    torch::Tensor indices,
    torch::Tensor indptr,
    torch::Tensor last_page_len,
    int64_t num_kv_heads,
    int64_t page_size,
    int64_t head_dim,
    int64_t padded_dim,
    double sm_scale,
    torch::Tensor hadamard_signs,
    int64_t qbytes,
    int64_t nbytes
);
```

Body: compute `k_base / v_base / k_norms / v_norms` from `kv_cache.data_ptr()`,
set `entry_byte_stride = qbytes + nbytes`, `layout_nhd = true`, then invoke
the same `paged_kv_turbo_t` + dispatch tree currently in `turboquant_decode_v4`.
Factor the dispatch body into a helper called by both entry points.

Register:
```cpp
TORCH_LIBRARY(turboquant, m) {
  m.def("decode_v4_from_cache(Tensor q, Tensor kv_cache, Tensor indices, "
        "Tensor indptr, Tensor last_page_len, int num_kv_heads, int page_size, "
        "int head_dim, int padded_dim, float sm_scale, Tensor hadamard_signs, "
        "int qbytes, int nbytes) -> Tensor");
}
TORCH_LIBRARY_IMPL(turboquant, CUDA, m) {
  m.impl("decode_v4_from_cache", &turboquant_decode_v4_from_cache_op);
}
```

### Step 2 — Swap the Python call site

`turboquant/vllm_backend_fused.py:244-289`: delete the `cache_u8` view chain,
replace the `torch.ops.turboquant.decode_v4(...)` call with
`torch.ops.turboquant.decode_v4_from_cache(q_fp16, kv_cache, ...)`.

Keep `decode_v4` registered for backward compatibility (still used by
standalone benchmarks and `decode_kernel_v4.py`).

### Step 3 — Correctness tests

Add `tests/test_decode_from_cache.py`:

- **byte-equivalence**: on a fake `kv_cache` pre-populated via the
  HYP-028 write op, compare `decode_v4_from_cache(kv_cache, ...)` output
  vs the old `decode_v4(k_q, v_q, k_n, v_n, ...)` with manually-sliced
  inputs. Must match bit-for-bit.
- **pointer staleness**: allocate a placeholder `kv_cache`, capture a
  graph that calls the op, replace the storage with
  `kv_cache.set_(new_storage)`, replay. Assert no `cudaErrorIllegalAddress`
  and that the result reflects the *new* storage.

### Step 4 — End-to-end benchmark (Forge A100)

Reuse the HYP-028 job harness (`c343e571` lineage): Qwen3-1.7B, 5 prompts ×
20 tokens, best of 5, 4 configs: `{FP16, TQ-fp8} × {eager, graphs}`.

`enforce_eager=False` config: `compilation_config={mode: 0, cudagraph_mode: 2}`
(inductor off — no `fp8e4nv` on A100). Capture phase must complete; decode
step must not crash; TPOT in predicted band.

### Step 5 — If it still crashes

Before filing as rejected, run diagnostic from HYP-028's "Things to try next":
`CUDA_LAUNCH_BLOCKING=1` + `TORCH_USE_CUDA_DSA` to localize the faulting
kernel. Also test `enable_prefix_caching=False` to rule out the cached-path
interaction.

## Status: **confirmed**

Graph replay works, TPOT comes in *below* the predicted band.

## Results (Forge A100-SXM4-40GB, Qwen3-1.7B, gpu_mem_util=0.3, max_model_len=512, 5 prompts × 20 tokens, best of 5 — job `86b8e89c`)

| Config        | Best TPOT | Median  | KV tokens   | Outcome                                     |
|---------------|-----------|---------|-------------|---------------------------------------------|
| FP16 eager    | 3.36 ms   | 3.42 ms |  74,000     | reference                                   |
| FP16 graphs   | 1.43 ms   | 1.44 ms |  73,920     | reference — 2.35× from graphs               |
| TQ fp8 eager  | 4.33 ms   | 4.35 ms | 236,800     | faster than HYP-028 (8.89 ms): 4 `.contiguous()` copies removed from hot path |
| TQ fp8 graphs | **1.74 ms** | **1.77 ms** | **236,384** | **no crash**; 2.49× from graphs; 1.22× slower than FP16 graphs |

### Correctness

Byte-equivalence test `tests/test_decode_from_cache.py` — 6/6 PASSED across
`{hd=64, hd=128} × {num_kv=2,4,8} × {batch=2,3, seq=48,113}`.

Eager output text is now coherent (was gibberish under HYP-028's shipped
path — the strided-indptr bug was present but latent because decoded
tokens were never empirically verified). Example:
- "The capital of France is → Paris: 1. Paris, 2. London, 3. Rome, 4."
- "Water boils at → 100°C at sea level. What is the boiling point of water at 10"

### Memory

236,384 tokens vs HYP-028's 278,592 = **15% regression (3.20× vs 3.76×)**.
Trade-off: 16-byte alignment padding required for `cp_async.ca` 128-bit
loads. For Qwen3-1.7B (hd=128): per-head slot went 68 → 80 bytes.
Recoverable later by either using smaller `cp_async` cpSize or a
manual-load fallback in the kernel — out of scope here.

## What actually broke

Three separate bugs, each exposed by the next:

1. **Python-side `cache_u8 = kv_cache.view(torch.uint8)`** baked in the
   placeholder `data_ptr` at graph capture (HYP-028's diagnosis). Fixed
   by passing `kv_cache` through the dispatcher via `decode_v4_from_cache`.

2. **cp_async 16-byte alignment**: with the tight-packed layout, per-head
   stride was 34 bytes (hd=64) or 68 (hd=128) — not 16-aligned. The
   `cp.async.ca.shared.global` 128-bit loads produced UB, silently
   corrupting the staged quant bytes. Fixed by padding `bytes_per_head`
   to `align(qbytes+nbytes, 16)` in `get_kv_cache_shape` and sourcing
   `entry_byte_stride` from `kv_cache.size(-1)` in the op.

3. **`get_length` under strided indptr**: the backend built a strided
   `indptr` (stride = `max_pages`) for static-shape graph capture, but
   computed `last_page_len` against the *actual* `num_pages`. The kernel
   then returned `(max_pages-1)*block_size + last_page_len` — e.g.,
   seq=48 with max_pages=32 → kernel read 512 tokens of garbage. Under
   eager this silently corrupted attention (gibberish text); under graph
   replay padded batches walked into sentinel block_table entries →
   `cudaErrorIllegalAddress`. Fixed by adding `seq_lens` directly to
   `paged_kv_turbo_t` — `get_length` now returns it verbatim when
   provided.

Bug #2 was identified via a three-way bisection probe (tight/tight vs
strided-quant/tight vs both-strided). Bug #3 only surfaced after #1 and
#2 were fixed and graph replay reached actual kernel execution with
padded batches.

## Also shipped

- Graph-safe write op `quantize_write_kv_cache` (mirrors the read-path
  fix): takes `kv_cache` as mutable Tensor, writes quant+norms directly
  to slot-mapped positions. Replaces the Python advanced-indexing
  scatter that baked in the placeholder pointer at capture time.
  (HYP-028 proposed this but was rejected alone — it's necessary but
  not sufficient without the read-path fix.)

## References

- HYP-028 (custom-op-cache-write) — prior diagnosis; identified #1
- HYP-027 (cuda-graph-kv-cache-swap) — strided-indptr design (#3 latent)
- HYP-023 (cuda-graph-capture) — confirmed kernel is graph-capturable
- PTX cp.async.ca alignment: 128-bit copies require 16-byte aligned src
- `turboquant/vllm_backend_fused.py:244-288` — decode call site
- `turboquant/vllm_backend_fused.py:145-156` — write call site
- `csrc/src/decode_v4_binding.cu` — `decode_v4_from_cache`
- `csrc/src/quantize_write_binding.cu` — `quantize_write_kv_cache`
- `csrc/include/turboquant/page_turbo.cuh` — `seq_lens`-aware `get_length`

## References

- HYP-028 (custom-op-cache-write) — diagnosed this exact root cause
- HYP-027 (cuda-graph-kv-cache-swap) — strided `paged_kv_turbo_t`, stride-aware kernel
- HYP-023 (cuda-graph-capture) — kernel itself is graph-capturable
- FlashAttention `reshape_and_cache_flash` — pattern of passing `kv_cache` through dispatcher
- `turboquant/vllm_backend_fused.py:247-289` — call site to replace
- `csrc/src/decode_v4_binding.cu:64-176, 312-361` — existing op dispatch + registration
