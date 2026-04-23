# HYP-027: TQ backend crashes under CUDA graphs due to post-capture KV cache swap

## Hypothesis

Wrapping TurboQuant's decode and quantize-write kernels as `torch.library` custom
ops (so the PyTorch dispatcher records tensor-identity references instead of raw
pointers) would make the TQ backend survive vLLM's CUDA graph capture.

## Prediction

TQ fp8 under CUDA graphs would reach ~4–5 ms TPOT on Qwen3-1.7B / A100, matching
the speedup FP16 sees from graphs (3.59 → 1.41 ms, ~2.5×).

## Status: **rejected**

`torch.library` registration compiled and ran in eager mode (identical 9.11 ms
TPOT vs 9.14 ms with pybind), but graphs still crash with
`cudaErrorIllegalAddress` at replay time.

## Results

| Config                                    | TPOT best | Notes                             |
|-------------------------------------------|-----------|-----------------------------------|
| FP16 eager                                | 3.59 ms   | FA2 reference                     |
| FP16 graphs                               | **1.41 ms** | graphs give 2.5× here           |
| TQ fp8 eager — `c60e2d9` baseline         | 12.29 ms  | boolean-mask page table           |
| TQ fp8 eager — + strided page table       | **9.14 ms** | -26 % (eager win)               |
| TQ fp8 eager — + strided + `torch.ops`    | 9.11 ms   | dispatcher overhead negligible    |
| TQ fp8 graphs — all three above           | **CRASH** | `illegal memory access` @ replay  |

Memory (KV cache token capacity at 30 % of 40 GB A100):
- FP16: 74,048 tokens
- TQ fp8: **278,592 tokens (3.76×)** — PR1 working

## Root cause (best current understanding)

vLLM V1 captures CUDA graphs **inside** `profile_cudagraph_memory()`, i.e. *before*
the real KV cache is allocated. Captured kernels see a **placeholder cache**
(e.g. `[2, 512, 16, 8, 68]`). After profiling, vLLM allocates the **final cache**
(e.g. `[2, 17412, 16, 8, 68]`) at a different address and swaps the storage on
the `kv_cache` tensor object.

For ops routed through PyTorch's dispatcher (FA's `reshape_and_cache_flash`, the
FP16 decode kernel, `tensor.view` / `contiguous` copies), the graph replay picks
up the new storage automatically — that's why FP16 graphs work.

For our path, the `torch.ops` registration handles dispatch for `decode_v4` and
`quantize_write_kv`, but the **Python-level advanced-indexing write**
`cache_u8[0, bids, boffs, :, :qbytes] = kq` inside `_write_to_cache` appears to
be the remaining graph-hostile piece. The captured write apparently still
references the placeholder storage even though `cache_u8` is a view of the
swapped `kv_cache`.

## Evidence

1. Eager never fails; every graph configuration with TQ fails with
   `cudaErrorIllegalAddress` during `async_copy_ready_event.synchronize()` at
   replay — i.e. async report of a bad access in a captured kernel.
2. Added debug print in `do_kv_cache_update` confirmed the shape shift:
   - eager: `cache[0].shape = (17412, 16, 8, 68)`
   - graphs: `cache[0].shape = (512, 16, 8, 68)` (profile placeholder)
3. Stores to `cache_u8[0] = 0x42` at a FIXED offset 0 — skipping all address
   arithmetic — still faulted, ruling out bugs in our address math.
4. `torch.ops.turboquant.decode_v4(...)` through the PyTorch dispatcher did not
   fix the failure, suggesting the faulting write isn't ours but the Python
   indexing assignment in `_write_to_cache`.
5. A historical diagnostic (`b3843868`, PR3 backend with fused write disabled,
   persistent `_latest_k_buf` / `_slot_buf_i32` and rotation matmul) did run
   under graphs at 9.68 ms. That suggests the failure mode depends on how the
   backend touches tensors around the cache swap, not on TQ being fundamentally
   incompatible.

## Things that did not work

- `enforce_eager=False` + `compilation_config={mode:0, cudagraph_mode:2}` —
  graphs on, inductor off (A100 has no `fp8e4nv`, so inductor must be off).
- Replacing boolean-mask page table with strided layout (fixed capture-time
  `cudaErrorStreamCaptureUnsupported`, but replay still faults).
- Guarding writes with `slot >= 0 && bid < num_blocks`.
- Persistent int32 slot buffer so the pointer is stable across replays.
- Writing to fixed offset 0 instead of computed address.
- `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` registration with `int64_t` / `double`
  wrappers for both `decode_v4` and `quantize_write_kv`.

## Things to try next

1. **Bisect the backend diff** between the failing current `c60e2d9 + patches`
   and the historical success `b3843868`. Candidate culprits:
   - Persistent `_latest_k_buf` / `_latest_v_buf` allocations (even if unused
     under the current path, creating them in `_ensure` may affect the graph
     pool warm-up).
   - Persistent int32 `_slot_buf_i32` + in-place `copy_` of slot_mapping.
   - Rotation matmul step that runs every forward.
   One of those allocator patterns apparently keeps graph capture consistent.
2. **Move the cache write out of `do_kv_cache_update`** into a custom op that
   takes `kv_cache` as a torch tensor arg — so the dispatcher, not Python
   indexing, performs the write. Mirrors what `reshape_and_cache_flash` does.
3. **Check vLLM's `cuda_graph.py`** for the storage-swap mechanism and whether
   there's a hook we need to register for our attention backend (like FA does).
4. **Ask the vLLM team** whether custom attention backends that don't reuse
   `reshape_and_cache_flash` need a specific integration contract to survive
   `profile_cudagraph_memory()`'s placeholder → real swap.

## Shipped outcome (for now)

- PR1 (vLLM upstream custom page size) — **merged**, 3.76× memory savings.
- Local `c60e2d9 + strided page table`: eager TPOT **9.14 ms** (vs 12.29 ms
  pre-fix), output correctness verified.
- `torch.ops` registration kept: architecturally correct, no cost, sets up
  the dispatcher path for any future graph-replay fix.
- CUDA graphs for TQ remain unavailable.

## References

- vLLM V1 cuda graph code: `vllm/v1/worker/gpu_model_runner.py`
  `profile_cudagraph_memory` → `_warmup_and_capture` → `_dummy_run`.
- PyTorch docs: `torch.library.custom_op`, `TORCH_LIBRARY`.
- Related: HYP-023 (cuda-graph-capture), HYP-025 (async-write-overlap),
  HYP-026 (packed-4bit-cache).

## Benchmarks used

Forge jobs (all Qwen3-1.7B, A100, `gpu_memory_utilization=0.3`,
`max_model_len=512`, 5 prompts × 20 tokens, best of 5):
- `f19167df` (baseline c60e2d9, 4 configs)
- `7301e094` (strided page table bench)
- `0acb7f92` (torch.ops compile error — int/float wrappers missing)
- `00265312` (torch.ops with int64/double wrappers, 4 configs)
- `05d7e8c6` (torch.ops TQ-graphs-only isolated)

## Follow-up

HYP-028 pursues path #2 above: move the Python advanced-indexing write into a
dispatcher-routed CUDA op (`quantize_write_kv_fp8_cache`) that takes `kv_cache`
as a `Tensor(a!)` argument. Rationale: mirrors FA's `reshape_and_cache_flash`,
which survives the vLLM storage swap precisely because it's dispatcher-routed.
