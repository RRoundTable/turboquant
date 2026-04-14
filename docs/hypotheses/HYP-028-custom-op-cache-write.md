# HYP-028: Move scatter cache-write into a dispatcher-routed CUDA op

## Hypothesis

Moving `_write_to_cache`'s Python advanced-indexing writes into a CUDA custom
op that takes `kv_cache` as a mutable tensor argument — mirroring FlashAttention's
`reshape_and_cache_flash` pattern — will let TurboQuant's fp8 backend survive
vLLM V1's CUDA graph capture and storage swap.

## Rationale

HYP-027 established that vLLM V1 captures graphs *before* the real KV cache is
allocated, then swaps the cache storage post-profiling via a Python object
replacement (`bind_kv_cache` in `vllm/utils.py:513`). Dispatcher-routed ops —
FA's `reshape_and_cache_flash`, our registered `torch.ops.turboquant.decode_v4` —
survive because the dispatcher records tensor *identity* and refreshes the
underlying storage on replay.

What remains graph-hostile is the Python advanced-indexing block at
`turboquant/vllm_backend_fused.py:161-164`:

    cache_u8[0, bids, boffs, :, :qbytes]          = kq
    cache_u8[0, bids, boffs, :, qbytes:qbytes+nb] = kn_u8
    cache_u8[1, bids, boffs, :, :qbytes]          = vq
    cache_u8[1, bids, boffs, :, qbytes:qbytes+nb] = vn_u8

PyTorch's advanced-indexing assignment is not a dispatcher op we can re-bind on
replay — the captured write targets the placeholder storage even though the
`cache_u8` Python view was rebuilt. HYP-027 proved this by writing to a fixed
offset (all address math skipped) and still faulting.

This hypothesis tests path #2 from HYP-027's "Things to try next": embed the
scatter write in a CUDA kernel that receives `kv_cache` through the dispatcher.

## Prediction

On Qwen3-1.7B / A100-SXM4-40GB, `gpu_memory_utilization=0.3`, `max_model_len=512`,
5 prompts × 20 tokens, best of 5:

| Config         | TPOT (predicted) | Success criterion                    |
|----------------|------------------|--------------------------------------|
| TQ fp8 eager   | 9.14 ms ± 5%     | no regression vs HYP-027 baseline    |
| TQ fp8 graphs  | **4–5 ms**       | no `cudaErrorIllegalAddress` crash   |
| Memory         | 278,592 tokens   | 3.76× savings preserved              |
| Output match   | exact            | same tokens as FP16 reference        |

TPOT of 4–5 ms under graphs mirrors the 2.5× speedup FP16 gets from graphs
(3.59 → 1.41 ms). The kernel compute itself is already graph-safe (HYP-023);
all we're removing is a capture-time fault.

## Method

1. Add `quantize_write_hadamard_scatter_kernel` to
   `csrc/include/turboquant/quantize_write_kernel.cuh` — same quant/FWHT math
   as the existing `quantize_write_hadamard_kernel`, but writes to
   `kv_cache[kv_idx, slot/block_size, slot%block_size, head, :]` instead of a
   packed `[num_tokens, num_heads, qbytes]` output.
2. Add C++ binding `quantize_write_kv_fp8_cache` in
   `csrc/src/quantize_write_binding.cu`. Signature takes `kv_cache` as
   `torch::Tensor` (mutable), `slot_mapping` as int32/int64 tensor, plus
   `qbytes/nbytes/block_size` scalars. Register via existing
   `TORCH_LIBRARY(turboquant_write, …)` block.
3. Replace `_write_to_cache` body in `turboquant/vllm_backend_fused.py` with a
   single `torch.ops.turboquant_write.quantize_write_kv_fp8_cache(...)` call.
4. Benchmark:
   - Eager: reuse HYP-027 harness (`enforce_eager=True`), confirm TPOT ±5 %.
   - Graph: `enforce_eager=False`, `compilation_config={mode:0, cudagraph_mode:2}`.
     No `fp8e4nv` on A100 so inductor must stay off.
5. Cross-check: small-scale pytest compares the new op against the old Python
   scatter path byte-for-byte on a fake cache.

## Status: **rejected**

The write-side fix was necessary but not sufficient. Graphs still crash with
`cudaErrorIllegalAddress` at replay time.

## Results (Forge A100-SXM4-40GB, Qwen3-1.7B, gpu_mem_util=0.3, max_model_len=512, 5 prompts × 20 tokens, best of 5 — job `c343e571`)

| Config        | TPOT best | KV tokens        | Outcome                                       |
|---------------|-----------|------------------|-----------------------------------------------|
| FP16 eager    | 3.54 ms   |  74,000          | reference                                     |
| FP16 graphs   | 1.42 ms   |  73,920          | reference — 2.5× from graphs                  |
| TQ fp8 eager  | 8.89 ms   | **278,592 (3.76×)** | matches HYP-027 baseline (9.14 ms); no regression |
| TQ fp8 graphs | —         | —                | **crash at replay** (`cudaErrorIllegalAddress`)   |

Pytest byte-equivalence vs the old Python scatter path:
`tests/test_quantize_write_cache.py` — **2/2 PASSED**
(`test_scatter_matches_python_path`, `test_scatter_skips_negative_slots`).

Graph capture itself succeeds: all 35 cudagraph capture sizes captured in 3 s,
0.58 GiB pool. The crash is on the FIRST generate step after capture, at
`run_busy_loop → _process_engine_step → step_with_batch_queue → future.result()`
— same location as HYP-027.

## Analysis

Write path is no longer the bottleneck. The remaining graph-hostile piece is
the DECODE READ path at `turboquant/vllm_backend_fused.py:260-266`:

    cache_u8 = kv_cache.view(torch.uint8)
    k_q = cache_u8[0][..., :self._qbytes].contiguous().view(-1)
    v_q = cache_u8[1][..., :self._qbytes].contiguous().view(-1)
    k_n = cache_u8[0][..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16).view(-1)
    v_n = cache_u8[1][..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16).view(-1)

`cache_u8 = kv_cache.view(...)` creates a Python-side view — not a dispatcher
op. Subsequent `.contiguous()` calls are dispatcher-routed, but their input
tensor is the Python view whose `data_ptr` was baked in at capture time.
vLLM's `bind_kv_cache` swap (`vllm/utils.py:513`) replaces the `kv_cache`
object; the view's stored storage pointer is stale on replay.

The dispatcher refreshes `kv_cache` storage only for tensors passed directly
to dispatcher ops — not for Python views derived from them before the op
call. FA survives only because it passes `key_cache, value_cache = kv_cache.unbind(0)`
DIRECTLY to `reshape_and_cache_flash`, and the reads inside FA's decode
kernel take the cache as a tensor argument too.

## Shipped outcome (for now)

- `quantize_write_kv_fp8_cache` op: architecturally cleaner (fuses quant +
  scatter, zero intermediate tensor allocations), pytest-verified, eager
  unchanged. **Not merged to main** per rejected-hypothesis rule.
- HYP-027 strided page table + dispatcher-routed decode_v4 op: already
  shipped (eager win).
- CUDA graphs for TQ fp8: still unavailable.

## Things to try next (HYP-029 candidate)

1. **Move the decode read into a custom op** — add a `decode_v4_from_cache`
   that takes `kv_cache` as a `Tensor` arg and does the slice / contiguous /
   dtype-view internally (or better: makes the kernel stride-aware so no
   `.contiguous()` copy is needed). This matches FA's pattern: one op
   receives `kv_cache` as an argument, the op body handles layout.
2. **Profile the capture-time vs replay-time behavior** — use
   `CUDA_LAUNCH_BLOCKING=1` and `TORCH_USE_CUDA_DSA` (hint shown in logs) to
   pinpoint the exact faulting kernel, not just "somewhere in the step".
3. **Investigate prefix caching interaction** — logs show
   `enable_prefix_caching=True`; cached-request decode paths may reuse
   captured graph segments that reference old cache views in a way that a
   fresh-decode path wouldn't.

## References

- HYP-027 (cuda-graph-kv-cache-swap) — prior investigation, path #2 motivation
- HYP-023 (cuda-graph-capture) — confirmed decode kernel is graph-capturable
- vLLM `_custom_ops.py:1640-1653` — `reshape_and_cache_flash` pattern
- vLLM `utils.py:513` (`bind_kv_cache`) — the storage swap this fix targets
- PyTorch docs: `TORCH_LIBRARY`, `TORCH_LIBRARY_IMPL`
