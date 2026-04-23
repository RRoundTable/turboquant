# Hypothesis archive — custom-kernel lineage

**Frozen on:** 2026-04-23

These hypotheses belong to the abandoned custom-CUDA-kernel track
(`csrc/**` + `turboquant/decode_kernel*.py` + `turboquant/vllm_backend_fused.py`).
They are kept for historical record and to prevent re-trying ideas that
were already rejected.

The project pivoted on 2026-04-23 to **improving upstream vLLM v0.20.0's
Triton TurboQuant kernels via a vLLM plugin**. See `docs/GOAL.md`,
`docs/ROADMAP.md`, and `docs/reference/vllm-upstream-turboquant-architecture.md`
for the new direction. Hypotheses kept at the `docs/hypotheses/` root:

- HYP-049, HYP-050, HYP-052, HYP-054, HYP-055c — QJL rejection evidence
  (justifies upstream's no-QJL choice).
- HYP-057 — upstream baseline confirmation (current baseline).

## Files

- `HYP-001-bdz-parallelism.md` — bdz parallelism will scale linearly with thread count
- `HYP-002-precompute-page-offsets.md` — Precomputing page offsets in smem will reduce divmod overhead
- `HYP-003-in-kernel-fwht.md` — In-kernel FWHT will eliminate 203μs Python overhead
- `HYP-004-flashinfer-compute-reuse.md` — Reusing FlashInfer's compute functions will close the SDPA gap
- `HYP-005-cpasync-staged-pipeline.md` — cp_async staged pipeline will overlap VRAM load with compute
- `HYP-006-fused-inline-dequant.md` — Fused inline dequant will reduce smem traffic and improve occupancy
- `HYP-007-tensor-core-dequant-pipeline.md` — Rotated-domain attention + dequant→fp16→tensor cores will match SDPA
- `HYP-007a-wmma-qk-feasibility.md` — WMMA tensor cores for QK dot product — feasibility test
- `HYP-008-bottleneck-isolation.md` — Isolate the real bottleneck — paging, dequant, or occupancy?
- `HYP-009-page-offset-precompute.md` — Precompute page offsets — now feasible with v4's low smem
- `HYP-010-larger-tile-size.md` — Larger tile_size_per_bdx will amortize per-tile overhead
- `HYP-011-larger-page-size.md` — Larger page_size will reduce page table overhead
- `HYP-012-profiling-baseline.md` — Profile TQ v4 vs FlashAttention to identify dominant stall
- `HYP-013-split-kv-parallelism.md` — Split-KV parallelism (FlashDecoding) to fill all SMs
- `HYP-014-register-cap.md` — `--maxrregcount` to increase occupancy
- `HYP-015-adaptive-splitkv.md` — Adaptive split-KV with per-block overhead reduction
- `HYP-016-kernelagent-optimization.md` — KernelAgent for systematic kernel optimization
- `HYP-017-contiguous-kv-layout.md` — Contiguous KV layout to eliminate paging overhead
- `HYP-018-contiguous-splitkv.md` — Contiguous + split-KV combined
- `HYP-019-int4-tensor-core.md` — INT4 tensor cores for QK dot product
- `HYP-020-persistent-warp-specialization.md` — Persistent kernel with warp specialization
- `HYP-021-cuda-write-kernel.md` — CUDA write kernel to fix prefill overhead
- `HYP-022-fused-combine.md` — Fuse combine into decode kernel to save 5μs/layer
- `HYP-023-cuda-graph-capture.md` — CUDA graph capture of TQ decode kernel
- `HYP-024-single-cache-architecture.md` — Single Quantized Cache Architecture
- `HYP-025-async-write-overlap.md` — Async Write Overlap for TPOT Reduction
- `HYP-026-packed-4bit-cache.md` — Packed 4-bit Cache (68 bytes/head, 3.76× savings)
- `HYP-027-cuda-graph-kv-cache-swap.md` — TQ backend crashes under CUDA graphs due to post-capture KV cache swap
- `HYP-028-custom-op-cache-write.md` — Move scatter cache-write into a dispatcher-routed CUDA op
- `HYP-029-decode-read-from-cache.md` — Dispatcher-routed decode READ op taking `kv_cache` directly
- `HYP-030-seqlen-sweep-bench.md` — Decode latency scaling vs seq_len on Qwen3-8B
- `HYP-031-tensor-core-dequant.md` — Tensor-core dequant for decode kernel
- `HYP-032-warp-shuffle-codebook.md` — Register-resident codebook with warp-shuffle LUT
- `HYP-033-v5-cuda-graph-capture.md` — Make v5 tensor-core decode CUDA-graph-safe
- `HYP-034-v5-splitkv-from-cache.md` — Port split-KV into v5 `decode_v5_from_cache_ws`
- `HYP-035-v5-paged-native.md` — Delete the gather — make v5 read paged KV directly
- `HYP-036-v-wmma-warp-softmax.md` — WMMA for V accumulate + warp-reduced softmax
- `HYP-037-wmma-split-accumulator.md` — Split WMMA_QK accumulator to break serial c_frag dependency
- `HYP-038-wmma-pipelined-loads.md` — Software-pipelined WMMA loads
- `HYP-040-ptx-ldmatrix-async.md` — Raw PTX `ldmatrix.async` proof-of-concept
- `HYP-041-v5-paged-vs-baseline-serving.md` — v5_paged vs baseline vLLM at end-to-end serving (Qwen3-8B, A100-40GB)
- `HYP-042-decode-step-attribution.md` — Attribute the per-decode-step gap between TQ and baseline at seq=8192 × b=8
- `HYP-042b-decode-step-at-ol128.md` — decode-step attribution at output_len=128
- `HYP-043-decode-kernel-dedup.md` — dedup / fuse the TQ decode-kernel pair
- `HYP-044-batch-aware-splitk.md` — batch-aware split-K to saturate SMs at batch > 1
- `HYP-045-preallocate-v5-workspace.md` — pre-allocate v5 workspace at engine init
- `HYP-046-h100-remeasure.md` — re-measure the whole HYP-041 / HYP-042b stack on H100
- `HYP-047-kv-offload-reuse.md` — TQ KV offload + reuse — validate transfer cost first
- `HYP-051-vllm-serving-cuda-graphs.md` — Enable end-to-end CUDA graphs for TQ backend under vLLM serving
- `HYP-053-outlier-aware-mse-only.md` — Outlier-aware 2.5-bit MSE-only (no QJL)
- `HYP-055-longbench-polarquant-vs-qjl.md` — LongBench re-verification — PolarQuant vs PolarQuant+QJL
- `HYP-055b-longbench-paper-full-recipe.md` — LongBench — paper's full recipe (outlier-aware + QJL on regulars)
- `HYP-056-outlier-aware-cuda-kernel.md` — Outlier-aware CUDA kernel for production serving (A_35_prime first)

## Deleted

- `HYP-056-paper-reproduction-llama-3-1-8b.md` — obsolete draft, never ran,
  superseded by HYP-057.
