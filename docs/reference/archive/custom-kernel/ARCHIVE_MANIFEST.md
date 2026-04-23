# Reference archive — custom-kernel lineage

**Frozen on:** 2026-04-23

These reference docs describe the abandoned custom-CUDA-kernel track
(`csrc/**` + `turboquant/decode_kernel*.py` + `turboquant/vllm_backend_fused.py`).
Kept for historical record.

The project pivoted on 2026-04-23 to **improving upstream vLLM v0.20.0's
Triton TurboQuant kernels via a vLLM plugin**. References kept at the
`docs/reference/` root:

- `turboquant-paper-methodology.md` — paper §3/§4 methodology + paper-vs-upstream delta.
- `vllm-upstream-turboquant-architecture.md` — upstream Triton kernel architecture (new baseline).

## Files

- `5model-benchmark.md` — 5-Model Full Benchmark: FP16 vs TQ 4-bit
- `architecture-comparison.md` — FlashInfer vs TurboQuant Architecture Comparison
- `baseline-comparison-splitkv.md` — TQ v4 + Split-KV vs FP16 SDPA Baseline — A100
- `baseline-comparison.md` — TurboQuant v4 vs FP16 SDPA Baseline — A100
- `core-implementation.md` — TurboQuant Core Implementation
- `correctness-test.md` — Correctness Test: FP16 vs TurboQuant 4-bit
- `e2e-benchmark.md` — E2E Benchmark: TQ 4-bit vs FP16 Baseline
- `e2e-corrected.md` — Corrected E2E Analysis: Real TQ Overhead with CUDA Graphs
- `experiment-report.md` — TurboQuant Experiment Report
- `flashinfer-analysis.md` — FlashInfer Architecture Analysis
- `flashinfer-comparison.md` — FlashInfer vs SDPA vs TurboQuant — Decode Kernel Comparison
- `flashinfer-decode-injection.md` — FlashInfer Decode Kernel — TurboQuant Injection Plan
- `forge-gpu-isolation-report.md` — Forge GPU isolation issue — admin report
- `memory-layout.md` — TurboQuantTile Memory Layout
- `memory-savings-analysis.md` — Phase 8b+8e: Memory Savings and Max Batch Size Analysis
- `multi-model-correctness.md` — Multi-Model Correctness: FP16 vs TQ 4-bit
- `optimization-plan.md` — TurboQuant Kernel Optimization Plan
- `throughput-analysis.md` — Throughput Analysis: FP16 SDPA vs TQ v4
- `vllm-kv-cache-analysis.md` — vLLM KV Cache Architecture Analysis
