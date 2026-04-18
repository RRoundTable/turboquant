# HYP-042: Attribute the per-decode-step gap between TQ and baseline at seq=8192 × b=8

## Context

HYP-041 showed v5_paged at 0.57× baseline tok/s at seq=8192, batch=8 in
end-to-end vLLM serving on A100-40GB (215 vs 380 tok/s ⇒ 1.77× per-step
slowdown).

The kernel-only gap on A100 has a known floor of ≈2.69× at seq=4096
(HYP-035), set by the smem→mma stall (HYP-037) and the absence of async
ldmatrix on SM80 (HYP-040). Since HYP-030 found the kernel is only ~11%
of TPOT, a 2.69× kernel gap *alone* would give ~1.15–1.20× end-to-end —
not 1.77×. So **roughly half of the HYP-041 slowdown is integration
overhead, not kernel compute**, and we have never profiled where it
goes.

## Hypothesis

The integration overhead is dominated, in this order, by:
1. **Per-step `_get_v5_ws` workspace allocation** (`torch.empty` of
   `(batch, num_kv_heads, max_len, qbytes)` each decode step) — also
   the OOM root cause from HYP-041.
2. **Extra kernel launches per layer** (separate quant-K, quant-V,
   write, then dequant+attn vs FlashInfer's single fused decode call)
   — eager-mode tax, asymmetric.
3. **Python dispatch** through the patched `attention.py` wrapper.

The kernel itself accounts for **less than half** of the per-step gap.

## Prediction

Single-decode-step nsys + ncu profile at seq=8192, batch=8, output_len=2,
TQ vs baseline back-to-back, eager mode (matching HYP-041):

| component                      | baseline | TQ    | TQ−baseline |
|--------------------------------|---------:|------:|------------:|
| total decode step              |  ~2.6 ms |  ~4.6 ms |    +2.0 ms |
| attention kernel (sum)         |  ~1.0 ms |  ~1.6 ms |    +0.6 ms |
| workspace alloc (cudaMalloc)   |  ~0     |  ~0.4 ms |    +0.4 ms |
| extra TQ-only kernels (quant+write) | ~0 |  ~0.6 ms |    +0.6 ms |
| dispatch / Python              |  ~0.2 ms |  ~0.4 ms |    +0.2 ms |

If the table holds, the fix priority is workspace pre-allocation +
fused write/dequant launch, *not* kernel micro-optimization (the A100
ceiling already prevents most of that anyway).

## Method

Single Forge job with `--security-profile profiling-debug` (required for
ncu perf counters) and `--shared-nfs`, A100-40GB, image `tq-hyp029:pr`.
Reuses HYP-041's runtime patch overlay so the vLLM stack is identical.

Workload per backend (baseline then TQ in fresh `LLM` instances):
- `Qwen/Qwen3-8B`, fp16, `enforce_eager=True`, `gpu_memory_utilization=0.85`
- `input_len=8192`, `batch=8`, `output_len=2` (1 prefill + 1 decode under profiler)
- 1 untraced warmup call → start nsys/ncu → 1 traced call

Tools, in order:
1. **nsys** (no perf-counters needed) — system timeline + nvtx ranges
   around `LLM.generate`. Output: `/workspace/shared/hyp042/{baseline,tq}.nsys-rep`.
2. **ncu --section SpeedOfLight --section WarpStateStatistics
   --kernel-name regex:"flash|turbo|TurboQuant|quantize"** — kernel-level
   classification (compute / memory / latency-bound) and stall
   reasons. Output: `…/{baseline,tq}.ncu-rep`.
3. **Python instrumentation** in the bench script:
   `torch.cuda.nvtx.range_push("workspace_alloc")` etc. around
   `_get_v5_ws`, `do_kv_cache_update`, attention call, output copy —
   so the nsys timeline groups time by phase without manual SASS
   reading.

Aggregation script (`results/hyp042/attribute.py`) reads `nsys stats`
JSON output and emits the table above per backend.

## Status: rejected (first pass; spawned HYP-042b)

## Result (output_len=2 first-pass profile)

torch.profiler via `LLM(profiler_config={"profiler":"torch", ...})`,
captured 1 prefill + 2 decodes for both backends.

|                    | baseline | tq      | Δ          |
|--------------------|---------:|--------:|-----------:|
| Self CUDA total    | 70.57 ms | 67.49 ms | **−3.1 ms** |
| Self CPU total     | 192.6 ms | 218.2 ms | +25.6 ms   |
| Attention kernels  | 29.1 ms  | 43.0 ms | **+13.9 ms (1.48×)** |
| GEMMs (linear)     | 37.3 ms  | 37.6 ms | ≈0 |
| Quant preamble (TQ-only: `quantize_write_hadamard_scatter` + `scaled_fp8_quant` + `_typeConvert` fp8) | 0 | 3.0 ms | +3.0 ms |
| `aten::empty_like` calls | unmeasured | 363 | (workspace alloc + activations; +4.8 ms CPU) |

Full table and per-kernel breakdown in `results/hyp042/SUMMARY.md`.

The CUDA total being slightly *lower* for TQ at output_len=2 means **at
this output length prefill dominates and TQ is not slower overall** —
the HYP-041 1.77× gap lives entirely in the decode steps.

## Why the prediction was wrong

HYP-042 predicted "<half of the per-step gap is the kernel; workspace
alloc + dispatch dominate". The data shows the opposite at the
attention hot path: **the kernel pair (`decode_v5_from_cache_paged…` +
`TurboQuantContiguousDecodeKernelV5T`) takes 1.48× the time of
baseline's `flash_fwd_splitkv` even when prefill amortizes over 16
prefill chunks**. The +14 ms gap between TQ and baseline at
output_len=2 is almost entirely kernel, with quant preamble (+3 ms
CUDA) and CPU dispatch (+26 ms CPU but largely overlapped) as
secondary contributors.

This is consistent with HYP-035's A100 ceiling of 2.69× FlashInfer at
seq=4096 (smem→mma stall, no async ldmatrix on SM80 — HYP-037,
HYP-040). The HYP-041 serving gap is the same kernel cost, scaled by
how much of the wall-time is decode-only.

## Next step → HYP-042b

Repeat with `output_len=128` and add per-phase `cudaEvent` timing
inside `bench_vllm_serve.py` so we can attribute the actual decode-step
1.77× gap (not the prefill-dominated total). If 042b confirms kernel
is still ≥80% of the per-step delta, the only A100 lever left is
**reducing the number of decode-step invocations of the kernel**
(workspace pre-allocation + fused write-quant), since the kernel
itself is at the SM80 ceiling.
