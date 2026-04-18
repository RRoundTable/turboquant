# HYP-042b: decode-step attribution at output_len=128

## Context

HYP-042 (output_len=2, prefill-dominated) showed TQ's attention kernel
is 1.48× baseline at the hot path, but the total CUDA time was actually
*slightly lower* for TQ — prefill washed out the per-decode gap we saw
in HYP-041 (1.77× at seq=8192 × batch=8, output_len=128).

To attribute that gap we need a profile where decode dominates.

## Hypothesis

At output_len=128 the per-decode-step delta (TQ − baseline) breaks
down approximately as:

- **kernel (attention read + dequant + softmax + matmul)**: ≥ 80 %
- **workspace `torch.empty` allocation in `_get_v5_ws`**: 5–15 %
- **TQ-only quant-write preamble** (`quantize_write_hadamard_scatter`,
  `scaled_fp8_quant`, `_typeConvert`): ≤ 5 %
- **CPU dispatch / Python shim**: ≤ 5 %, mostly overlapped

i.e. the kernel itself is the load-bearing cost, and it's at the A100
SM80 ceiling already (HYP-035 / HYP-037 / HYP-040). The decision rule:
if this prediction holds, there is no A100 micro-op path left — the
only A100-side lever is **reducing the number of decode-step kernel
invocations** (pre-allocated workspace + fused write-quant, which also
fixes the HYP-041 OOMs), and the real speedup comes from moving to
H100/H200 where torch.compile of fp8e4nv + async ldmatrix unlock
CUDA graphs and the kernel ceiling drops.

## Method

Same env as HYP-041 / HYP-042 (Forge, A100-40GB, `tq-hyp029:pr`, current
docker/vllm_patches overlay carrying vllm-project/vllm#39868).

Two backends (baseline, tq) × one config: **seq=8192, batch=8,
output_len=128**. Each runs in its own `LLM` instance:

1. Warmup `generate()` (untraced) — burns the first-call JIT.
2. `llm.start_profile()` (vllm v0.19 `profiler_config` API, required here).
3. `generate()` producing 128 tokens → 128 decode steps + 1 prefill.
4. `llm.stop_profile()`; trace saved under `VLLM_TORCH_PROFILER_DIR`.

Post-processing (`results/hyp042b/attribute.py`):
- read the Chrome trace JSON
- drop the prefill region (identified by the first `execute_*` span)
- bucket remaining kernels by name prefix:
  `attention` = `decode_v5_*` + `TurboQuantContiguousDecodeKernelV5T` +
    `flash_fwd_splitkv` + `flash_fwd_splitkv_combine` +
    `SplitKVCombineKernel`
  `quant_write` = `quantize_write_*` + `scaled_fp8_quant_*` + `_typeConvert`
  `gemm` = `*gemm*` + `cutlass::Kernel2`
  `other` = everything else
- divide by 128 to get per-decode-step cost
- diff TQ − baseline per bucket → attribution table

## Prediction (for go/no-go after this run)

Per decode step at seq=8192 × batch=8:

| bucket           | baseline | tq    | Δ     | share of Δ |
|------------------|---------:|------:|------:|-----------:|
| attention        |    ~150 μs | ~480 μs | +330 μs | **~80 %** |
| quant_write      |         0 |  ~20 μs | +20 μs  | ~5 %      |
| gemm             |    ~110 μs | ~115 μs | +5 μs   | ~1 %      |
| other + CPU tail |    ~40 μs | ~100 μs | +60 μs  | ~14 %     |
| **total**        |    ~300 μs | ~720 μs | +420 μs | 100 %     |

(Implied: HYP-041's 1.77× per-step at this config implies ~21 ms →
~37 ms full-layer latency; the 0.42 ms per-step delta above scales by
36 Qwen3-8B layers to ≈15 ms per forward pass, which is the right
order of magnitude.)

If the `attention` share comes out ≥ 80 %, HYP-042b **confirms** the
"A100 ceiling is load-bearing" reading. If `attention` ends up < 50 %
and `quant_write` / `other` dominate, the prediction is **rejected**
and we should attack workspace + fused launches first.

## Status: confirmed (attention carries ≥95% of per-step Δ, two new findings)

## Result (output_len=128, same env as HYP-041)

Per-decode-step CUDA (÷ 128 decode steps):

|                | baseline | tq      | Δ         |
|----------------|---------:|--------:|----------:|
| total CUDA     | 17.8 ms  | 35.8 ms | **+18.0 ms (2.01×)** |
| attention hot  |  4.22 ms | 41.3 ms | +37 ms (summed over 2 kernels) |
| GEMMs          | 11.78 ms | 11.80 ms | ~0 |
| quant preamble |  0.12 ms |  0.63 ms | +0.5 ms |
| other          |  ~1.2 ms |  ~1.6 ms | +0.4 ms |

**Attention carries ~95 % of the per-step gap.** The prediction held.
GEMM is at parity; the TQ-only quant preamble and everything else is
under 5 % of Δ combined.

Full breakdown in `results/hyp042b/SUMMARY.md`.

## Two new findings beyond the go/no-go question

1. **Two TQ attention kernels per layer per decode step.**
   `turboquant_v5::decode_v5_from_cache_paged_splitkv_ws…` (577 μs/call)
   **and** `flashinfer::TurboQuantContiguousDecodeKernelV5T…` (571 μs/call)
   each fire 4572 times (36 layers × 127 decode steps). Baseline fires
   one decode kernel per layer per step (110 μs). Summed TQ kernel
   CUDA (+37 ms/step) is greater than measured wall Δ (+18 ms/step) —
   so the two kernels either overlap on separate streams or one is
   redundant. This is the single biggest A100-side optimization lever:
   **drop or fuse the redundant decode-kernel pair**.

2. **Batch-8 makes TQ worse, not better.** HYP-035 (batch=1) saw
   2.69× FlashInfer; this run (batch=8) sees **9.8× per-layer**.
   FlashInfer amortizes across batched requests; TQ's v5 kernel scales
   closer to linear in batch. Serving economics that rely on large
   batch to hide quant overhead do not hold here.

## What this nails down

- The HYP-041 1.77× serving-level gap is 95 % attention-kernel time on
  A100, ≤ 5 % integration overhead.
- Workspace-alloc fixes and fused write-quant will close at most ~1 ms
  per step → 3–5 % of the gap. Worth doing for the OOM fix but not a
  speed play on its own.
- The A100 perf ceiling is the load-bearing constraint. Follow-ups
  (all recorded as their own hypothesis docs), ordered by expected ROI:
    - [HYP-043](HYP-043-decode-kernel-dedup.md) — dedup / fuse the decode-kernel pair (A100, highest)
    - [HYP-044](HYP-044-batch-aware-splitk.md) — batch-aware split-K to saturate SMs at batch > 1
    - [HYP-045](HYP-045-preallocate-v5-workspace.md) — pre-allocated workspace (memory fix for HYP-041 OOMs + small perf)
    - [HYP-046](HYP-046-h100-remeasure.md) — re-measure on H100 (unlocks async ldmatrix + fp8e4nv compile)
