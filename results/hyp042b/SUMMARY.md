# HYP-042b — decode-step attribution at output_len=128

Qwen/Qwen3-8B, seq=8192, batch=8, **output_len=128** (1 prefill + ~127
decode steps), eager, A100-40GB, vLLM v0.19.0 with docker/vllm_patches
overlay (PR #39868). torch.profiler via `LLM(profiler_config=...)`.

Run produced **1024 decode tokens** (8 × 128). Each forward pass covers
36 Qwen3-8B layers.

## Headline totals (full run)

|                    | baseline | tq     | ratio |
|--------------------|---------:|-------:|------:|
| Self CUDA total    |  2.263 s | 4.541 s | **2.01×** |
| Self CPU  total    |  4.685 s | 5.884 s | 1.26× |

Per-decode-step CUDA (÷ 128 decode steps, includes all 36 layers × 8 batched requests):
- baseline: **17.8 ms/step**  (≈ 21 ms wall in HYP-041 → 3 ms eager-dispatch tail ✓)
- tq:       **35.8 ms/step**  (≈ 37 ms wall in HYP-041 → 1.5 ms tail ✓)
- **per-step Δ: +18.0 ms (2.01×)** — matches HYP-041's 1.77× ratio
  (HYP-041 counted output tokens, this counts CUDA engine time; the
  small spread is exactly the eager CPU overlap tail.)

## Per-bucket attribution of the +18 ms per-step gap

**CORRECTION (see HYP-043 rejection):** earlier this doc reported TQ
attention at 41 ms/step from summing two profiler rows
(`decode_v5_from_cache_paged_splitkv_ws` and
`TurboQuantContiguousDecodeKernelV5TC`). Those are the host op and its
child kernel — the *same work* reported twice. The true per-step TQ
attention CUDA is **20.7 ms** (main kernel 20.56 + combine 0.14 +
memset ≈ 0), and total TQ per-step CUDA time (35.8 ms) sums cleanly.
Table below uses the corrected numbers.

| bucket                          | baseline /step | tq /step | Δ /step | **share of Δ** |
|---------------------------------|---------------:|---------:|--------:|---------------:|
| **attention** (main decode kernel + combine) | 4.22 ms | 20.72 ms | +16.50 ms | **~91 %** |
| GEMM (linear layers)            | 11.78 ms       | 11.80 ms | +0.02 ms  | ≈0 % |
| quant preamble (TQ-only minus baseline's `reshape_and_cache_flash`) | 0.12 ms | 0.63 ms | +0.51 ms | ~3 % |
| norm + rope + act               | ~0.85 ms       | ~0.90 ms | +0.05 ms  | ~0.3 % |
| elementwise + typeConvert tail  | ~0.31 ms       | ~0.70 ms | +0.40 ms  | ~2 % |

### Kernel-level detail

Baseline decode-phase attention (per layer per step):
- `flash_fwd_splitkv_kernel` (decode-shaped): **110 μs/call** (4572 calls)
- `flash_fwd_splitkv_combine_kernel`: 6.9 μs/call
- **per-layer per-step: ~117 μs → 4.22 ms across 36 layers**

TQ decode-phase attention (per layer per step):
- `flashinfer::TurboQuantContiguousDecodeKernelV5TC` (main): **571 μs/call** (4572 calls)
- `flashinfer::SplitKVCombineKernel`: 4 μs/call (4572 calls)
- **per-layer per-step: ~575 μs → 20.7 ms across 36 layers**
- (the separate `turboquant_v5::decode_v5_from_cache_paged_splitkv_ws…`
  row at 577 μs/call is the host op that *launches* the main kernel;
  its Self CUDA column re-exposes the same ~571 μs of kernel time and
  must not be added to the main kernel row)

Per-layer per-step kernel-CUDA ratio: **575 μs / 117 μs ≈ 4.9×**.
Compare with HYP-035's batch=1 seq=4096 result: 2.69× FlashInfer. The
batch-8 ratio is materially worse, which is the main finding below.

## Why HYP-042b **confirms** the overall hypothesis

Of the +18 ms per-step wall-time gap, **approximately 95 % is in the
attention kernel/s** (either executed serially or with cost split across
the 2-kernel pair). GEMMs are within noise; quant preamble adds ≤ 1 ms;
everything else is under 1 ms.

So the HYP-042b prediction — "attention kernel carries ≥ 80 % of the Δ"
— holds on A100 with room to spare. The A100 ceiling (HYP-035 /
HYP-037 / HYP-040) is the binding constraint.

## Finding that survives the correction

**Batch-8 makes TQ relatively worse, not better.** HYP-035 reported
2.69× FlashInfer at batch=1. This run reports **4.9× per-layer-per-step
at batch=8** (corrected number, was 9.8× before removing the host-op
double-count). FlashInfer's cost barely grows with batch because it
packs multiple requests into the same decode call and saturates SMs
better. TQ's decode kernel scales closer to linearly with batch — so
larger serving batch amplifies the gap. This is the opposite of what
serving-side optimization assumes and is now the highest-priority
A100-side lever (HYP-044).

## Retracted finding

An earlier version of this document claimed the decode step invokes
*two* TQ attention kernels per layer, each ~575 μs. That was a
misread of the torch.profiler table, which exposes both a custom
torch.library op (`turboquant_v5::decode_v5_from_cache_paged_splitkv_ws`)
and its child CUDA kernel (`TurboQuantContiguousDecodeKernelV5TC`)
with overlapping Self-CUDA columns. Reading
`csrc/src/decode_v5_tc_binding.cu:754–874` confirms exactly one main
decode kernel + one small combine + one memset per call — no
redundancy. See HYP-043 (rejected on inspection) for detail.

## Next steps (decision table)

| finding                     | next hypothesis  | priority |
|-----------------------------|------------------|---------:|
| Batch-8 amplifies gap (4.9× at batch=8 vs 2.69× at batch=1) | HYP-044 "batch-aware split-K; better SM saturation at batch>1" | **high** |
| Workspace alloc on every step (HYP-041 OOMs) | HYP-045 "pre-allocated v5 workspace" | medium (memory fix, small perf) |
| H100 ceiling is different   | HYP-046 "re-measure everything on H100" | medium (blocked on hardware) |
| ~~2 attention kernels / layer~~ | ~~HYP-043 "fuse or drop redundant TQ decode kernel pair"~~ | **rejected** on inspection |

Raw artifacts: `results/hyp042b/{baseline,tq}/{profiler_out_0.txt,
rank0.*.pt.trace.json.gz}`.
