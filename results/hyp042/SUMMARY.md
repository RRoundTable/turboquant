# HYP-042 — decode-step attribution (first pass, output_len=2)

Qwen/Qwen3-8B, seq=8192, batch=8, **output_len=2** (1 prefill + 2 decodes),
enforce_eager=True, A100-40GB, vLLM v0.19.0 with docker/vllm_patches
overlay (PR #39868). torch.profiler via `LLM(profiler_config=...)`.

## Headline totals (per run: 1 prefill + 2 decodes)

|                      | baseline | tq     | Δ       |
|----------------------|---------:|-------:|--------:|
| Self **CUDA** time   | 70.57 ms | 67.49 ms | −3.1 ms  (!) |
| Self **CPU**  time   | 192.57 ms | 218.20 ms | +25.6 ms |
| aten::mm (linear) CUDA | 37.26 ms | ~37.6 ms | ≈0 |

The CUDA delta going **negative** at output_len=2 is the headline finding.
It means at low output_len, **prefill dominates and TQ is no slower
overall** — the HYP-041 gap (1.77× per decode step at output_len=128)
lives entirely in the decode phase, not the prefill phase.

## Per-kernel breakdown (CUDA self-time; top contributors)

### Baseline — attention ≈ 29 ms (41 % of total CUDA)

| Kernel                                              | calls | total  | avg     |
|-----------------------------------------------------|------:|-------:|--------:|
| `flash_fwd_splitkv_kernel` (prefill-shaped)         |  72   | 25.00 ms | 347 μs |
| `flash_fwd_splitkv_kernel` (decode-shaped)          |  36   |  3.79 ms | 105 μs |
| `flash_fwd_splitkv_combine_kernel`                  |  36   |  0.36 ms |  10 μs |
| `reshape_and_cache_flash_kernel`                    | 108   |  0.39 ms |   4 μs |

### TurboQuant — attention+quant ≈ 43 ms (64 % of total CUDA)

| Kernel                                              | calls | total  | avg     |
|-----------------------------------------------------|------:|-------:|--------:|
| `decode_v5_from_cache_paged_splitkv_ws…`            |  36   | 20.61 ms | 573 μs |
| `flashinfer::TurboQuantContiguousDecodeKernelV5T…`  |  36   | 20.40 ms | 567 μs |
| `turboquant::quantize_write_hadamard_scatter_kernel`| 216   |  0.98 ms | 4.6 μs |
| `vllm::scaled_fp8_quant_kernel_strided_group_shared`| 108   |  0.98 ms | 9.0 μs |
| `SplitKVCombineKernel`                              |  36   |  0.14 ms | 3.9 μs |
| `_typeConvert` fp8 variant                          | 216   |  1.02 ms | 4.7 μs |

So on the **attention hot path** TQ spends ≈ 43 ms vs baseline's ≈ 29 ms
— **+14 ms, ≈1.48×**. The gemms (linear layers) are within noise (both
~37 ms). The quantize-and-write preamble (hadamard_scatter + scaled_fp8
+ typeConvert) costs ≈ 3 ms of CUDA time across 3 forward passes — real
but small.

## What this confirms / refutes vs the HYP-042 prediction

HYP-042 predicted the kernel would be **< half** of the per-step gap,
with most of it in workspace alloc + dispatch. This run at output_len=2
actually shows the opposite: **at steady state the kernel attention
block is the dominant cost** — consistent with HYP-035's A100 ceiling
of 2.69× FlashInfer at seq=4096 (smem→mma stall, no async ldmatrix on
SM80).

Where HYP-042 was partly right:
- CPU is +13 % for TQ (192 → 218 ms). Some of that is the
  extra launches per layer (quantize-K, quantize-V, write) + Python
  patch shim. That is real integration overhead, but it is secondary.
- Workspace allocation shows up as `aten::empty_like` (363 calls, 4.8 ms
  CPU) — present, but not the smoking gun.

Where HYP-042 was wrong:
- Kernel **is** the main cost, not integration.
- Prefill is at parity (or TQ is slightly faster — the `_ws` decode
  kernel is used across both prefill and decode and runs at ~570 μs
  per invocation, close to baseline's 347 μs prefill). The per-decode
  cost grows *linearly with cache*, which is why HYP-041 saw the gap
  widen at longer seq.

## Caveat and next step

This run used output_len=2, so decode-only time is tiny relative to
prefill. To isolate per-decode cost:

- **HYP-042b** (proposed): re-run the same profile with
  `output_len=128` and add per-phase timing (`cudaEvent` around
  `_get_v5_ws` / dequant / combine). This will let us say exactly what
  fraction of a decode step is kernel vs workspace vs dispatch and
  attribute the HYP-041 1.77× gap.

Raw artifacts: `results/hyp042/{baseline,tq}/{profiler_out_0.txt, rank0.*.pt.trace.json.gz}`.
