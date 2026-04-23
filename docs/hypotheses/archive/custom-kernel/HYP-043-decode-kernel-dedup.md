# HYP-043: dedup / fuse the TQ decode-kernel pair

## Evidence (from HYP-042b)

At seq=8192 × batch=8 × output_len=128 on A100-40GB, every decode step
invokes **two** TQ attention kernels per layer:

- `turboquant_v5::decode_v5_from_cache_paged_splitkv_ws…` — 4572 calls, **577 μs/call**
- `flashinfer::TurboQuantContiguousDecodeKernelV5T…` — 4572 calls, **571 μs/call**

Baseline fires one decode kernel per layer per step (110 μs/call).
Summed TQ kernel CUDA per step is **+37 ms** greater than baseline, but
the measured end-to-end wall Δ is **+18 ms** (HYP-041 / HYP-042b) — so
the two kernels either overlap on different streams (wall ≈ max of the
two, not sum) or one of them is redundant work.

## Hypothesis

One of the two kernels is doing work that the other already did, or
both kernels are running on independent streams and we can either
drop one or fuse the dispatch so only one kernel is launched per layer
per decode step.

## Prediction

After dedup/fuse:

- per-layer per-step attention CUDA drops from ~1150 μs to **≤ 600 μs**
- per-decode-step wall Δ (vs baseline) drops from +18 ms to **≤ +10 ms**
- end-to-end decode tok/s (seq=8192 × b=8) rises from 215 to **≥ 300 tok/s**
  (recovering ~40 % of the HYP-041 gap vs baseline 380 tok/s)

If the two kernels are genuinely complementary (e.g. K vs V, or paged
vs contiguous for different page regions) and not redundant, the
prediction fails and we instead file a fuse-inside-one-kernel hyp.

## Method

1. Read the kernel invocation sites in
   `turboquant/vllm_backend_fused.py` and the CUDA bindings in
   `csrc/src/decode_v5_tc_binding.cu` to see exactly what the two
   kernels compute and on which streams.
2. If redundant: gate the second kernel behind a runtime check, re-run
   a unit test for numeric parity (cosine vs CPU ref), re-run
   HYP-042b with `output_len=128` at seq=8192 b=8 for timing.
3. If complementary but serializable: fuse into one kernel or launch
   on the same stream with shared smem.

## Status: rejected on code inspection — there is no redundant kernel pair

## Finding

Reading `csrc/src/decode_v5_tc_binding.cu` lines 754–874
(`decode_v5_from_cache_paged_splitkv_ws`), the host function performs
exactly three GPU operations per call:

1. `cudaMemsetAsync` on `o_ws` (line 805)
2. `TurboQuantContiguousDecodeKernelV5TC<128, BDY, 4, true, V, CP>`
   — the main split-K decode kernel (line 847)
3. `SplitKVCombineKernel<__half>` — the combine kernel (line 863)

The torch.profiler table in HYP-042b shows two rows at ~4572 calls each:

- `turboquant_v5::decode_v5_from_cache_paged_splitkv_ws…` (host op)
- `flashinfer::TurboQuantContiguousDecodeKernelV5TC…` (child kernel)

These are the **same work reported at two levels**: torch.profiler
attributes a custom torch.library op's Self CUDA to include its child
kernel launches. Both rows add into `Self CUDA time total`, so that
total *double-counts* the attention kernel for TQ.

## Corrected per-decode-step attribution (from HYP-042b data)

Per decode step (÷ 128 decode steps), CUDA time from the actual
kernel-level rows (memset is ≤ 0.1 ms/step, ignored):

| bucket                                           | baseline | tq      | Δ        | **share of Δ** |
|--------------------------------------------------|---------:|--------:|---------:|---------------:|
| attention (main decode kernel + combine)         |  4.22 ms | 20.72 ms | +16.50 ms | **~91 %** |
| GEMMs                                            | 11.78 ms | 11.80 ms | ≈0 |  ≈0 % |
| quant preamble (TQ-only − baseline `reshape_and_cache_flash`) | 0.12 ms | 0.63 ms | +0.51 ms | ~3 % |
| norm + rope + act + elementwise tail             | ~1.2 ms | ~1.6 ms | +0.4 ms | ~2 % |
| (unaccounted, fits rounding)                     | — | — | ~+0.6 ms | ~4 % |

**HYP-042b's qualitative conclusion stands — attention is ≥ 90 % of
the per-step gap.** The revised per-layer-per-step kernel ratio is
**4.9×** (not 9.8× as the double-counted sum in HYP-042b suggested),
matching the picture: worse than HYP-035's batch=1 result (2.69×) but
not by an order of magnitude.

## Implication for the follow-up ranking

- HYP-043 (dedup / fuse) is **moot** — there is nothing to dedup.
- **HYP-044 (batch-aware split-K) becomes the highest-priority A100
  lever.** The 4.9× at batch=8 vs 2.69× at batch=1 ratio gap points
  straight at it.
- HYP-045 (pre-alloc workspace) remains the memory fix.
- HYP-046 (H100) remains the "change-the-architecture" play.

## Action

- Patch HYP-042b SUMMARY and doc to correct the "two kernels per
  layer per step" framing and the 9.8× number.
- Re-order the HYP-042b next-steps table to put HYP-044 at the top.
- No code change from this hypothesis.
