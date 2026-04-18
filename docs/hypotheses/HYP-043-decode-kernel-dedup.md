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

## Status: pending
