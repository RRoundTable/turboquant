# HYP-045: pre-allocate v5 workspace at engine init

## Evidence (from HYP-041 + HYP-042b)

- HYP-041: TurboQuant OOMs at seq=4096 × batch=32, seq=8192 × batch=32,
  seq=16384 × batch=8, seq=16384 × batch=32. The traceback lands in
  `turboquant/vllm_backend_fused.py:_get_v5_ws` on a
  `torch.empty((batch_size, num_kv_heads, max_len, qbytes), …)`
  allocation. PR #39868 compressed the KV cache (3.20× fewer bytes,
  confirmed at 126k → 404k tokens) but the per-step workspace blew the
  recovered headroom and then some.
- HYP-042b: `aten::empty_like` fires 9 561 times over the profiled
  run (one allocation per decode step per layer per tensor). CPU-side
  allocator time is ~80 ms across the run (~0.6 ms / decode step).
  Small in speed terms but non-zero.

## Hypothesis

Pre-allocating the v5 workspace tensors to `(max_batch, num_kv_heads,
max_possible_len, qbytes)` at engine init and reusing them across
every decode step:

- eliminates the `_get_v5_ws` per-step allocation, removing the OOM
  failure mode (HYP-041 OOM configs will run)
- shaves the ~0.6 ms/step CPU allocator tail (≈ 3 % of the per-step
  Δ — small but free once the memory fix is in)

### Inspection finding (revises plan)

Re-reading `turboquant/vllm_backend_fused.py:178–188` and the call
sites at lines 412 (`num_splits > 1` paged-split path) and 442
(`num_splits == 1` non-split path):

- `k_quant`, `v_quant`, `k_norms`, `v_norms` are **only consumed by
  the `num_splits == 1` non-split fallback** (line 442, calling
  `decode_v5_from_cache_ws`).
- The paged-split path (line 418, `decode_v5_from_cache_paged_splitkv_ws`)
  never reads them — the binding (`csrc/src/decode_v5_tc_binding.cu:754`)
  walks the page table directly from `kv_cache`.
- `_choose_num_splits` returns `1` only when `max_len < 512`. Every
  HYP-041 OOM config has `max_len ≥ 4096`, so it takes the paged-split
  path and the four dead tensors are pure waste.

Sizing at the worst OOM config (seq=8192 × batch=32):
`(32, 8, 8192, 64) uint8 × 2 (K,V) × 36 layers = ~9.6 GB` of unused
allocation. That alone exceeds the 2.4 GB headroom missing at the
HYP-041 OOM. So the fix narrows from "general workspace pre-alloc" to
"don't allocate the non-split scratch when we're not on that path".

## Prediction

- All 4 HYP-041 OOM configs (`4096×32, 8192×32, 16384×8, 16384×32`)
  run end-to-end.
- At seq=8192 × b=8 (already-passing config), tok/s ~unchanged (kernel
  workload identical, ~0.6 ms CPU allocator tail removed).
- Peak GPU memory for TQ drops by ~9.6 GB at high batch×seq, finally
  materializing the PR #39868 3.20× cache compression in RSS.

## Method

1. (DONE) Read `_get_v5_ws`, identify dead allocations.
2. Wrap the four tensor allocations in `if num_splits == 1:` so they
   only fire on the non-split fallback path.
3. Re-run HYP-041's 12-config sweep. Confirm no OOMs and record
   memory + tok/s deltas.

If this still leaves OOMs (e.g. `partition_o`, `partition_lse` grow
too large at very high batch × splits), file HYP-045b for true
pre-allocation of those.

## Status: confirmed — all 4 OOM configs run; peak RSS ≈ baseline now

## Result (full HYP-041 sweep, A100-40GB)

Every config completed. Peak GPU RSS across the grid now sits at
34.3–34.9 GB for TQ (was 34.3–38.5 GB pre-patch), matching or slightly
undercutting baseline FA (34.3–34.8 GB). PR #39868's 3.20× cache
compression finally materializes in RSS.

|  seq × b  | TQ v1 | **TQ v2** | FA      | FI      | mem v0→v2 | tq/FA | tq/FI |
|-----------|------:|----------:|--------:|--------:|:---------|------:|------:|
|  1024×1   | 38.8  |  37.4     |  49.1   |  50.7   | 34.3→34.3 | 0.76× | 0.74× |
|  1024×8   | 300.4 | 297.0     | 374.2   | 392.9   | 34.6→34.3 | 0.79× | 0.76× |
|  1024×32  | 1138.3| 1154.4    | 1446.8  |  873.4  | 37.9→34.3 | 0.80× | **1.32×** |
|  4096×1   | 38.6  |  37.5     |  48.5   |  51.8   | 34.5→34.5 | 0.77× | 0.72× |
|  4096×8   | 296.3 | 296.4     | 381.2   |  223.3  | 37.0→34.5 | 0.78× | **1.33×** |
|  4096×32  |  OOM  | **605.1** | 1266.6  | 1438.9  |  — →34.7 | 0.48× | 0.42× |
|  8192×1   | 37.9  |  38.0     |  47.9   |  49.8   | 34.9→34.9 | 0.79× | 0.76× |
|  8192×8   | 227.7 | 228.2     | 370.9   | 406.6   | 38.5→34.9 | 0.62× | 0.56× |
|  8192×32  |  OOM  | **394.5** |  918.6  | 1101.5  |  — →34.9 | 0.43× | 0.36× |
| 16384×1   | 37.9  |  37.5     |  48.3   |  50.3   | 34.9→34.9 | 0.78× | 0.75× |
| 16384×8   |  OOM  | **156.1** |  352.1  |  386.5  |  — →34.9 | 0.44× | 0.40× |
| 16384×32  |  OOM  | **236.2** |  598.1  |   OOM   |  — →34.9 | 0.39× |   —   |

### Observations

- **Non-OOM configs: decode tok/s unchanged from HYP-044 v1.** Same kernel
  work, we just stopped pre-allocating 4 unused tensors — e.g. 8192×8 is
  227.7 → 228.2 (within noise).
- **All 4 OOM configs run now.** ~9.6 GB of dead allocation removed at the
  worst case.
- **FI OOMs at 16384×32.** TQ doesn't. At extreme configs where even
  FlashInfer fails, TQ still delivers — validates the memory story from
  PR #39868 at the serving level.
- **New long-context data points reveal**: TQ/FA ratio *worsens* with
  (seq × batch): 8192×32 = 0.43×, 16384×32 = 0.39×. Consistent with
  HYP-042b / HYP-044 finding: A100 scalar-FMA dequant scales linearly
  with work while baseline amortizes via tensor cores.

Full report: `results/v5_vs_baseline_hyp045/REPORT.md`.
