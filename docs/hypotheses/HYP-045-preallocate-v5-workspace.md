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

## Prediction

- All 12 HYP-041 configs run end-to-end (no OOMs)
- At seq=8192 × b=8, decode tok/s improves by **2–4 %** (0.4–0.8 ms
  less CPU tail per step)
- Peak GPU memory for TQ drops below baseline at matched configs (the
  PR #39868 3.20× cache compression finally materializes in RSS)

## Method

1. Read `_get_v5_ws` in `turboquant/vllm_backend_fused.py`; enumerate
   the tensors it allocates, their shapes, and lifetime.
2. Add a per-layer `WorkspaceCache` keyed on `(max_batch, max_seq)`,
   lazy-allocated on first decode call, reused afterwards.
3. Re-run HYP-041's 12-config sweep. Confirm no OOMs and record
   tok/s deltas.

## Status: pending
