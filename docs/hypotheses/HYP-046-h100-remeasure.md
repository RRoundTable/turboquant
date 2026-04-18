# HYP-046: re-measure the whole HYP-041 / HYP-042b stack on H100

## Evidence (from HYP-037, HYP-040, HYP-041, HYP-042b)

- HYP-037 / HYP-040: A100 SM80 has no async `ldmatrix` variant;
  smem → mma data-dependency stall (~688 cycles) is irreducible.
  Raw-PTX rewrite gave 0 % speedup. This is an architectural ceiling,
  not a kernel bug.
- HYP-041 required `enforce_eager=True` because vLLM's fp8 path lowers
  to `fp8e4nv`, which A100 cannot torch.compile (`ValueError: type
  fp8e4nv not supported in this architecture`). Setting
  `kv_cache_dtype='fp8_e5m2'` was rejected by `TurboQuantBackend`.
- HYP-042b showed 95 % of the per-step Δ is the attention kernel and
  that A100 kernel perf is at that ceiling.

H100 / H200 (SM90) have:
- native async `ldmatrix` (`ldmatrix.sync.aligned.m8n8.x4.shared.b16`
  via wgmma descriptors), which lets the smem → mma stall hide behind
  other instructions
- `fp8e4nv` hardware support → vLLM compiles the fp8 path → CUDA graphs
  available → `enforce_eager=True` constraint lifts
- more SMs (132 on H100 SXM vs 108 on A100 SXM) → batch-8 split-K fits
  better without rework (may partially absorb HYP-044)

## Hypothesis

Moving the exact HYP-041 / HYP-042b setup to H100 (or H200) drops the
serving-level gap from 1.77× (A100 eager) to **≤ 1.2× (H100 graphs)**
and the per-step attention-kernel ratio from 9.8× to **≤ 3×**.

## Prediction

Same sweep grid as HYP-041 (seq ∈ {1024, 4096, 8192, 16384} × batch ∈
{1, 8, 32}), Qwen3-8B, CUDA graphs enabled:

- H100: TQ ≥ 0.85× baseline tok/s in every config
- H100: no OOMs even without HYP-045 (the compressed cache
  materializes in RSS because the workspace is relatively smaller
  vs 80 GB)
- Per-decode-step attention ratio drops to HYP-035-at-batch-1 levels
  (~2.7×) or better thanks to graphs

## Method

Blocked on H100 / H200 access. When available:

1. Build the tq-hyp029 image on a SM90 base.
2. Keep `enforce_eager=False` for both backends (let torch.compile
   apply).
3. Re-run HYP-041's 12-config sweep unchanged.
4. Re-run HYP-042b's profile at seq=8192 × batch=8.
5. File the results as HYP-046a (serving sweep) and HYP-046b (profile).

## Status: pending (hardware gated)
