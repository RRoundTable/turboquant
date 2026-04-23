# Goal

Improve upstream vLLM v0.20.0's Triton TurboQuant kernels on A100 (SM80)
while preserving bit-exact LongBench accuracy, and establish the baseline
for extending to SM90+ once hardware is available.

The work ships as an installable vLLM plugin that monkey-patches specific
Triton kernel entry points in upstream vLLM
(`vllm.v1.attention.ops.triton_turboquant_decode._tq_decode_stage1`,
`vllm.v1.attention.ops.triton_turboquant_store._tq_fused_store_mse`, etc.)
with optimized variants. The user-visible CLI surface
(`--kv-cache-dtype turboquant_*_nc`) is unchanged; only performance
differs.

## Background

HYP-057 confirmed that upstream vLLM v0.20.0's native
`turboquant_*_nc` Triton kernels (PR #38479, refined by PR #40194)
reproduce the paper's fp16-parity claim on LongBench with a non-QJL
recipe — three of four upstream presets land within ±1.5 pp of fp16 on
Llama-3.1-8B-Instruct (`small_balanced`).

HYP-055c rejected the paper-faithful QJL recipe on both Qwen3-8B and
Llama-3.1-8B (−7.44 pp at 3.5-bit vs the simpler regulars-only variant),
corroborating upstream's QJL omission. The custom-kernel track ran 50+
hypotheses against `csrc/**` + `turboquant/decode_kernel*.py` +
`turboquant/vllm_backend_fused.py`; that lineage is now archived under
`docs/hypotheses/archive/custom-kernel/` and `docs/reference/archive/custom-kernel/`.

Upstream's current Triton kernels leave several memory-hierarchy axes
untouched: `num_warps=1, num_stages=1, BLOCK_KV=4` at
`_tq_decode_stage1:586-587,552`; centroid gather from HBM every tile at
`triton_turboquant_decode.py:193-197`; midpoints reloaded every
binary-search iteration at `triton_turboquant_store.py:290`. These are
the targets.

## Success Criteria

1. **Bit-exact LongBench accuracy** on Llama-3.1-8B-Instruct
   `small_balanced` per HYP-057 baseline. Default gate is **SHA-256 of
   prediction token strings** matching the baseline byte-for-byte per
   (preset, task, sample). A specific HYP may relax to
   `mean_score within ±0.002 pp per task` *iff* the HYP doc contains a
   written mathematical justification why the optimization is
   parity-preserving in infinite precision (reduction-order change,
   different split count, tensor-core fp16 accumulate). Burden of proof
   is on the HYP.
2. **≥ 15 % TPOT reduction** at `seq=4096 batch=1` on at least one
   confirmed optimization.
3. **No regression > 3 %** on TPOT at shorter seq ∈ {512, 1024, 2048}
   under default load (concurrency 1).
4. **Every change backed by paired nsys + ncu traces** (before/after)
   captured under Forge `--security-profile profiling-debug`. The
   dominant warp-stall class shift in the ncu delta must match the HYP's
   prediction.
5. **Each confirmed optimization is single-topic upstream-PR-able** —
   one Triton kernel diff (or one launch-config change) with bench +
   regression test attached. Plugin form is the development vehicle;
   upstream PR is the optional shipping vehicle.

## Out of Scope

These directions are explicitly frozen and not targets of this track:

- **Custom CUDA kernels** (`csrc/**`, `turboquant/decode_kernel*.py`).
  The custom-kernel track is archived; the production target is
  upstream Triton.
- **QJL reintroduction** (any form: dense Gaussian, structured Hadamard
  JL, residual JL on outliers, …). HYP-049/050/052/054/055c rejected
  QJL on our stack; upstream omits it; HYP-057 confirms fp16 parity is
  reachable without it.
- **Outlier-aware mixed precision.** Our HYP-053/054/055c outlier-aware
  variants and HYP-056 outlier-aware CUDA kernel are not part of the
  upstream track. Reconsider only if a clean upstream PR proposes it as
  a new preset.
- **Our `turboquant/vllm_backend_fused.py` plugin path.** The
  pre-pivot plugin shipped a CUDA decode kernel competing with FlashInfer.
  All new plugin work targets the *upstream Triton kernels*, not a
  competing CUDA backend.
- **`turboquant_k8v4` on A100 debugging.** Upstream's FP8 path is broken
  on A100 (HYP-057 §k8v4); an upstream issue is the right venue, not
  this track.
- **`docker/vllm_patches/` overlay.** v0.20.0 merged the custom-page-size
  hook natively (`v1/kv_cache_interface.py +26 LOC`); the overlay is
  redundant on v0.20.0+.
- **vLLM source forks of any kind.** Plugin-only — no source edits.
- **H100/H200/B200 experiments.** Blocked on Forge hardware access;
  Phase 4 in `ROADMAP.md` is staged but gated.

## Reference

- Paper: Zandieh et al., "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate", arXiv:2504.19874, 2025 (ICLR 2026).
  See `docs/reference/turboquant-paper-methodology.md` for the §3/§4
  methodology extract.
- Upstream architecture: `docs/reference/vllm-upstream-turboquant-architecture.md`.
- Baseline confirmation: `docs/hypotheses/HYP-057-upstream-vllm-turboquant-longbench.md`.
