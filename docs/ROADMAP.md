# Roadmap

Improve upstream vLLM v0.20.0's Triton TurboQuant kernels via a vLLM
plugin. Five phases, each gated on the prior phase's output. Only "Now"
phases are active targets; everything else is staged.

The accuracy gate, plugin architecture, and out-of-scope items are
defined in `docs/GOAL.md`. This document plans *what* to do; the
hypothesis docs in `docs/hypotheses/HYP-058+` capture each experiment.

---

## Now — Phase 1: Baseline lock

Reproduce HYP-057 on a fresh Forge run, capture reference nsys + ncu
traces under `--security-profile profiling-debug`, and scaffold the
per-kernel baseline reference doc that Phase 2 fills in.

- [x] **HYP-058** — Baseline lock (CONFIRMED 2026-04-23, Forge job
      `fb2e708a`). Accuracy + 16-cell perf grid landed at byte-exact
      340/340 vs HYP-057. Profiling (nsys + ncu) deferred to HYP-059's
      setup because the custom image is incompatible with
      `profiling-debug`. See
      `docs/hypotheses/HYP-058-baseline-lock.md` for the full table.

Files touched: none in `vllm`/`turboquant/`. Only adds
`docs/hypotheses/HYP-058-*.md` and writes baseline TBD cells.

---

## Next — Phase 2: Kernel-level profiling

Analyse Phase 1's nsys + ncu output. Per-kernel warp-stall attribution,
occupancy ceiling, and ordered ROI list for Phase 3. No code; analysis only.

- [x] **HYP-059** — `_tq_decode_stage1` profiling (PARTIAL, 2026-04-23,
  Forge job `a10d3ae4`). nsys kernel-time attribution **confirms
  `_tq_decode_stage1` as dominant TQ kernel (79.4 % of TQ time)**. ncu
  warp-stall / occupancy / register sections blocked by Forge DCGM perf-counter
  contention — see HYP-059 §Forge job iterations. Phase 3 ROI ranking
  unblocked; warp-stall attribution stays a follow-up until DCGM
  coordination is resolved or H100 access lands.
- [x] **HYP-060** — folded into HYP-059. nsys data shows
  `_tq_fused_store_mse` is 12.3 % of TQ time / 0.6 % of total kernel
  time; HYP-064 prediction (0.5–1 % TPOT win) is right-sized — no
  separate analysis needed.
- [x] **HYP-061** — folded into HYP-059. `_tq_full_dequant_kv` is 8.3 %
  of TQ time on `small_balanced` (continuation prefill rare on this
  workload); deferred until a continuation-heavy bench surfaces.

**Output**: ranked optimization-axis table (one row per kernel × bottleneck
class × predicted ROI). This becomes Phase 3's HYP order.

Files touched: only updates to
`docs/reference/upstream-triton-kernel-baseline.md` + new
`docs/hypotheses/HYP-059..061-*.md`.

---

## Then — Phase 3: Memory-hierarchy optimization

Plugin-monkey-patch experiments on the upstream Triton kernels. Each HYP
must clear the SHA-256 parity gate (or document a mathematical opt-out
per `GOAL.md` §1) AND the paired nsys/ncu trace requirement before merge.

Plugin package layout (incremental — files added per HYP, not all at once):

```
turboquant/
├── vllm_plugin.py                 # extend register() with _patch_triton_kernels()
├── kernels/
│   ├── decode_stage1.py           # HYP-062/063/065 target
│   ├── store_mse.py               # HYP-064 target
│   └── store_fp8.py               # (future)
└── dispatch.py                    # SM-tier config selection (Phase 4)
```

`pyproject.toml` already declares the entry point
`[project.entry-points."vllm.plugins"] turboquant = "turboquant.vllm_plugin:register"`.

- [x] **HYP-062** — Joint retune of `(num_warps, num_stages, BLOCK_KV)`
  REJECTED 2026-04-23 (Forge job `2d616eee`). 27-cell sweep found
  upstream defaults `(1, 1, 4)` are at the local optimum within this
  grid; best variant lands at +0.20 % TPOT (noise) vs 8–15 % predicted.
  See `docs/hypotheses/HYP-062-decode-stage1-launch-retune.md` for the
  full table + Amdahl analysis (TQ kernels are only 4.7 % of total
  decode time at this cell, capping kernel-only ROI). **Plugin
  infrastructure works** (parity 85/85 byte-exact, monkey-patch
  verified) — `turboquant/kernels/decode_stage1.py` +
  `turboquant/vllm_plugin.py` are merged as the foundation HYP-063+
  reuse, defaulting to upstream's `(1, 1, 4)` when env vars are
  unset (so the merge is a no-op without `TQ_PATCH_DECODE=1`).

- [ ] **HYP-065** — Adaptive `NUM_KV_SPLITS` per batch-size bucket
  (**promoted** from "next" to first Phase 3 code HYP after HYP-062).
  Different lever than HYP-062 — changes the GRID, not block params.
  Targets the largest absolute gap cell from HYP-058
  (`3bit_nc × s8192 × c8` +35 ms vs fp16) where SM saturation is
  plausibly the binding constraint.
  - Plugin patches `TurboQuantMetadataBuilder.build` (method-level
    monkey-patch).
  - **Predicted impact**: 3–6 % TPOT at `concurrency ≥ 8`.
  - **Gate (opt-out)**: SHA-256 may break — reduction order changes.
    `mean_score within ±0.002 pp per task` with mathematical
    justification.
  - **Files**: `turboquant/vllm_plugin.py` (extend
    `_patch_triton_kernels()` to also patch
    `TurboQuantMetadataBuilder.build`).

- [ ] **HYP-063** — SMEM pre-stage of centroids at decode-kernel entry.
  **Lowered priority after HYP-062.** Different mechanism than HYP-062
  (replaces per-tile HBM gather with one-time SMEM stage), so HYP-062's
  rejection doesn't predict HYP-063's outcome — but Amdahl bound
  applies: even a perfect HYP-063 caps at ~2 % TPOT improvement at
  the `s8192 × c1` cell. Worth trying after HYP-065 if/when we add a
  workload where TQ kernels are >10 % of total time.
  - **Predicted impact**: 2–4 % TPOT (revised down post-HYP-062 ceiling).
  - **Gate**: SHA-256 parity preserved (no math change).
  - **Files**: `turboquant/kernels/decode_stage1.py` (extend).

- [ ] **HYP-064** — Midpoints pre-load in `_tq_fused_store_mse`.
  **Demoted to research-only after HYP-059/062.** HYP-059 confirmed
  `_tq_fused_store_mse` is 0.6 % of total kernel time. Improvement
  ceiling ~0.3 % TPOT — below noise floor. Skip unless an upstream PR
  is otherwise blocked and we want to land a small clean diff.
  - **Predicted impact**: ≤ 0.3 % TPOT.
  - **Files**: `turboquant/kernels/store_mse.py` (new, deferred).

---

## Later — Phase 4: Arch-aware async dispatch (blocked on H100/H200 quota)

SM-tier launch-config dispatch and SM90+-only async paths. Both HYPs
opt out of SHA-256 parity (fp16 roundoff differs from fp32). Both
target an additional 15–25 % TPOT on A100 if accuracy holds, plus
~5 % more on SM90+.

- [ ] **HYP-066** — `tl.dot` QK with fp16 accumulator on the decode path.
  - **Predicted impact**: 8–12 % TPOT on A100; opens TMA path on SM90+.
  - **Gate (opt-out)**: SHA-256 will break — fp16 accumulate ≠ fp32.
    Use `mean_score within ±0.002 pp per task`. Justified by error-bound
    analysis in the HYP doc.
- [ ] **HYP-067** — `tl.dot` V accumulate, plus `cp.async.bulk.tensor`
  (TMA) load path for SM90+.
  - **Predicted impact**: 5–10 % TPOT additive on H100/H200.
  - **Gate**: same opt-out as HYP-066.
- [ ] **`dispatch.py`** — SM-tier config selection (A100 / H100 / B200);
  monkey-patches the launch wrapper to pick the right kernel variant
  per `torch.cuda.get_device_capability()`.

Blocked: no H100/H200 access on the team's Forge quota. Re-evaluate
when hardware lands.

---

## Later — Phase 5: Upstream contribution

Each Phase 3 confirmed HYP becomes a single-topic upstream PR. The plugin
remains the integration test surface; the PR is the shipping vehicle.

- [ ] **HYP-068** — Land HYP-062 (joint launch-config retune) upstream.
  PR template: one Triton diff + bench script + regression test.
- [ ] One PR per confirmed HYP afterwards (HYP-063 / 064 / 065 / 066 /
  067 as they confirm). Bundle only when topics are inseparable
  (e.g. HYP-062+063 if the retune ROI depends on the centroid pre-stage).

---

## First-week Forge jobs (in order; each gated on prior success)

1. **Job 1 — Phase 1 baseline** (`--security-profile profiling-debug`,
   1 GPU, ~2 h). Bench grid + nsys + ncu. Writes `/workspace/shared/hyp058_phase1/`.
2. **Job 2 — Phase 2 analysis** (local, no GPU). Read Job 1 outputs;
   fill baseline doc TBDs; decide Phase 3 ordering.
3. **Job 3 — HYP-062 joint sweep** (default profile, 1 GPU, ~1 h).
   27-cell `(num_warps, num_stages, BLOCK_KV)` × 3 surviving presets.
   Pick per-preset winner; SHA-256 parity gate; 16-cell perf re-measure.
