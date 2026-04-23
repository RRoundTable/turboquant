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

- [ ] **HYP-058** — Baseline lock (no code, pure measurement).
  - Bench grid: `{4bit_nc, k3v4_nc, 3bit_nc, fp16}` × `seq ∈ {1024, 8192}` ×
    `concurrency ∈ {1, 8}` = 16 cells, via
    `tests/bench_serve_upstream_entry.sh` adapted to v0.20.0.
  - Profile subset: `4bit_nc × seq=8192 × concurrency=1` with nsys
    (`-t cuda,nvtx`) and ncu
    (`--section WarpStateStatistics,SpeedOfLight,MemoryWorkloadAnalysis`,
    `--kernel-name regex:_tq_.*`).
  - Populate the TBD cells in
    `docs/reference/upstream-triton-kernel-baseline.md`.
  - **Gate**: bench JSON + nsys + ncu archived under
    `/workspace/shared/hyp058_phase1/`. SHA-256 of fp16 prediction
    strings recorded as the parity reference.

Files touched: none in `vllm`/`turboquant/`. Only adds
`docs/hypotheses/HYP-058-*.md` and writes baseline TBD cells.

---

## Next — Phase 2: Kernel-level profiling

Analyse Phase 1's nsys + ncu output. Per-kernel warp-stall attribution,
occupancy ceiling, and ordered ROI list for Phase 3. No code; analysis only.

- [ ] **HYP-059** — `_tq_decode_stage1` warp-stall attribution.
  Identify dominant stall class (`long_scoreboard` / `short_scoreboard` /
  `math_pipe_throttle` / `wait`); occupancy ceiling at default config
  (`num_warps=1, num_stages=1, BLOCK_KV=4`).
- [ ] **HYP-060** — `_tq_fused_store_mse` warp-stall attribution.
  Quantify the redundant midpoint loads in the binary-search loop
  (`triton_turboquant_store.py:290`).
- [ ] **HYP-061** — `_tq_full_dequant_kv` profile (continuation prefill).
  Occupancy + memory bandwidth for the bulk-dequant path.

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

- [ ] **HYP-062** — Joint retune of `(num_warps, num_stages, BLOCK_KV)` on
  `_tq_decode_stage1` via patched launch wrapper.
  - Default: `num_warps=1, num_stages=1, BLOCK_KV=4`.
  - Sweep: `num_warps ∈ {1,2,4}` × `num_stages ∈ {1,2,3}` × `BLOCK_KV ∈ {4,8,16}` = 27 cells × 3 surviving presets.
  - **Predicted impact**: 8–15 % TPOT reduction at `seq=8k, conc=1`.
  - **Gate**: `long_scoreboard` stall % must drop ≥ 30 % vs Phase 1 baseline (ncu observable). SHA-256 parity preserved.
  - **Files**: `turboquant/kernels/decode_stage1.py` (new),
    `turboquant/vllm_plugin.py` (`_patch_triton_kernels()` call).

- [ ] **HYP-063** — SMEM pre-stage of centroids at decode-kernel entry.
  Replaces per-tile HBM gather (`triton_turboquant_decode.py:193-197`)
  with select-chain on a register tensor staged once.
  - **Predicted impact**: 2–4 % TPOT additive on top of HYP-062.
  - **Gate**: SHA-256 parity preserved (no math change).
  - **Files**: `turboquant/kernels/decode_stage1.py`.

- [ ] **HYP-064** — Midpoints pre-load in `_tq_fused_store_mse`.
  Eliminate the 4× repeated load in the binary-search loop
  (`triton_turboquant_store.py:290`).
  - **Predicted impact**: 0.5–1 % TPOT via freed L2 bandwidth.
  - **Gate**: SHA-256 parity preserved (load-reorder only).
  - **Files**: `turboquant/kernels/store_mse.py` (new).

- [ ] **HYP-065** — Adaptive `NUM_KV_SPLITS` per batch-size bucket.
  Plugin patches `TurboQuantMetadataBuilder.build` (method-level
  monkey-patch) so the split count adapts to `(batch, kv_len)` instead
  of the constant default. Graph-capture-aware.
  - **Predicted impact**: 3–6 % TPOT at `concurrency ≥ 8`.
  - **Gate (opt-out)**: SHA-256 may break — reduction order changes
    when split count changes. Use `mean_score within ±0.002 pp per task`.
    Mathematical justification documented in HYP-065 doc.
  - **Files**: `turboquant/vllm_plugin.py` (extend
    `_patch_triton_kernels()` to also patch
    `TurboQuantMetadataBuilder.build`).

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
