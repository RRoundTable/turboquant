# HYP-069: HYP-062 retry — `_tq_decode_stage1` launch sweep at `s32k × c1`

Re-run of HYP-062's 27-cell `(num_warps, num_stages, BLOCK_KV)` sweep
at the new primary Phase 3 cell that HYP-068 surfaced. Same plugin
code, same kernel-body invariant; only the perf cell changes.

## Hypothesis

HYP-062 returned +0.2 % at `s8k × c1` because TQ kernels were 4.7 % of
total decode time → Amdahl ceiling ~2 %. HYP-068 measured TQ_extra at
`s32k × c1` to be **45 %** of TQ TPOT for `4bit_nc` and **53 %** for
`3bit_nc`. The same sweep at this cell should land a real winner.

## Prediction

At `4bit_nc × s32k × c1` (HYP-068 baseline 28.82 ms):
- **Primary pass**: any config improves TPOT ≥ 5 %.
- **Stretch pass**: best config improves ≥ 12 %.
- Most likely sweet spot — based on HYP-062 trend (`w1-s3-b4` was the
  noise-best at `s8k`) — `w1-s2-b4` or `w1-s3-b4` at `s32k`. But
  `BLOCK_KV ∈ {8, 16}` may now win because longer per-CTA loops can
  amortize larger tile setup costs.

At `3bit_nc × s32k × c1` (HYP-068 baseline 33.31 ms): expect similar
or slightly bigger relative win because TQ_extra ratio is higher (53 % vs 45 %).

## Pass / fail

- **Primary pass**: any config improves TPOT ≥ 5 % at `4bit_nc × s32k × c1`.
- **Stretch pass**: ≥ 12 %.
- **Soft pass**: any config improves ≥ 2 % AND no regression > 1 % at
  `s8k × c1` (the regression-guard cell, sanity-checked at the winner).
- **Kill**: best config improvement is < 2 %. Kernel-only optimization
  is fundamentally bounded on A100 even where Amdahl room exists.
  Pivot to HYP-063 (centroid SMEM pre-stage, different mechanism)
  or wait for H100/H200/B200.
- Parity: SHA-256 byte-exact preserved by construction (same kernel
  body, only launch config changes — confirmed by HYP-062's 85/85).
  We don't re-verify here.

## Method

Identical to HYP-062's plugin path (already shipped in
`turboquant/kernels/decode_stage1.py` + `turboquant/vllm_plugin.py`).
Only changes:
- Perf cell: `4bit_nc × s32k × c1` (instead of `s8k × c1`).
- Add `3bit_nc × s32k × c1` for breadth.
- Add winner-only sanity at `4bit_nc × s8k × c1` (regression-guard).

Sweep config: 27 cells (`num_warps ∈ {1,2,4} × num_stages ∈ {1,2,3} ×
BLOCK_KV ∈ {4,8,16}`) × 2 presets (`4bit_nc, 3bit_nc`) × `s32k × c1`
= 54 perf cells.

Per-cell ~3 min (s32k × c1 is decode-light, just longer per-step).
Total: ~3 h.

If too long, can drop `3bit_nc` second pass and only re-add at the
winner config — saves half. Going with full 54 first to capture
preset-dependent winners cleanly.

## Status: REJECTED — kernel-only retune is exhausted on A100

The launch-retune mechanism has no headroom left, even at the cell where
TQ kernels are 45% of TQ TPOT. Best variant `w1-s1-b4` lands within
**+0.05%** of the HYP-068 baseline (28.81 ms vs 28.82 ms — noise). All
other 26 cells regress by 2.6 % – 35 %. Same outcome as HYP-062 at
s8k × c1: upstream's `(num_warps=1, num_stages=1, BLOCK_KV=4)` is the
local optimum within this grid, regardless of workload region.

The Amdahl-ceiling pivot motivated by HYP-068 was correct *as a
diagnosis* (kernel time is no longer negligible at long context) but
the retune axis itself is saturated — there is no `(num_warps,
num_stages, BLOCK_KV)` triple in `{1,2,4} × {1,2,3} × {4,8,16}` that
beats upstream meaningfully.

### Forge run

- Job: `acbdd74d` (`tq-hyp069-sweep-s32k`, SUCCEEDED 2026-04-24).
- Image: custom (`ce745b54`), default profile (no nsys/ncu — kernel
  body unchanged from HYP-062, so the SHA-256 parity gate carries over
  for free; nothing new to verify).
- 27 launch-config cells × `4bit_nc × s32k × c1`, plus winner sanity
  at `3bit_nc × s32k × c1` and `4bit_nc × s8k × c1`.

### Results — Phase A grid (`4bit_nc × s32k × c1`)

Baseline (HYP-068, no plugin): TPOT 28.82 ms.

| cell | TPOT (ms) | Δ vs baseline | tput (tok/s) |
|---|---:|---:|---:|
| `w1-s1-b4` (= upstream defaults) | **28.81** | **+0.05 %** | 15.5 |
| `w1-s2-b4` | 28.81 | +0.05 % | 16.0 |
| `w1-s3-b4` | 28.81 | +0.03 % | 16.0 |
| `w1-s2-b8` | 29.58 | -2.65 % | 15.8 |
| `w1-s1-b8` | 29.60 | -2.71 % | 15.3 |
| `w1-s3-b8` | 29.60 | -2.71 % | 15.8 |
| `w2-s2-b8` | 30.36 | -5.35 % | 15.6 |
| `w2-s1-b8` | 30.38 | -5.43 % | 15.6 |
| `w2-s3-b8` | 30.39 | -5.44 % | 15.6 |
| `w2-s3-b4` | 31.78 | -10.28 % | 15.3 |
| `w2-s1-b4` | 31.81 | -10.37 % | 15.3 |
| `w2-s2-b4` | 31.81 | -10.38 % | 15.3 |
| `w4-s1-b4` | 33.28 | -15.48 % | 15.0 |
| `w4-s3-b4` | 33.28 | -15.49 % | 15.0 |
| `w4-s2-b4` | 33.30 | -15.53 % | 14.9 |
| `w2-s3-b16` | 34.62 | -20.11 % | 14.6 |
| `w2-s2-b16` | 34.64 | -20.18 % | 14.7 |
| `w2-s1-b16` | 34.64 | -20.21 % | 14.7 |
| `w4-s2-b8` | 34.81 | -20.79 % | 14.6 |
| `w4-s1-b8` | 34.82 | -20.82 % | 14.6 |
| `w4-s3-b8` | 34.84 | -20.88 % | 14.6 |
| `w4-s1-b16` | 36.35 | -26.14 % | 14.3 |
| `w4-s2-b16` | 36.36 | -26.17 % | 14.3 |
| `w4-s3-b16` | 36.38 | -26.22 % | 14.3 |
| `w1-s1-b16` | 38.98 | -35.25 % | 13.4 |
| `w1-s3-b16` | 38.98 | -35.25 % | 13.8 |
| `w1-s2-b16` | 38.99 | -35.29 % | 13.8 |

Pattern is monotone:
- `BLOCK_KV` is the dominant axis. `b=4` is optimal; `b=8` already costs
  3–5 %; `b=16` is catastrophic (-20 % to -35 %). Larger tiles increase
  per-CTA register pressure and SMEM staging without amortizing across
  enough iterations at decode (only one query per step).
- `num_warps=4` always regresses. The kernel doesn't have enough
  parallel work per CTA at decode-time — adding warps just adds
  scheduling overhead.
- `num_stages` is noise within fixed (`num_warps`, `BLOCK_KV`).

### Phase B — winner × `3bit_nc × s32k × c1`

| variant | TPOT median (ms) | Δ vs HYP-068 baseline (33.31 ms) |
|---|---:|---:|
| `w1-s1-b4` (= upstream defaults) | 33.31 | 0.0 % |

No-op, as expected — winner *is* the upstream default.

### Phase C — winner × `4bit_nc × s8k × c1` (regression guard)

| variant | TPOT median (ms) | Δ vs HYP-058 baseline (17.5 ms) |
|---|---:|---:|
| `w1-s1-b4` (= upstream defaults) | 17.45 | +0.3 % |

No regression. (Trivially so — same kernel binary as upstream.)

### Why HYP-068's prediction was right but the experiment still failed

HYP-068 measured TQ_extra/TQ = 45 % at this cell, implying a perfect
50 % kernel-time cut would yield ~22 % TPOT win. That math is correct,
but it requires a *kernel-time-cut* mechanism. Launch-config retune
within `{warps, stages, BLOCK_KV}` is not such a mechanism here —
upstream's choice already minimizes per-CTA overhead for this
decode shape on SM80.

The candidate optimization mechanisms that *could* still cut kernel
time at this cell:

1. **Centroid SMEM pre-stage (HYP-063)**: replaces per-tile HBM gather
   of K/V centroids with a one-time SMEM stage. Different bottleneck
   (HBM bandwidth vs warp scheduling) — orthogonal to the launch grid.
2. **`tl.dot` QK with fp16 accumulator (HYP-066)**: opt out of SHA-256
   parity, use tensor cores. Targets compute throughput rather than
   launch overhead. Predicted 8–12 % on A100.
3. **Cross-step persistence**: keep K/V tile in SMEM across decode
   steps when the same KV slot is re-read. Out of plugin scope (needs
   stateful kernel registration).

### Implications

- **HYP-063 is the next ROI move on A100**, not a deeper sweep of
  HYP-062-style launch grids. Predicted impact stays at 4–8 % at this
  cell (HYP-068 §"Implications"). Worth running.
- **Phase 3 launch-config track is closed** for A100. Don't extend the
  grid beyond `{1,2,4} × {1,2,3} × {4,8,16}` — the regression pattern
  shows we've already exited the basin of upstream's tuned point.
- Multi-arch note: H100/H200 may have a different launch-config optimum
  (different SM count, register file, warp scheduler). Re-run this
  same 27-cell sweep when hardware lands — the rejection here doesn't
  predict H100/H200 outcomes. The plugin code is already in place to
  flip launch params via env vars.
- B200 (FP4 native) likely makes this whole question moot.

### Artefacts

- `/workspace/shared/hyp069_sweep/` (Forge NFS): 29 perf JSONs +
  `hyp069_run.log` + `winner.txt`.
- Local mirror: `results/hyp069/perf_grid/` + `sweep_summary.csv` +
  `hyp069_run.log`.
- Forge job ID: `acbdd74d`.

### Follow-ups

- **[immediate]** HYP-063 (centroid SMEM pre-stage) at `s32k × c1`. The
  only A100 mechanism left in Phase 3 with predicted Amdahl-respecting
  headroom. Different bottleneck than this HYP — HBM gather cost, not
  launch overhead.
- **[doc]** Update ROADMAP.md to mark HYP-062 retry CLOSED and promote
  HYP-063 to active.
- **[deferred]** HYP-066/067 (tensor-core opt-out path) when we're
  willing to pay the SHA-256 byte-exact opt-out cost — different
  mechanism, much larger predicted win but breaks the parity gate.
