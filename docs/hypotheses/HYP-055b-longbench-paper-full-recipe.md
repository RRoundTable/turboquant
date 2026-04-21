# HYP-055b: LongBench — paper's full recipe (outlier-aware + QJL on regulars)

## Context

HYP-055 (in flight) tests uniform `TurboQuantProd(2)` on all 128 channels.
Preliminary results (qasper 0.02 vs fp16 0.54, hotpotqa 0.01 vs 0.59) show
catastrophic regression — but the configuration **is not the paper's
recipe**. Paper §4.3 always applies QJL only to "regular" low-variance
channels after outlier extraction.

User review surfaced the exact mechanism:

> QJL-induced estimator variance is `(π/2m)·‖r‖²`. If QJL is applied
> uniformly, outlier channels' huge post-1-bit-MSE residual dominates
> `‖r‖²` → variance explosion → score corruption.

HYP-055's uniform configuration tests this failure mode, not whether
QJL helps in its intended regime. HYP-055b tests the intended regime.

## Hypothesis

At a 2.5-bit average KV cache budget with outlier-aware mixed precision
— 32 outlier dims @ 4-bit MSE + 96 regular dims @ `TurboQuantProd(2)` =
(1-bit MSE + 1-bit QJL) — **B' matches or exceeds A'** on LongBench
aggregate task scores, where:

- **A' = outlier-aware MSE-only** (HYP-053 recipe, already measured at
  `out_cos = 0.991` on synthetic score metric; task-level not yet tested)
- **B' = outlier-aware + QJL on regulars** (paper Algorithm 2 as §4.3
  actually applies it)

The difference A' → B' is the **incremental value of QJL** in its
theoretically correct regime: applied only to low-variance channels where
post-MSE residual is small.

## Prediction

Paper's Table 1 reports LongBench avg 49.44 at 2.5-bit avg outlier-aware
+ QJL, vs 50.06 at full-precision (−0.62 pt degradation). On our 5-task
subset (Qwen3-8B, 100 samples per task):

| method     | narrativeqa | qasper | hotpotqa | gov_report | passage |
|------------|-----------:|------:|---------:|-----------:|--------:|
| fp16       | 0.28       | 0.54  | 0.59     | 0.20       | 1.00    |
| A (unif 2b)| 0.00–0.10  | 0.12  | 0.01–0.05| 0.15       | 0.00–0.30 |
| B (unif 1b+QJL) | ~0    | 0.02  | ~0       | ~0         | ~0       |
| **A'** (outlier MSE-only) | **≥ 0.22** | **≥ 0.40** | **≥ 0.45** | **≥ 0.17** | **≥ 0.85** |
| **B'** (outlier + QJL on regs) | **A' + ?** | **A' + ?** | **A' + ?** | **A' + ?** | **A' + ?** |

Whether B' beats A' at the task level is the actual question QJL needs
to answer.

## Method

### Configuration

Per head (calibrate once on first 4 k K/V tokens per layer):
- **32 outlier dims** (highest K-variance): `TurboQuantMSE(32, bit_width=4)`.
  Pure MSE — no QJL on outliers, since HYP-050 showed QJL is vestigial at
  4-bit.
- **96 regular dims**: `TurboQuantProd(96, bit_width=2)`. This is
  `TurboQuantMSE(96, 1) + QJL(96)` — paper's Alg 2 at the small-residual
  regime.

Effective bit-rate: (32·4 + 96·2) / 128 = **2.5 bits/dim** (same as HYP-053,
same as paper).

### Benchmark

Reuse `tests/bench_longbench_kv_quant.py` from HYP-055. Add two new modes:

- **A' (outlier-aware MSE-only):** apply per-layer K/V outlier mask;
  outliers go to `TurboQuantMSE(32,4)`, regulars to `TurboQuantMSE(96,2)`.
- **B' (outlier-aware + QJL on regulars):** same mask; outliers to
  `TurboQuantMSE(32,4)`, regulars to `TurboQuantProd(96,2)`.

Calibration K/V source: reuse `/workspace/shared/hyp050_kv_real.pt` to
derive the outlier masks for each layer. The bench attention hook then
applies the quantization at runtime using those precomputed masks.

### Forge run plan

1. **Smoke first** (~2 min): qasper 10 samples on A' and B' to confirm
   the hook + mask path works and scores are meaningfully positive
   (not ~0 like HYP-055 B).
2. **small_balanced** (~35 min per method on B-like speeds):
   85 samples total on A' and B' simultaneously. Compare against
   HYP-055's fp16@85 and A@85 for cross-reference.
3. **Decision at small_balanced**:
   - If B' − A' ≥ 1.5 pt on ≥ 3 of 5 tasks → QJL reinstated, file Gate 1 kernel work.
   - If B' − A' < 1.5 pt or regresses → paper's Algorithm 2 is not a
     load-bearing improvement over outlier MSE-only on Qwen3-8B.
     Ship HYP-053's outlier MSE-only as the sub-4-bit option.

### Engineering note

`reconstruct_outlier_plus_qjl` from HYP-054's test is already written in
the worktree. Generalize it to accept either `TurboQuantMSE` or
`TurboQuantProd` for the regular tier (A' vs B'), and plug it into the
attention hook in place of the current `polarquant_roundtrip`.

## Pass / fail

**Primary:** `B' − A' ≥ 1.5 pt` on ≥ 3 of 5 tasks (task-level
confirmation that QJL's unbiasedness helps softmax decisions).

**Secondary:** `fp16 − B' ≤ 2 pt` on ≥ 3 of 5 tasks (paper's
4.5×-compression-at-marginal-loss envelope reproduced).

**Kill:** B' strictly worse than A' on every task (QJL adds variance
even in its intended regime — closes the QJL story for good with a much
stronger claim than HYP-055 alone).

## Relationship to prior hypotheses

- HYP-049 / 050 / 052 / 054: all uniform-QJL configurations, all rejected.
  Generalization to "QJL is harmful at uniform bit budgets" stands.
- HYP-053: outlier-aware MSE-only, partial pass at 0.991 `out_cos` on
  synthetic metric — becomes A' here at the task level.
- HYP-055: uniform TurboQuantProd(2), task-level confirmation that
  uniform QJL destroys LongBench scores. Finishes in parallel; result
  feeds A and B rows of the comparison table.
- HYP-055b (this): first test of paper's exact recipe on real tasks.
  Distinguishes "QJL never helps" from "QJL helps only inside the outlier
  wrapper."

## Status: pending

Queued behind HYP-055's sweep completion so fp16@85 and A@85 rows are
available for comparison.
