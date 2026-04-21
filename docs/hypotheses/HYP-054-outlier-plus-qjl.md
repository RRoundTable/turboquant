# HYP-054: Outlier-aware + QJL on regular channels — paper's Alg 2 recipe

## Context

HYP-053 reproduced the paper's 4.5× compression point with outlier-aware
MSE-only: `out_cos = 0.991` at seq 32 k, 44 B/head/token. fp16 is 1.000;
the gap is 0.009. HYP-054 asks whether QJL — finally applied in its
intended regime (small residuals on low-variance channels, after outliers
are extracted) — closes that gap.

HYP-049/050/052 + the seed-sweep diagnostic showed QJL at uniform bit
budgets adds more variance than bias it removes: `Var[T_QJL] =
(π/2m)·‖q‖²·‖residual‖²` is proportional to `‖residual‖²`. At uniform
2-bit, residuals are huge and QJL hurts. At outlier-aware 2-bit on the
*regular* 96 channels, residuals are small because the high-variance
signal is routed to the outlier tier. This is the exact regime the
paper's §4.3 uses QJL in.

## Hypothesis

At a 2.5-bit average budget with outlier-aware mixed precision, replacing
the 96 regular-channel quantizer from `TurboQuantMSE(2)` to
`TurboQuantProd(2)` = 1-bit MSE + 1-bit QJL recovers the 0.009 `out_cos`
gap to fp16 parity, with negligible short-context regression.

## Prediction

| method                              | bits/dim avg | QJL? | `out_cos` @ 32 k (pred) |
|-------------------------------------|-------------:|------|------------------------:|
| fp16                                | 16           | —    | 1.000                   |
| MSE 4-bit (today's shipped)         | 4.0          | no   | 0.9995                  |
| **HYP-053** (outlier MSE-only)      | 2.5          | no   | 0.9908 (measured)       |
| **HYP-054** (outlier + QJL on regs) | 2.5          | yes  | **≥ 0.997**             |

Target: HYP-054 ≥ 0.997 at seq ≥ 16 k (close ≥ 60 % of HYP-053's 0.009
gap to fp16).

## Configuration

Per head:
- **Outliers** (32 dims, highest variance): `TurboQuantMSE(bit_width=4)`
  — same as HYP-053. QJL is vestigial at b=4 (HYP-050), don't add it.
- **Regulars** (96 dims): `TurboQuantProd(bit_width=2)` = 1-bit MSE + 1-bit QJL.
  The residual QJL corrects now comes from low-variance channels only;
  `‖residual_reg‖² ≪ ‖residual_uniform_2bit‖²`, so QJL variance is
  correspondingly small.

Bit budget:
```
32 × 4 bits (outliers) + 96 × 2 bits (regulars) = 128 + 192 = 320 bits
320 / 128 dims = 2.5 bits/dim  ← same as HYP-053
```

Memory:
```
outliers:    32 × 4/8 = 16 B quant + 2 B MSE norm     = 18 B
regulars:    96 × 1/8 = 12 B MSE quant
           + 96 × 1/8 = 12 B QJL signs
           + 2 B MSE norm + 2 B residual norm          = 28 B
total raw:                                              46 B
16-B aligned:                                           48 B  (same tile bucket as HYP-053)
```

## Method

### Gate 0 — Python reference on cached real K/V

1. Extend `tests/test_qjl_long_context_bias.py` in the worktree to add a
   fifth method `Outlier_plus_QJL_2_5bit`: same outlier mask calibration
   as HYP-053, regular-tier quantizer swapped to `TurboQuantProd(2)`.
2. `turboquant/quantizer.py` already has `TurboQuantProd`. Use it
   directly on the regular channels — no modification to the existing
   production code.
3. Compare five methods at seq ∈ {1 k, 4 k, 16 k, 32 k} on
   `/workspace/shared/hyp050_kv_real.pt`:
   - fp16 (reference)
   - MSE 4-bit (today, baseline)
   - MSE 2-bit uniform (HYP-052 baseline)
   - Outlier 2.5-bit MSE (HYP-053, the measured 0.991 row)
   - **Outlier + QJL 2.5-bit** (this hypothesis)
4. Metrics: same as prior (`score_cos`, `abs_err`, `bias`,
   `softmax_cos`, `out_cos`), averaged across 3 captured layers.

### Forge run

One A100, `--shared-nfs`, `--disk-mount tq-models:/mnt/models`. Reuse
the HYP-053 staging path. Expected ~1 min.

## Pass / fail

**Primary — does QJL add measurable value on top of outlier-awareness?**
- `out_cos(HYP-054) − out_cos(HYP-053) ≥ 0.003` at seq ≥ 16 k.

**Secondary — does the combination approach fp16 parity?**
- `out_cos(HYP-054) ≥ 0.997` at seq ≥ 16 k.

**Short-ctx sanity:**
- `out_cos(HYP-054) − out_cos(HYP-053) ≥ −0.002` at seq ≤ 4 k.
  (Must not regress short ctx by more than 0.002 — QJL's variance
  should be small enough that it's a net-positive everywhere.)

## Decision tree

| outcome                                   | next action                                                 |
|-------------------------------------------|-------------------------------------------------------------|
| Primary + secondary both pass             | **Paper's full recipe reproduced.** File Gate 1 kernel work: mixed-precision layout with QJL sign bits on regulars. |
| Primary passes, secondary falls short     | QJL helps but not to parity. Ship HYP-054 as best-effort 4.5× option; parity requires backing off to 3.5-bit (HYP-055). |
| Primary fails (QJL still vestigial)       | QJL genuinely doesn't pay anywhere in our stack. Close QJL permanently. Ship HYP-053 (outlier MSE-only) as the tier-2 compression path. |
| Short-ctx regresses > 0.002               | QJL variance not yet fully tamed even at small residuals. Re-examine; possibly larger `m` (rectangular projection with `m > 96`) in a follow-up. |

## Relationship to GOAL / SPEC

- **GOAL SC#3** (target: 3.5-bit avg, 4.5× compression, LongBench
  parity): this hypothesis at 2.5-bit avg tests whether the full paper
  recipe is better than MSE-only. If it hits parity at 2.5-bit, we
  overshot the goal; if not, HYP-055 (3.5-bit) becomes the parity path.
- **GOAL SC#4** (stretch: 2.5-bit avg, ≤1 pt LongBench drop): this
  hypothesis is the direct test. Paper reports 0.62 pt drop at 2.5-bit.
- **SPEC §2** (Unbiased Attention Estimation): if HYP-054 passes, this
  spec behavior is satisfied in production for the first time.

## Status: REJECTED — QJL retired from our stack (4 consecutive rejections)

Forge job `d54ea205` SUCCEEDED, 20 s wall, real Qwen3-8B K/V, 3 layers.

### Result table (full five methods)

| seq   | fp16  | MSE 4-b | MSE 2-b | HYP-053 (outlier MSE) | **HYP-054 (outlier + QJL regs)** |
|------:|:-----:|--------:|--------:|----------------------:|---------------------------------:|
|  1 k  | 1.000 | 0.9989  | 0.9827  | 0.9922                | **0.9399**                       |
|  4 k  | 1.000 | 0.9990  | 0.9770  | 0.9925                | **0.9444**                       |
| 16 k  | 1.000 | 0.9993  | 0.9672  | 0.9914                | **0.9513**                       |
| 32 k  | 1.000 | 0.9995  | 0.9642  | 0.9908                | **0.9516**                       |

**All three criteria fail:**

```
primary   (Δ vs HYP-053 @ 16 k):  -0.040   (threshold >= +0.003)
primary   (Δ vs HYP-053 @ 32 k):  -0.039
secondary (absolute @ 16 k):       0.951   (threshold >= 0.997)
short-ctx (Δ vs HYP-053 @ 4 k):   -0.048   (threshold >= -0.002)

VERDICT: FAIL-PRIMARY
```

### Why the "small residual" intuition was wrong

`TurboQuantProd(bit_width=2)` doesn't *add* QJL on top of a good MSE
quantizer — it *splits* the 2-bit regular budget into
`1-bit MSE + 1-bit QJL`. So at the regular channels:

- HYP-053: 2-bit MSE → residual ≈ 0.27 · ‖x_reg‖ (tolerable)
- HYP-054: **1-bit** MSE → residual ≈ 0.77 · ‖x_reg‖ (huge)

Outlier-awareness shrank the *magnitude* of the regular channels'
signal vs the full vector, but the *relative* residual-to-signal ratio
after 1-bit MSE is unchanged. QJL inherited the same
variance-dominates-signal problem as HYP-052 (uniform 2-bit).

The binding quantity is the MSE bit-width. QJL can't rescue a
coarser codebook regardless of which channels it's applied to.

### QJL empirical record (retired)

| hyp | config                                              | verdict                             |
|-----|-----------------------------------------------------|-------------------------------------|
| 049 | 4-bit MSE + QJL-in-pad (extra 0.5 bit/dim)          | rejected                            |
| 050 | `TurboQuantProd(4)` = 3-b MSE + 1-b QJL uniform     | rejected                            |
| 052 | `TurboQuantProd(2)` = 1-b MSE + 1-b QJL uniform     | rejected (4× variance inversion)    |
| diag | 8 QJL seed sweep                                    | **no bug** — unbiasedness confirmed |
| 054 | outlier-aware + `TurboQuantProd(2)` on regulars     | rejected (Δ = −0.04)                |

**QJL is retired from this project.** On real Qwen3-8B K/V cache, at
every MSE-bit-split tested, QJL's JL variance
`(π/2m)·‖q‖²·‖residual‖²` exceeds the inner-product bias it's correcting.
The paper's Theorem 2 (unbiasedness) holds theoretically (our seed
diagnostic verified `|bias|<0.01` across 8 S draws), but unbiasedness
is purchased at a variance cost that softmax amplifies nonlinearly
and attention quality doesn't tolerate.

### Where this leaves the sub-4-bit effort

| option | desc                                                    | trade              |
|--------|---------------------------------------------------------|--------------------|
| **A**  | Ship HYP-053 (outlier MSE-only, 2.5-bit avg) as tier-2 | 4.5× compression, 0.009 out_cos gap at 32 k |
| **B**  | HYP-055: 32 × 4-bit + 96 × 3-bit = **3.25-bit avg**    | 4.3× compression, predicted near-parity (paper's 3.5-bit point) |
| **C**  | Close sub-4-bit effort                                  | stay at 3.2× today |

Recommended: **A + B in parallel**. A gives us the 4.5× product
option; B likely reaches fp16 parity at 4.3×. They're
non-overlapping points on the compression-quality curve and users can
pick per workload.
