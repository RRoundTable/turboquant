# HYP-053: Outlier-aware 2.5-bit MSE-only (no QJL)

## Context

HYP-049 / 050 / 052 rejected pure uniform TurboQuant Algorithm 2
(MSE + QJL) at every tested budget on real Qwen3-8B K/V. Alg 1
(pure MSE) strictly beat Alg 2 at b ∈ {2, 4}. The paper's headline
4.5× quality numbers (LongBench 50.06 at 3.5-bit avg) are always
**outlier-aware + Alg 2**, never pure uniform.

This hypothesis isolates the **outlier-aware** contribution in
isolation — without QJL — to determine whether outlier-awareness
alone carries the quality win, with Alg 2 vestigial.

SPEC.md §3 has specified outlier-aware behavior since the project
started but has never been implemented. HYP-053 is the first
experiment that actually exercises SPEC §3.

## Hypothesis

At a fixed 2.5 bits/dim average budget on real Qwen3-8B K/V with
outlier-aware mixed precision (32 outlier dims @ 4-bit MSE + 96
regular dims @ 2-bit MSE, no QJL), attention-output cosine matches
or exceeds today's 4-bit uniform MSE baseline at every seq length
— or at minimum stays above 0.99, closing the gap between uniform
2-bit (0.97) and 4-bit (0.999).

## Prediction

| method                         | bits/dim avg | QJL? | `out_cos` @ 16 k (pred) |
|--------------------------------|-------------:|------|------------------------:|
| fp16 (reference)               | 16           | —    | 1.000                   |
| MSE 4-bit uniform (shipped)    | 4.0          | no   | ≥ 0.998                 |
| **Outlier-aware 2.5-bit MSE**  | **2.5**      | no   | **≥ 0.990**             |
| MSE 2-bit uniform (HYP-052)    | 2.0          | no   | 0.967 (baseline)        |

If the outlier-aware row beats pure uniform 2-bit by ≥ 0.02 on
`out_cos`, the outlier-detection mechanism is load-bearing. If it
also reaches ≥ 0.995, the next step is to ship SPEC §3 directly
and skip QJL entirely.

## Method

### Outlier selection (offline calibration)

Per layer × per head:
1. Use the first 4 k tokens of the cached
   `/workspace/shared/hyp050_kv_real.pt` as a calibration set.
2. Compute per-dim **variance across tokens**: `var[d] = K[:4096, head, d].var(dim=0)`.
3. Select top-32 dims (descending variance) as the outlier mask.
4. The 96 remaining dims are regulars.

Masks are cached and applied at encode/decode time. Expected
calibration runtime: < 1 s per layer (pure tensor ops).

### Reconstruction per head

```
outlier_dims (32):  TurboQuantMSE(bit_width=4).quantize_dequantize(x_outliers)
regular_dims (96):  TurboQuantMSE(bit_width=2).quantize_dequantize(x_regulars)
x̂ = combine(outlier_recon, regular_recon, outlier_mask)
```

Average budget: (32·4 + 96·2) / 128 = 320 / 128 = **2.5 bits/dim**.

### Gate 0 — Python reference on cached real K/V

1. Extend `tests/test_qjl_long_context_bias.py` (worktree
   `worktree-agent-a8d9f08d`) to include:
   - MSE 4-bit uniform (shipped baseline, sanity)
   - MSE 2-bit uniform (pure Alg 1 @ 2 bits, HYP-052 baseline)
   - **Outlier-aware 2.5-bit MSE** (new, this hypothesis)
   - fp16 reference
2. Run on the same cached K/V across seq ∈ {1 k, 4 k, 16 k, 32 k}.
3. Same metrics: `score_cos`, `abs_err`, `bias`, `softmax_cos`,
   `out_cos`. Plus report **effective bytes/token/head** to verify
   the 2.5-bit claim numerically.

### Forge run

One A100 job (~30 s expected). Reuse `tq-models` disk and cached
K/V exactly as HYP-052. No model redownload.

## Pass / fail

**Primary (does outlier-awareness alone beat uniform 2-bit?):**
- `out_cos(outlier-aware) − out_cos(MSE 2-bit uniform) ≥ 0.02` at
  seq ≥ 4 k.

**Secondary (does it approach quality parity?):**
- `out_cos(outlier-aware) ≥ 0.99` at seq ≥ 16 k.

**Kill criterion:** if `out_cos(outlier-aware) < out_cos(MSE 2-bit
uniform)`, the outlier mask is miscalibrated; stop, inspect, don't
declare verdict.

## Decision tree

| outcome                                   | next action                                                          |
|-------------------------------------------|----------------------------------------------------------------------|
| Primary + secondary both pass             | **Ship SPEC §3 without QJL.** File Gate 1: kernel layout for mixed bit-width per head, outlier-mask in `__constant__` memory. |
| Primary passes, secondary fails (0.97–0.99) | Outlier-awareness helps but not enough. File HYP-054: outlier-aware + Alg 2 on regulars, paper's exact recipe. |
| Primary fails                             | Outlier-awareness doesn't translate on Qwen3-8B either. Close the sub-4-bit effort; ship 4-bit pure MSE as final product. |

## Status: pending

Dispatched to Forge.
