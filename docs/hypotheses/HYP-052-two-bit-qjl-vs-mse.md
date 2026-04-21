# HYP-052: 2-bit TurboQuant — QJL vs MSE-only at identical memory

## Context

HYP-049 and HYP-050 rejected QJL at the 4-bit total budget. The paper
(Zandieh et al. 2025, Figure 2) claims QJL's contribution is
**largest at bit-widths b ≤ 2**, where MSE introduces a strong
multiplicative bias (`E[⟨y, Q_mse^{-1}(Q_mse(x))⟩] = (2/π)·⟨y,x⟩` at
b=1). This is the regime we've never tested and where the paper's
Algorithm 2 (MSE + QJL decomposition) is designed to matter.

**Scope of this hypothesis:** pure TurboQuant, **no outlier-aware
wrapper**. Uniform 2-bit quantization across all dims. Outlier-aware
mixed precision stays in GOAL.md SC#3/SC#4 and SPEC.md §3 as
a future extension; it is not a TurboQuant-core contribution and is
out of scope for this gate. If pure TurboQuant Alg 2 can't hit the
quality bar on its own at 2 bits, the outlier wrapper is the next
lever — but we measure each mechanism in isolation first.

## Hypothesis

At a fixed 2-bits-per-dim budget on real Qwen3-8B K/V, **TurboQuant
Algorithm 2 (1-bit MSE + 1-bit QJL = `TurboQuantProd(bit_width=2)`)
outperforms TurboQuant Algorithm 1 (2-bit MSE only =
`TurboQuantMSE(bit_width=2)`) on attention-output cosine by a margin
≥ 0.02**, and its inner-product bias is ≥ 5× smaller.

Memory is identical in both methods (2 bits/dim → 32 B quant per
128-dim head), so the delta isolates QJL's contribution cleanly.

## Prediction

### Leading metric: attention-output cosine on real Qwen3-8B K/V

| method                                 | bits/dim | QJL? | `out_cos` @ 32 k (pred) |
|----------------------------------------|---------:|------|------------------------:|
| fp16 (reference)                       | 16       | —    | 1.000                   |
| 4-bit MSE only (today's prod)          | 4.0      | no   | ≥ 0.998                 |
| **2-bit MSE only** (TurboQuant Alg 1)  | 2.0      | no   | ~0.85–0.92 (biased)     |
| **2-bit MSE + QJL** (TurboQuant Alg 2) | 2.0      | yes  | ≥ 0.97 (unbiased)       |

Delta on `out_cos`: **`Alg 2 − Alg 1 ≥ 0.02`** is the headline win.

### Inner-product bias (the property QJL exists for)

MSE-only at 2 bits has `E[⟨Q_mse^{-1}Q_mse(x), y⟩] ≈ (2/π)·⟨x, y⟩` —
a **0.64× multiplicative bias**. MSE+QJL gives `E = ⟨x, y⟩` exactly.

Prediction: `|bias(Alg 1) @ 32 k|` is at least **5× larger** than
`|bias(Alg 2) @ 32 k|`. This tests Theorem 2 directly on our data.

### Memory check (equality must hold)

```
Alg 1 (2-bit MSE only):
  128 dims × 2 bits / 8 = 32 B quant
  + 2 × fp16 norms      =  4 B
  → 36 B raw, 48 B aligned → 5.33× vs fp16

Alg 2 (1-bit MSE + 1-bit QJL):
  128 dims × 1 bit / 8  = 16 B quant
  + 128 dims × 1 bit / 8= 16 B QJL signs
  + 2 × fp16 MSE norms  =  4 B
  + 2 × fp16 res norms  =  4 B
  → 40 B raw, 48 B aligned → 5.33× vs fp16
```

**Identical 5.33× compression in both variants** — any quality delta
attributable to the algorithm, not the budget.

## Method

### Gate 0 — Python reference on cached real K/V

1. Reuse `/workspace/shared/hyp050_kv_real.pt` (Qwen3-8B layers
   {8, 16, 24} × 32 k-token prompt, captured in HYP-050 run).
2. Extend `tests/test_qjl_long_context_bias.py` with **four methods**:
   - fp16 (reference)
   - `TurboQuantMSE(bit_width=4)` — shipped baseline
   - `TurboQuantMSE(bit_width=2)` — **Alg 1 at 2 bits**
   - `TurboQuantProd(bit_width=2)` — **Alg 2 at 2 bits** (1 MSE + 1 QJL)
3. Same metrics as HYP-049/050: `score_cos`, `abs_err`, `bias`,
   `softmax_cos`, `out_cos`. Averaged across the 3 captured layers.
4. Seq sweep: {1 k, 4 k, 16 k, 32 k}.
5. Also report per-method **effective memory (bytes/head/token)** as
   a sanity check that Alg 1 and Alg 2 at b=2 are equal.

### Forge run

Single A100 job, `--disk-mount tq-models:/mnt/models`, `--shared-nfs`.
Python-only, reuses model cache, expected ~2 min wall clock (no
HuggingFace download — K/V already staged).

### Pass conditions

**Primary (QJL contribution at 2 bits):**
- `out_cos(Alg 2) − out_cos(Alg 1) ≥ 0.02` at seq ≥ 4 k AND
- `|bias(Alg 1)|/|bias(Alg 2)| ≥ 5` at seq ≥ 4 k (bias ratio).

**Secondary (absolute quality):**
- `out_cos(Alg 2) ≥ 0.97` at seq ≥ 16 k. If this fails, 2-bit is just
  too aggressive on Qwen3-8B regardless of QJL, and the next step is
  outlier-aware wrapping (SPEC §3, deferred).

## Decision tree

| outcome                                           | next action                                                  |
|---------------------------------------------------|--------------------------------------------------------------|
| Primary + secondary both pass                     | **Reproduce TurboQuant Alg 2 in production.** File Gate 1: rectangular-QJL extension of `qjl.py`, new 48 B tile layout, Python end-to-end reference. |
| Primary passes, secondary fails (Alg 2 < 0.97)    | QJL works *as predicted*, but 2-bit regardless is too aggressive. Escalate to outlier-aware HYP-053 — start from SPEC §3 with Alg 2 on regular channels. |
| Primary fails (Alg 2 ≈ Alg 1)                     | QJL does not help even at 2 bits on real Qwen3-8B. TurboQuant Alg 2 is empirically vestigial on this model. Close; focus on pure MSE + compression tricks elsewhere. |

## Kill criteria

- If `TurboQuantMSE(2)` or `TurboQuantProd(2)` produces NaNs or
  numerical instability in PyTorch reference → fix the Python path
  first, re-run; don't declare a verdict on a broken implementation.
- If the memory-equality check fails (Alg 1 and Alg 2 at b=2 are not
  both ~5.33× compression) → the comparison isn't apples-to-apples;
  stop and recalibrate the layout before judging.

## Relationship to the paper

This is the **minimal reproduction** of TurboQuant's Theorem 2 /
Figure 2 on Qwen3-8B activations. The paper's headline 2.5-bit /
3.5-bit results use outlier-aware mixed precision on top of Alg 2 —
but Alg 2 alone, on uniform bit-widths, is the core contribution. If
Gate 0 passes, we've empirically validated TurboQuant's central
theoretical claim on real LLM K/V. If it fails, we have grounds to
re-examine the paper's applicability to modern fp16 activations
(post-RoPE, post-RMSNorm distributions that differ from the
rotationally-symmetric synthetic data the paper proves bounds on).

## Status: REJECTED — Alg 2 strictly worse than Alg 1 at 2-bit budget

Forge job `d4f6bf2a-1f8a-49b6-a02b-146ee865743a` SUCCEEDED (~30 s).

| method          | out_cos @ 1k | @ 4k   | @ 16k  | @ 32k  |
|-----------------|-------------:|-------:|-------:|-------:|
| fp16            | 1.000        | 1.000  | 1.000  | 1.000  |
| MSE_4bit (prod) | 0.9989       | 0.9990 | 0.9993 | 0.9995 |
| **MSE_2bit**    | **0.9827**   | **0.9770** | **0.9672** | **0.9642** |
| Prod_2bit       | 0.8554       | 0.8731 | 0.8703 | 0.8689 |

```
out_cos(Prod_2bit) − out_cos(MSE_2bit) @ 4k  = −0.1039   (expected ≥ +0.02)
out_cos(Prod_2bit) − out_cos(MSE_2bit) @ 16k = −0.0969
out_cos(Prod_2bit) − out_cos(MSE_2bit) @ 32k = −0.0953
|bias(MSE_2bit)| / |bias(Prod_2bit)| @ 16k   = 0.236     (expected ≥ 5)
```

QJL at 2 bits is **strictly harmful** on our data:

1. **Delta is the wrong sign.** Prod loses 9.7 percentage points on
   `out_cos` at long ctx, not gains 2.
2. **Bias ratio is inverted.** Prod is ~4× *more* biased than MSE_2bit
   (+0.085 vs −0.020). Theorem 2's "unbiased in expectation" property
   holds asymptotically; on a finite 3-layer × 32 k sample it does not.
3. **Absolute quality collapses.** Prod at 0.87 is well below the
   secondary threshold (0.97).

### Mechanism

At 2-bit total budget, Alg 2 decomposes as `TurboQuantMSE(1) + QJL(1)`:
- 1-bit MSE has huge reconstruction error (`abs_err = 0.58–0.63`, vs
  `0.27` for 2-bit MSE). The "residual" QJL is asked to correct is
  almost the full original vector.
- QJL's variance `(π/2m)·‖res‖²` scales with the residual magnitude.
  A huge residual makes the JL correction's variance dominate every
  attention score.
- The net effect: QJL adds more noise than it removes bias, and
  softmax amplifies the noise nonlinearly.

### Pattern across HYP-049 / 050 / 052

| HYP | total b | Alg 1 (MSE) out_cos @ 16k | Alg 2 (MSE + QJL) out_cos @ 16k | delta |
|-----|--------:|--------------------------:|--------------------------------:|------:|
| 050 |       4 | 0.9993                    | 0.9965 (3 MSE + 1 QJL)           | −0.003 |
| 052 |       2 | 0.9672                    | 0.8703 (1 MSE + 1 QJL)           | **−0.097** |

**Alg 2 is worse than Alg 1 at every uniform bit budget we've tested on
real Qwen3-8B.** The gap widens as budget shrinks. This empirically
rejects pure TurboQuant Alg 2 as a practical quantizer for Qwen3-8B
K/V cache.

### Bias claim was seed lottery — rejection stands on variance

After-the-fact diagnostic (Forge `0df2a604`, 8 different QJL seeds ×
4 different query seeds × 3 layers × 3 seq lengths) confirms that
**there is no bug** in `qjl.py` or `TurboQuantProd`:

| metric                            | 8-seed result       | meaning                           |
|-----------------------------------|---------------------|-----------------------------------|
| `|bias_mean|` across seeds @ 32 k | **0.007** (< 0.01)  | unbiasedness holds in expectation |
| per-seed bias range @ 32 k        | −0.058 to +0.043    | the single seed (10042) we used   |
|                                   |                     | was in the tail                   |
| best seed `out_cos` @ 32 k        | **0.889**           | still far below MSE_2bit's 0.984  |
| MSE_2bit `out_cos` @ 32 k         | 0.984               | deterministic, no seed dependence |

So the bias-inversion claim in the earlier draft of this rejection
was a finite-sample artifact. The **real reason** the hypothesis
fails is that JL **variance** — not bias — is the binding cost at
a 2-bit budget on real Qwen3-8B activations. No seed choice flips
the verdict.

### Why the paper's numbers don't reproduce here

The paper's LongBench/NIAH quality tables (§4.3) are **always**
outlier-aware + Alg 2 on regular channels + higher-bit MSE on outliers.
The paper never benchmarks pure uniform Alg 2 on LLM activations.
Figure 2's "QJL wins at low b" plot is on **synthetic isotropic
Gaussians**, not real post-RoPE/post-RMSNorm activations. Our three
hypotheses establish that the synthetic → real gap is large enough
to invert the MSE vs Prod ranking at the relevant budgets.

### Implication for GOAL.md SC#3/SC#4

Reproducing the paper's 4.5× compression at quality parity requires
**outlier-aware mixed precision as the primary mechanism**, with
Alg 2 only on the low-bit regular channels. Without outlier-aware,
uniform sub-4-bit TurboQuant is not viable on Qwen3-8B K/V. This
promotes SPEC.md §3 from "aspirational" to "critical path."
