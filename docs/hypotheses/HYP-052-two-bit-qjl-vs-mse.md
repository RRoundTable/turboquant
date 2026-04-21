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

## Status: pending

Dispatched to Forge. Reuses cached `/workspace/shared/hyp050_kv_real.pt`
and `tq-models` disk.
