# TurboQuant paper — experimental methodology reference

Source: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, 2025 (ICLR 2026).

This is a condensed reference extracted directly from the paper PDF
(§3 algorithms, §4 experiments, Table 1, Figure 4). Purpose: a single
place reimplementers can check when asking "what exactly did the paper
do at the 3.5-bit operating point?" Page numbers refer to the arXiv PDF.

---

## 1. Core algorithms (§3, pp. 10–12)

### Algorithm 1 — TurboQuant_mse (MSE-optimal, §3.1 p.10)

```
input: dimension d, bit-width b
global:
  Π ∈ R^{d×d}   — random rotation matrix
  c_1,…,c_{2^b} ∈ [−1,1]   — Lloyd-Max centroids minimising Eq. 4
Quant_mse(x):
  y = Π · x
  idx_j = argmin_k |y_j − c_k|      for j ∈ [d]     # b-bit indices
  return idx
DeQuant_mse(idx):
  ỹ_j = c_{idx_j}
  return Π^T · ỹ
```

Distortion bound (Thm 1): `D_mse ≤ √(3π)/2 · 4^{−b}`.
For `b = 1,2,3,4`: `D_mse ≈ 0.36, 0.117, 0.03, 0.009`.

### Algorithm 2 — TurboQuant_prod (inner-product-unbiased, §3.2 p.12)

```
input: dimension d, bit-width b
global:
  TurboQuant_mse at bit-width (b − 1)           # Algorithm 1 instance
  S ∈ R^{d×d}, S_{i,j} ~ N(0,1) i.i.d.          # dense Gaussian
Quant_prod(x):
  idx = Quant_mse(x)
  r = x − DeQuant_mse(idx)                      # MSE residual
  qjl = sign(S · r)                             # QJL on residual, 1 bit/dim
  return (idx, qjl, ‖r‖_2)
DeQuant_prod(idx, qjl, γ):
  x̃_mse = DeQuant_mse(idx)
  x̃_qjl = √(π/2) / d · γ · S^T · qjl
  return x̃_mse + x̃_qjl
```

Key facts:
- QJL adds **exactly 1 extra bit per dim** on top of the MSE base
  (so `Alg-2(b)` = `Alg-1(b−1)` + 1-bit QJL = b bits per dim total).
- `S` is **dense Gaussian**, not structured/Hadamard. Per-token cost
  is O(d²) FMAs for both quantize (`S·r`) and dequant (`S^T·qjl`).
- Dequant requires storing `‖r‖_2` (fp16 per token per head, one scalar).
- Claim (Thm 2): the resulting inner-product estimator is **unbiased**
  with variance `O(1/d)·‖y‖²·‖x‖²`, beating Alg-1's `O(2/π)` bias.

---

## 2. Outlier-aware mixed precision (§4.3 p.18, the load-bearing trick)

Verbatim from §4.3:

> "We evaluate our method using 2.5-bit and 3.5-bit quantization during
> text generation. These non-integer bit precisions result from our
> strategy of splitting channels into outlier and non-outlier sets, and
> applying **two independent instances of TurboQuant** to each,
> allocating higher bit precision to outliers. … For example, in our
> **2.5-bit setup, 32 outlier channels are quantized at 3 bits, while
> the remaining 96 channels use 2 bits**, leading to an effective bit
> precision of (32 × 3 + 96 × 2)/128 = 2.5. For 3.5-bit quantization, a
> different ratio of outliers and regular channels leads to a higher
> effective bit precision."

Interpretation:
- Head dim `d = 128` (paper uses Llama-3.1-8B and Ministral-7B, both
  with head_dim 128).
- "TurboQuant" in §4.3 = `Alg 2` (inner-product-unbiased), applied
  **to each tier independently**. Each tier's Alg 2 carries its own
  MSE codebook + its own `S` matrix + its own `‖r‖_2` per token.
- 2.5-bit split is explicit: **32 × (2b MSE + 1b QJL) + 96 × (1b MSE + 1b QJL)**.
- 3.5-bit split is **not given explicitly** — paper says "a different
  ratio … leads to a higher effective bit precision". Compatible
  splits: `32·5 + 96·3 = 448/128 = 3.5` or `64·4 + 64·3 = 448/128 = 3.5`
  or `64·5 + 64·2 = 448/128 = 3.5`. The paper does not disambiguate.
- Outlier **selection criterion is not stated in §4.3**. Paper cites
  [63] (KVQuant) and [51] (RotateKV) as the "consistent prior work"
  for outlier extraction — these select outlier channels by
  per-channel variance measured on a calibration sample.

Prior work the outlier treatment is "consistent with":
- [63] = Hooper et al., "KVQuant: Towards 10 Million Context Length
  LLM Inference with KV Cache Quantization" (NeurIPS 2024).
- [51] = Su et al., "RotateKV: Accurate and robust 2-bit KV cache
  quantization for LLMs via outlier-aware adaptive rotations" (2025).

---

## 3. End-to-end LongBench evaluation (§4.3 p.18)

### Dataset

- **LongBench-E** (Bai et al., 2023, ref [10]) — the uniform-length
  subset of LongBench-V1. Paper's rationale: "designed with a more
  uniform length distribution … a fair assessment of each model's
  performance across varying context sizes". Six task categories:
  **SingleQA, MultiQA, Summarization, Few-shot, Synthetic, Code**.
  Paper reports per-category averages + overall average.

### Models

- **Llama-3.1-8B-Instruct** (primary, Table 1 top).
- **Ministral-7B-Instruct** (secondary, Table 1 bottom, 2.5-bit only).

### Generation / KV-quant application detail

Verbatim:
> "Unlike existing approaches such as KIVI and PolarQuant, which leave
> generated tokens unquantized, **our method applies quantization even
> during the streaming generation process**."

→ The paper quantizes KV entries **both during prefill and decode**.
This is a stronger test than "quantize prefill KV, leave decode KV
unquantized" — many baselines cheat here.

### Baselines compared

KIVI (3 bits & 5 bits), PolarQuant (3.9 bits), and TurboQuant at
2.5 bit & 3.5 bit.

### Paper Table 1 — headline numbers (KV Size column is bits/dim avg)

**Llama-3.1-8B-Instruct, LongBench-E avg (scores × 100):**

| Method | KV bits | SingleQA | MultiQA | Summ | Few-shot | Synth | Code | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full cache (fp16) | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | **50.06** |
| KIVI | 3 | 43.38 | 37.99 | 27.16 | 68.38 | 59.50 | 44.68 | 48.50 |
| KIVI | 5 | 45.04 | 45.70 | 26.47 | 68.57 | 59.55 | 46.41 | 50.16 |
| PolarQuant | 3.9 | 45.18 | 44.48 | 26.23 | 68.25 | 60.07 | 45.24 | 49.78 |
| **TurboQuant (ours)** | **2.5** | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | **49.44** |
| **TurboQuant (ours)** | **3.5** | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | **50.06** |

**Ministral-7B-Instruct:**

| Method | KV bits | Avg |
|---|---:|---:|
| Full cache | 16 | 49.89 |
| TurboQuant | 2.5 | 49.62 |

### Headline claim (§4.3, last line)

> "Despite using fewer bits than competing techniques, TurboQuant
> maintains performance comparable to unquantized models. Remarkably,
> we achieve this **while compressing quantized vectors by at least a
> factor of 4.5×**."

→ 4.5× = `16 / 3.5` for the 3.5-bit operating point; 6.4× for 2.5-bit.

---

## 4. Needle-In-A-Haystack (NIAH) — §4.2, Figure 4 p.19

- **Model:** Llama-3.1-8B-Instruct.
- **Context-length sweep:** 4k, 6k, 10k, 16k, 26k, 41k, 65k, 104k
  (8 token-limit buckets, log-spaced).
- **Needle depth percent:** 11 buckets from 0 % (start) to 100 % (end).
- Score = fraction of needles retrieved.

Figure 4 results (single scalar per method, averaged across the grid):

| Method | NIAH score |
|---|---:|
| Full Precision | 0.997 |
| **TurboQuant** | **0.997** (same as fp16) |
| PolarQuant | 0.995 |
| KIVI | 0.981 |
| PyramidKV | 0.895 |
| SnapKV | 0.858 |

Paper claims TurboQuant at the 3.5-bit operating point (4.5× compression)
preserves NIAH **exactly** on Llama-3.1-8B-Instruct.

---

## 5. What the paper **does not** specify (open questions for a reimplementer)

These are material for reproduction but aren't given in §4.3:

1. **3.5-bit outlier/regular split.** Could be `32×5+96×3`, `64×4+64×3`,
   or `64×5+64×2`. Paper only says "a different ratio". (Our HYP-055c
   tested `32×5 + 96×3` on both Qwen3-8B and Llama-3.1-8B and found a
   6–7 pp gap vs the outlier-MSE-only variant — i.e. paper-strict
   does **not** reproduce on that split.)
2. **Outlier calibration corpus.** "Consistent with [63,51]" hints at
   variance-based selection on a calibration sample but the sample
   source (WikiText? The task's own context? A separate held-out set?)
   is unstated.
3. **Outlier mask dynamism.** Static per-layer (calibrated once) vs
   per-prompt (re-selected on each prompt's context). Prior work ([63]
   KVQuant) does static; paper inherits this by reference.
4. **QJL's dense `S` matrix per tier.** Is `S` regenerated per-layer,
   per-head, or global? Paper's Alg 2 text implies one `S` per tier
   instance, but §4.3 doesn't clarify if tiers share/differ across layers.
5. **Generation hyperparameters for LongBench.** `max_new_tokens` per
   task, temperature (greedy assumed), max context length, chat
   template — none are stated. The canonical LongBench protocol is
   documented in THUDM/LongBench and is the standard defaults the
   reimplementer should use.
6. **Entropy coding of MSE indices.** §3.1 p.11 mentions it could
   reduce 4-bit average to ~3.8 bits but paper **opts out**: "given
   the limited gain, we have chosen not to incorporate this technique".
   Reimplementers should not enable entropy coding by default.

---

## 6. Paper vs upstream vLLM's `turboquant` module — algorithm comparison

**This is the most consequential section for us.** The `turboquant/`
module on vLLM `main` (and v0.20.0) is NOT an implementation of this
paper's Algorithm 2. From the upstream docstring:

> "Hadamard rotation + per-coordinate Lloyd-Max scalar quantization for
> **keys**, **uniform quantization for values**. The technique
> implemented here consists of the scalar case of the **HIGGS**
> quantization method (Malinovskii et al., NAACL 2025; arXiv:2411.17525):
> rotation + optimized grid + optional re-normalization, applied to KV
> cache compression. A first application of this approach to KV-cache
> compression is in **'Cache Me If You Must: Adaptive Key-Value
> Quantization for Large Language Models'** (Shutova et al., ICML 2025;
> arXiv:2501.19392). Both these references **pre-date the TurboQuant
> paper**. … **QJL is intentionally omitted** — community consensus
> (5+ independent groups) found it hurts attention quality by
> amplifying variance through softmax."

Dimension-by-dimension comparison:

| aspect | TurboQuant paper (Zandieh 2026) | vLLM native `turboquant` |
|---|---|---|
| Keys | Alg 2 (MSE + QJL residual) | Lloyd-Max MSE + Hadamard rotation, **no QJL** |
| Values | Alg 2 (MSE + QJL residual) | **Uniform** quantization, not Lloyd-Max |
| QJL | Core (makes inner-product unbiased) | **Intentionally omitted** per community consensus |
| Mixed precision | Outlier + regular tier (32+96) | Uniform across all channels |
| `S` projection | Dense Gaussian R^{d×d}, shared per tier | N/A |
| Residual norm | Stored per token (fp16) | N/A |
| Named presets | 2.5-bit, 3.5-bit | `k8v4`, `4bit_nc`, `k3v4_nc`, `3bit_nc` |
| Compression at near-lossless | 4.5× @ 3.5-bit (paper claim) | 2.6× @ `k8v4` (+1.17% PPL), 3.8× @ `4bit_nc` (+2.71% PPL) |
| Closest to paper's 3.5-bit op-point | — | `turboquant_k3v4_nc` (~3.5× compression, +10.63% PPL per upstream) |

**Bottom line:** upstream's "TurboQuant" borrows the name from this
paper but implements a different algorithm derived from HIGGS + Cache
Me If You Must. The paper's 3.5-bit = fp16 parity claim is a claim
about paper's Algorithm 2 + outlier-aware mixed-precision, not about
upstream's uniform rotation + Lloyd-Max. Verifying one via the other
answers a different question: "does the HIGGS-style rotation-only
recipe also hit paper's parity target without QJL?"

## 7. Our own prior findings that corroborate upstream's omit-QJL choice

Five consecutive rejections in our hypothesis record all show QJL is
net-negative on our stack:

- HYP-049: uniform 2-bit QJL residual — rejected.
- HYP-050: 3-bit MSE + 1-bit QJL uniform — rejected (QJL vestigial at 4 bits).
- HYP-052: 2-bit Alg 2 uniform vs Alg 1 uniform — Alg 2 strictly worse.
- HYP-054: outlier-MSE + QJL-on-regs at 2.5-bit — rejected.
- HYP-055c: paper-strict 3.5-bit (QJL on both tiers) vs outlier-MSE +
  QJL-on-regs-only — paper-strict loses by **−6.14 pp (Qwen3-8B)** /
  **−7.44 pp (Llama-3.1-8B-Instruct)** on med-group 5-task subset.
  Paper's own-model reproduction fails.

Our results + upstream's "5+ groups found QJL hurts" is a seven-way
independent convergence that the paper's §3.2 inner-product-unbiased
claim does not translate into downstream LongBench accuracy on
post-softmax attention.
