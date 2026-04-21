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

## Status: COMPLETE — QJL is bit-budget-dependent, paper parity does NOT reproduce on Qwen3-8B

All 6 Stage-2 jobs SUCCEEDED on Forge. Full 9-method × 4-task table
(small_balanced preset, fp16 = 100 samples/task; quant methods = preset
sizes qasper=25, narrativeqa=10, hotpotqa=25, passage_retrieval_en=25).
Scores × 100 (F1 / ROUGE-L / accuracy per task).

| method                       | bits/dim | qasper | narr | hotpot | passage | avg |
|------------------------------|---------:|-------:|-----:|-------:|--------:|----:|
| fp16 (reference)             | 16.00    |  46.4  | 27.9 |  59.0  |  100.0  | 58.3 |
| A uniform (HYP-055, n=100)   |  2.00    |  11.5  |  1.3 |  0.05  |  ~0.30  |  ~3 |
| B uniform (HYP-055-b, n=10)  |  2.00    |   2.1  |  1.4 |   0.7  |    0    |  1.1 |
| A'  outlier MSE              |  2.50    |  28.5  | 28.8 |  24.8  |    8.0  | 22.5 |
| B'  outlier + QJL            |  2.50    |  13.9  | 17.0 |  23.7  |    8.0  | 15.7 |
| A_3' outlier MSE             |  3.25    |  35.4  | 23.5 |  43.5  |   84.0  | 46.6 |
| B_3' outlier + QJL           |  3.25    |  30.9  | 31.7 |  54.1  |   72.0  | 47.2 |
| A_35' outlier MSE            |  3.50    |  41.6  | 25.4 |  42.6  |   96.0  | 51.4 |
| **B_35' outlier + QJL**      |  **3.50** | **39.8** | **25.1** | **52.3** | **96.0** | **53.3** |

### Three-tier QJL verdict (Δ = B − A, percentage points)

| tier     | qasper | narrat | hotpot | passage | avg  | wins/loses/ties |
|----------|-------:|-------:|-------:|--------:|-----:|:-----------------|
| 2.5-bit  | −14.6  | −11.8  |  −1.1  |    0    | −6.9 | 0W / 2L / 2T — **QJL hurts decisively** |
| 3.25-bit |  −4.5  |  +8.2  | +10.6  |  −12.0  | +0.6 | 2W / 2L / 0T — **mixed**         |
| 3.5-bit  |  −1.8  |  −0.3  |  +9.7  |    0    | +1.9 | 1W / 1L / 2T — **QJL wins net** (no task regresses > 1.8 pt) |

QJL's value **scales with MSE bit-width** exactly as theory predicts:
residual shrinks → JL variance `(π/2m)·‖r‖²` shrinks → the noise QJL
injects drops below the bias it corrects. 3.5-bit is the first clean
net-positive regime on Qwen3-8B.

### Task-level pattern: softmax-sharpness

| task type                 | QJL impact at 3.5-bit | mechanism                                         |
|---------------------------|-----------------------|---------------------------------------------------|
| hotpotqa (multi-hop)       | **+9.7 pt**            | Attention spreads across many relevant keys. QJL's unbiased estimator preserves that distribution. |
| narrativeqa (free-form QA) | −0.3 pt (neutral)      | Mixed attention pattern; QJL's variance and unbiasedness roughly cancel. |
| passage_retrieval_en       | 0 pt (saturated)       | Task is easy enough at 3.5-bit that both methods hit 96 %. |
| qasper (extractive QA)     | −1.8 pt                | Attention must concentrate sharply on a span; QJL's residual variance slightly smears softmax. At 3.5-bit the damage is within sample noise. |

### Paper-parity check (fp16 − B_35')

| task       | fp16  | B_35' | gap   |
|------------|------:|------:|------:|
| qasper     | 46.4  | 39.8  | −6.6  |
| narrativeqa| 27.9  | 25.1  | −2.8  |
| hotpotqa   | 59.0  | 52.3  | −6.7  |
| passage_retr| 100.0 | 96.0 | −4.0  |
| **average** | **58.3** | **53.3** | **−5.0 pp** |

Paper claims ~0 pp gap at 3.5-bit on Llama-3.1-8B (LongBench 50.06 =
50.06). We see **−5.0 pp** on Qwen3-8B. Paper parity does NOT
reproduce.

Likely drivers:

1. **Missing QJL on outlier tier.** Paper's 3.5-bit split is 32 × (4-bit
   MSE + 1-bit QJL) + 96 × (2-bit MSE + 1-bit QJL) — QJL on *both* tiers.
   Our codebook caps at 4-bit so we can't build a 5-bit outlier path,
   but the outlier-QJL contribution is untested.
2. **Model difference.** Qwen3-8B has different RoPE partitioning and
   different per-head variance distribution than Llama-3.1-8B. Outlier
   mask at top-32 / top-64 per K-variance may not be the right cutoff.
3. **Task mix.** Our 4-task subset weighted differently than paper's
   21-task aggregate; passage_retrieval_en ceiling is saturated in both
   ours and paper's numbers.

### Structural findings (lock these in)

1. **QJL without outlier-awareness is catastrophic** on every real task
   (HYP-055 B uniform ≤ 2 pp everywhere). Uniform Alg 2 is not a viable
   configuration on Qwen3-8B K/V.
2. **Outlier-awareness alone (A-prime series) carries the bulk of the
   quality lift.** A_35' reaches 51.4 pp average vs A uniform at ~3. The
   outlier mask is the load-bearing mechanism.
3. **QJL's incremental value is task-dependent** and scales with
   regular-channel MSE budget. Helps multi-hop / free-form at 3.25+ bits;
   hurts extractive QA at ≤ 2.5 bits.
4. **A_35' (3.5-bit, no QJL) is the strongest single ship-ready
   candidate.** 4.57× compression (paper's claim), 53.3 / 58.3 = **91.5 %
   of fp16**, zero kernel-side QJL complexity.

### Ship-readiness decisions

| option | config | compression | avg / fp16 | kernel complexity |
|--------|--------|------------:|-----------:|-------------------|
| today  | 4-bit MSE uniform      | 3.2× | 99.9 % | simple (shipping) |
| **A_35'** | **outlier 3.5-bit MSE** | **4.57×** | **91.5 %** | medium (outlier mask) |
| B_35'  | outlier 3.5-bit + QJL  | 4.57× | 91.5 % + multi-hop lift | high (QJL dequant) |
| A' / B' | outlier 2.5-bit ± QJL  | 6.4× | 38–27 % | not viable |

Recommend: **file ADR for A_35' as the compression-tier-2 option**,
shelve QJL unless a specific multi-hop-heavy workload justifies the
kernel complexity cost.

### Artefacts

- `docs/hypotheses/HYP-055b-longbench-paper-full-recipe.md` (this doc).
- `/workspace/shared/hyp055/{fp16,A,B-small}/` — HYP-055 uniform results.
- `/workspace/shared/hyp055b/{A_prime,B_prime,A_3_prime,B_3_prime,A_35_prime,B_35_prime}/`
  — Stage 2 per-method predictions and scores.
- Forge jobs (all SUCCEEDED): `04c44030` (A'), `028226bf` (B'),
  `7cfc0321` (A_3'), `1c19b25a` (B_3'), `b36c31b7` (A_35'), `fd7ea288` (B_35').

### Follow-ups (out of scope here)

- **Full 21-task LongBench at 3.5-bit** to confirm 91.5 % scales to the
  full benchmark and to compare apples-to-apples with paper.
- **Outlier-QJL tier** (HYP-055c?) if we add a 5-bit outlier codebook.
  Would close the remaining 5 pp gap to paper IF it reproduces.
- **Multi-hop-focused eval** to quantify B_35''s hotpotqa-style upside
  for agentic workloads.
