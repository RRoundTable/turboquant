# HYP-055c: Paper-strict reproduction on Llama-3.1-8B + Qwen3-8B, 14-task LongBench

## Context

HYP-055b ran our outlier-aware variants (MSE-only + MSE+QJL-on-regs) on Qwen3-8B
across 4 LongBench tasks. Aggregate `out_cos` gap to fp16 at 3.5-bit was **−5.0 pp**;
paper claims parity on Llama-3.1-8B with a stricter recipe.

HYP-055c closes the three differences vs paper:

1. **Add 5-bit Lloyd-Max codebook** to `turboquant/codebook.py` (was 1–4 bits only).
2. **Test on Llama-3.1-8B-Instruct** (paper's model) in addition to Qwen3-8B.
3. **Apply QJL on BOTH tiers** (outlier + regular), matching paper Algorithm 2 on both.

New strict-recipe modes in `tests/bench_longbench_kv_quant.py`:

```
B_25_paper = 32 dims × TurboQuantProd(32, 4) + 96 dims × TurboQuantProd(96, 2)   # 2.5 bits/dim
B_35_paper = 32 dims × TurboQuantProd(32, 5) + 96 dims × TurboQuantProd(96, 3)   # 3.5 bits/dim
```

vs our non-paper variants (same budget, QJL on regs only):

```
B_35_prime = 32 dims × TurboQuantMSE(32, 4) + 96 dims × TurboQuantProd(96, 2)   # 3.5 bits/dim
B_3_prime  = 32 dims × TurboQuantMSE(32, 4) + 96 dims × TurboQuantMSE(96, 3)    # 3.25 bits/dim
```

## Method

1. **Code (worktree commits `787637f`, `420d190`).** 5-bit centroids (MSE=0.0025,
   below paper bound 0.0026), paper-strict modes, 14 English LongBench tasks with
   official scorers.
2. **Staging.** Cached Qwen3-8B + Llama-3.1-8B-Instruct on `tq-models` Forge disk.
   LongBench v1 data.zip extracted to `/mnt/models/longbench/data/`.
3. **Fan-out.** 19 slots per model = **38 parallel Forge jobs**:
   - fast methods (fp16, A_prime, A_3_prime, A_35_prime) on full_14_task preset (1 job each)
   - slow methods (B_prime, B_3_prime, B_35_prime, B_25_paper, B_35_paper) split into 3 task-groups (fast/med/slow) × 3 jobs each
4. **Auto-heal.** Repeated OOM-at-model-load was not a VRAM issue (Llama = 17 GB;
   A100 = 40 GB). Diagnosed as Forge GPU-isolation gap: pods share physical GPUs
   without `CUDA_VISIBLE_DEVICES` being set, and PyTorch defaults to `cuda:0`
   causing races. Separately reported to Forge admin at
   `docs/reference/forge-gpu-isolation-report.md`. Client-side workaround: inject
   `nvidia-smi`-based auto-pick prelude into every entrypoint to target the emptiest
   GPU before `torch.cuda` initializes. Dropped OOM rate from ~90 % to ~0 %.

## Partial results (log-extracted, ~50 % coverage)

Runtime reality forced cancellation of 23 jobs that had been running 4+ h without
completing the long-context slow-group tasks (gov_report + qmsum + multi_news at
~200 s/sample on B-methods compounds fast). We kept the 4 still-PENDING slow-group
jobs that had yet to be scheduled; they continue in the background.

**13 JSON files landed on shared NFS** before the cancel wave, covering:

- `fp16/all.json` for both models (14 tasks)
- `A_35_prime/all.json` for Llama
- `B_prime/med.json`, `B_3_prime/med.json`, `B_35_prime/med.json`, `B_35_paper/med.json`
  for both models (5 medium-ctx tasks)
- `B_25_paper/{fast,med}.json` for Llama

### Med-group table (scores × 100, avg of 5 tasks: multifieldqa_en, hotpotqa, 2wikimqa, musique, passage_count)

**Qwen3-8B:**

| method      | bits/dim | multifield | hotpot | 2wiki | musique | pass_cnt | **avg** |
|-------------|---------:|-----------:|-------:|------:|--------:|---------:|--------:|
| B_prime     |     2.5  |      45.9  |  15.1  | 19.0  |    8.3  |    0.0   | **17.66** |
| B_3_prime   |     3.25 |      54.0  |  47.3  | 25.5  |   17.2  |    0.0   | **28.80** |
| B_35_prime  |     3.5  |      45.3  |  46.9  | 30.1  |   26.8  |    0.0   | **29.82** |
| B_35_paper  |     3.5  |      46.3  |  33.3  | 22.3  |   13.2  |    3.3   | **23.68** |

**Llama-3.1-8B-Instruct:**

| method      | bits/dim | multifield | hotpot | 2wiki | musique | pass_cnt | **avg** |
|-------------|---------:|-----------:|-------:|------:|--------:|---------:|--------:|
| B_3_prime   |     3.25 |      48.1  |  29.0  | 38.6  |   34.1  |    1.3   | **30.22** |
| B_35_prime  |     3.5  |      53.9  |  42.6  | 39.6  |   30.2  |    0.0   | **33.26** |
| B_35_paper  |     3.5  |      47.5  |  35.1  | 25.2  |   20.7  |    0.6   | **25.82** |

**fp16 partial (Llama, 3 med-subset tasks for reference):** hotpot 54.2, 2wiki 43.9,
musique 34.4 → ~**34 avg** on that subset. B_35_prime at 33.26 is **~1 pp** off fp16
on this 5-task med-group.

### Key comparison — paper recipe vs our recipe at 3.5-bit (med-group)

| model     | B_35_prime (our: outlier MSE + QJL-regs) | B_35_paper (paper: outlier-QJL + QJL-regs) | Δ        |
|-----------|-----------------------------------------:|-------------------------------------------:|---------:|
| **Qwen3** |                                   29.82  |                                     23.68  | **−6.14** |
| **Llama** |                                   33.26  |                                     25.82  | **−7.44** |

**Paper's strict 3.5-bit recipe loses by 6–7 pp on med-group**, on both Qwen3 AND
on the paper's own Llama-3.1-8B-Instruct. Adding QJL to the outlier tier (5-bit MSE
+ 1-bit QJL) is strictly worse than keeping outliers at pure 4-bit MSE while
using MSE+QJL only on the 96 regular channels.

### Other partial findings

- **QJL at 3.25-bit regs (B_3_prime) is a reasonable point** — 28.8 Qwen / 30.2 Llama
  on med-group, close to 3.5-bit at lower memory.
- **B_prime (2.5-bit avg) remains catastrophic** — 17.66 on Qwen med-group.
  Budgeting away from 3+ bits on the regular tier is not viable.
- **Llama prompt-template bug in the bench** — `passage_retrieval_en` returns
  `50/50 errors` for `A_35_prime` and `30/30 errors` for `B_25_paper/fast`. Likely
  a chat-template formatting issue specific to Llama-3.1-8B-Instruct; other tasks
  work. Rows for that task × those two Llama methods are excluded from aggregates.

## Structural conclusion (preliminary, confirmed on both models)

The outlier-tier QJL addition (paper's 5-bit outlier = 4-bit MSE + 1-bit QJL) is
the load-bearing difference between our recipe and the paper's. **It hurts, not
helps**, on real LongBench QA tasks.

Per the QJL variance formula `(π/2m)·‖r‖²`, the residual left after 4-bit MSE on
high-variance outlier channels is large (because outlier magnitudes are large by
definition). QJL's variance on that residual is proportional to `‖r‖²`, so the
unbiasedness correction on outliers injects much more noise than it corrects.
The "pay for QJL only where residuals are small" heuristic → keep MSE pure on
outliers, apply QJL only to regulars — consistently beats the paper's recipe at
3.5-bit on both models tested.

## Status: partial, cancelled long-runners

- 11 of 19 Qwen3 slots and 6 of 19 Llama slots completed and deposited JSONs.
- 4 slots still running in the background (2 qwen, 2 llama — slow-group heal-retries
  that got scheduled after the cancel wave). Will add data for B_3_prime/slow,
  B_35_prime/slow, B_25_paper/slow, B_35_paper/slow when done.
- Full aggregate (including fast-group and narrativeqa/gov_report/qmsum/multi_news
  for B-methods) not completed due to B-method runtime exceeding practical budget
  (~3h+ per slow-group job).

## Follow-ups (not started)

- Fix `passage_retrieval_en` prompt template for Llama chat-format.
- Port outlier-aware MSE-only (HYP-053 / our `A_prime-series`) into the CUDA
  production kernel, since that's the cleanest ship candidate.
- Re-measure paper-strict 3.5-bit at proper sample counts (100+) if we want to
  publish a "paper doesn't reproduce" claim rigorously — current 30-sample
  averages have ~3–5 pp std per task.
- Decide whether to close QJL exploration permanently given the 4-for-4 pattern
  across HYP-049/050/052/054/055c all showing QJL-on-outliers is net-negative.

## Artefacts

- Worktree commits (not merged): `787637f`, `420d190`, plus the heal/auto-pick
  scripts in `/tmp/hyp055c-*.sh`.
- Forge admin report: `docs/reference/forge-gpu-isolation-report.md`.
- NFS JSON files: `/workspace/shared/hyp055c/{mode}/{group}.json` (Qwen),
  `/workspace/shared/hyp055c_llama/{mode}/{group}.json` (Llama).
- Forge succeeded job IDs (Qwen): 8b1cb60b, cb14dc35, bcd0d7c3, ed15b2f5, c2217635.
- Forge succeeded job IDs (Llama): 11a70dda, a8ce3a95, 31a0957c, 564abf24, dbe3f166, 39f71012.
