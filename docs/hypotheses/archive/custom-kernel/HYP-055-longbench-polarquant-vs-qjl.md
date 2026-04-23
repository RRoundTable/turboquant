# HYP-055: LongBench re-verification — PolarQuant vs PolarQuant+QJL

## Context

Four consecutive QJL rejections (HYP-049/050/052/054) all relied on a
synthetic proxy metric: score-cosine between `<random_Q, K̂>` and
`<random_Q, K>`, using pre-RoPE K captured from a single repeating
32 k prompt. Three methodology issues bias the test against QJL:

1. **K is pre-RoPE** (hook on `k_proj` fires before rotary embedding).
   Production quantizes post-RoPE K. Per-dim statistics differ; our
   outlier calibration is on the wrong distribution.
2. **Queries are random Gaussians**, not real Q from the same model.
   QJL's value is preserving *semantic* inner products where Q
   concentrates on a few keys — random Q has no concentration, so
   preserving its scores unbiasedly vs biasedly is measurably identical
   to noise.
3. **Prompt is a repeating snippet.** K/V tokens are strongly periodic.
   Attention distributions differ from natural text.

End-to-end task scores on real benchmarks (LongBench / L-Eval) are the
correct measurement. This hypothesis re-tests QJL in the regime it was
designed for.

## Nomenclature

- **PolarQuant** at `b` bits = L2 norm extraction + FWHT rotation +
  Lloyd-Max scalar quantization of the unit-sphere direction.
  Implemented in `turboquant/quantizer.py` as `TurboQuantMSE(b)`.
- **PolarQuant + QJL** at `b` bits = PolarQuant at `b−1` bits + 1-bit
  QJL on the residual. Implemented as `TurboQuantProd(b)`.

## Hypothesis

At a 2-bit KV cache budget on Qwen3-8B, **PolarQuant(1) + QJL(1)
(method B)** matches or beats **PolarQuant(2) MSE-only (method A)** on
LongBench aggregate task scores, because:

1. Real-attention inner products concentrate on a few semantically
   related keys; QJL's unbiased estimator preserves the top-k structure
   that determines softmax output quality.
2. Task-level metrics (F1, ROUGE-L, accuracy) aggregate over many
   softmax decisions, averaging over QJL's per-pair variance so the
   unbiasedness property starts to dominate the variance cost.
3. These two effects were invisible in our prior random-Q / cosine-fidelity
   measurements.

## Prediction

On the LongBench 5-task subset at Qwen3-8B, expect:

| task                        | metric   | fp16 (ref) | A (PolarQuant 2b) | B (Polar 1b + QJL 1b) |
|-----------------------------|----------|-----------:|------------------:|----------------------:|
| narrativeqa                 | F1       | ~18–22     | A_0               | **≥ A_0 + 1.5**       |
| qasper                      | F1       | ~30–35     | A_1               | **≥ A_1 + 1.5**       |
| hotpotqa                    | F1       | ~42–48     | A_2               | **≥ A_2 + 1.5**       |
| gov_report                  | ROUGE-L  | ~30–33     | A_3               | **≥ A_3 + 1.5**       |
| passage_retrieval_en        | accuracy | ~70–80     | A_4               | **≥ A_4 + 2**         |

These thresholds are deliberately modest — paper's 2.5-bit result
showed a 0.62 pt LongBench drop vs full precision (with outlier-aware).
At pure 2-bit without outliers, we expect a larger drop, but **B should
still beat A by a measurable margin if QJL's unbiasedness pays off**.

## Pass criteria

- **Primary:** B's score ≥ A's score + 1.5 pt on **at least 3 of 5 tasks**.
- **Secondary:** B's score within 2 pt of fp16 on **at least 3 of 5 tasks**.
- **Kill criterion:** if B is strictly worse than A on every task, QJL
  is confirmed vestigial at 2-bit on task-level metrics too, and the
  earlier synthetic rejections were directionally correct despite the
  methodology flaws.

## Method

### Pipeline

1. `tests/bench_longbench_kv_quant.py`:
   - Load Qwen3-8B fp16 via HuggingFace.
   - Monkey-patch the `Qwen3Attention.forward` (or equivalent) to apply
     a `kv_quant_fn(k, v)` hook **after** `apply_rotary_emb` but
     **before** the attention scaled-dot-product.
   - Three variants of `kv_quant_fn`:
     - `fp16`: identity.
     - `A = PolarQuant(2)`: `TurboQuantMSE(2).quantize → .dequantize` per head.
     - `B = PolarQuant(1) + QJL(1)`: `TurboQuantProd(2).quantize → .dequantize` per head.
   - Run 5 LongBench tasks (`narrativeqa`, `qasper`, `hotpotqa`,
     `gov_report`, `passage_retrieval_en`). Limit to ~100–200 samples
     per task to keep runtime tractable; this is enough for the 1.5 pt
     thresholds to be distinguishable from noise.
   - Use LongBench's official scorer from its HuggingFace dataset card.

2. **Smoke test first**: 10 samples of `qasper` only, fp16 path only.
   Validates that the attention hook works and LongBench scoring runs
   before burning Forge time on the full sweep.

3. **Full sweep**: 3 methods × 5 tasks × ~150 samples. Parallelize
   across 3 Forge jobs (one per method) using `tq-models` disk for the
   model cache and shared NFS for benchmark data.

### Forge runs

Data staging (one-time, ~15 min):
- LongBench dataset from HuggingFace → `tq-models` disk under
  `/mnt/models/longbench`.

Per-method job (estimated ~60 min on 1× A100 eager at ~10–16 k context):
- `--gpu 1 --shared-nfs --disk-mount tq-models:/mnt/models`
- Writes `results/hyp055/{fp16,A,B}/scores.json` to shared NFS.

Aggregation:
- Final step reads the three JSONs, prints a table, applies pass/fail
  thresholds, emits verdict.

## Decision tree

| outcome                                     | action                                                    |
|---------------------------------------------|-----------------------------------------------------------|
| B beats A on ≥ 3 tasks by ≥ 1.5 pt, and within 2 pt of fp16 | **QJL reinstated.** File Gate 1 kernel work for mixed PolarQuant + QJL layout. Close prior synthetic rejections as "methodology-artifact, superseded by HYP-055." |
| B beats A on 1–2 tasks only                 | **Mixed.** QJL helps for some long-ctx tasks, not all. Drill into which task types benefit; possibly ship QJL as opt-in for retrieval-heavy workloads. |
| B ≤ A on every task                         | **QJL permanently retired.** Task-level confirmation closes the question. Ship PolarQuant-only path (HYP-053 at 2.5-bit with outliers, or 4-bit today). |
| fp16 scores don't match published Qwen3-8B LongBench numbers | **Harness broken.** Stop, fix the monkey-patch / scoring, don't trust any result. |

## Engineering notes

- Qwen3-8B attention structure differs from Llama; monkey-patch must
  target the correct method. Inspect `transformers.models.qwen3.modeling_qwen3.Qwen3Attention`
  (or `Qwen2Attention` if that's the class Qwen3 uses) to find the
  exact point after RoPE and before scaled-dot-product.
- Use `@torch.no_grad()` throughout the hook.
- LongBench has a standard prompting format; don't reinvent it. Use the
  `longbench` dataset entries' built-in prompt template.
- Per-sample max output length: LongBench specifies per-task limits —
  honor them.

## Status: pending

Smoke test dispatched to Forge first; full sweep on smoke pass.
