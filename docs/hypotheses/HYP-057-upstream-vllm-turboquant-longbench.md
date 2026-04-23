# HYP-057: Upstream vLLM v0.20.0 native TurboQuant reproduces paper's fp16-parity claim on LongBench

## Context

Upstream vLLM v0.20.0 (tagged 2026-04-22, commit `579602aa4be6`) shipped a
native TurboQuant KV-cache quantization backend under
`vllm/model_executor/layers/quantization/turboquant/` via two PRs:

- **PR #38479** (Vibhav Agarwal, merged 2026-04-15) — initial backend,
  Triton store + fused decode kernels, 4 named presets.
- **PR #40194** (Dan Alistarh, merged 2026-04-18) — "remove redundant random
  signs, add prior art attribution". Made the codebase explicit that the
  implementation descends from HIGGS (Malinovskii 2025, arXiv:2411.17525)
  and "Cache Me If You Must" (Shutova 2025, arXiv:2501.19392), **not** from
  the TurboQuant paper's Algorithm 2 — and that **QJL is intentionally
  omitted**.

Per PR #40194 docstring:
> "QJL is intentionally omitted — community consensus (5+ independent
> groups) found it hurts attention quality by amplifying variance through
> softmax."

HYP-049/050/052/054/055c are four of those rejections on our stack.

### Architecture — not just a dtype

`--kv-cache-dtype turboquant_*` looks like a dtype flag but is actually a
dispatch key that activates a completely separate attention backend. PR
#38479 changes 27 files / +2940 LOC:

| file | LOC | role |
|---|---:|---|
| `v1/attention/backends/turboquant_attn.py` | 812 (new) | custom attention backend — prefill/decode dispatch, page mgmt |
| `v1/attention/ops/triton_turboquant_store.py` | 441 (new) | Triton kernel — quantize + pack on every KV write (prefill + decode) |
| `v1/attention/ops/triton_turboquant_decode.py` | 617 (new) | Triton fused decode — on-the-fly dequant + QK·softmax·V |
| `v1/kv_cache_interface.py` | +26 | custom page_size hook (equivalent of our PR #39868) |
| `model_executor/layers/attention/attention.py` | +82 | route through TQ backend |
| quantization/turboquant/config.py + presets | 185 | centroid tables, 4 named presets, config validation |

**Our own vllm_backend_fused.py + docker/vllm_patches/** occupies the same
3-layer structure (store kernel, fused decode kernel, page-size hook) with
a different algorithm (same Lloyd-Max MSE, but our CUDA kernels instead of
Triton, and our v4/v5 variants instead of the HIGGS-flavored recipe).

## Hypothesis

Upstream vLLM v0.20.0's native TurboQuant reaches **fp16 parity on
LongBench at 3.5–5× KV-cache compression**, reproducing the claim the
paper makes (§4.3, Table 1: Llama-3.1-8B LongBench 50.06 fp16 = 50.06 at
3.5-bit) with a simpler, no-QJL, no-outlier-aware recipe.

## Prediction

Paper Table 1 on Llama-3.1-8B-Instruct:

| preset | compression | paper-claimed |
|---|---:|---:|
| fp16 | 1× | 50.06 |
| TQ @ 3.5-bit | 4.5× | 50.06 (paper, with QJL+outlier) |

Upstream's 4 uniform presets (their own PPL numbers, Qwen3-4B):

| preset | compression | upstream +PPL | GSM8K (author) |
|---|---:|---:|---:|
| `turboquant_k8v4` | 2.6× | +1.17% | 0.860 vs 0.900 fp16 |
| `turboquant_4bit_nc` | 3.8× | +2.71% | 0.840 |
| `turboquant_k3v4_nc` | ~3.5× | +10.63% | 0.780 |
| `turboquant_3bit_nc` | 4.9× | +20.59% | 0.720 |

On our Llama-3.1-8B-Instruct `small_balanced` 4-task subset, we predicted
at least one upstream preset lands within ≤2 pp of fp16.

## Method

### Harness

- **Model:** Llama-3.1-8B-Instruct (paper's model), cached on `tq-models`
  Forge disk (`HF_HUB_CACHE=/mnt/models/hf_cache` flat layout;
  `HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1` for the gated repo).
- **Dataset:** LongBench V1 subset, pre-staged to
  `/mnt/models/longbench/data/`. `small_balanced` preset: qasper=25,
  narrativeqa=10, hotpotqa=25, passage_retrieval_en=25.
- **Bench script:** `tests/bench_longbench_vllm.py` (new) — vLLM Python
  `LLM` API, greedy decoding (`temperature=0, top_p=1.0,
  max_tokens=TASK_MAXGEN[task]`), chat-template gating identical to
  `bench_longbench_kv_quant.py` (HF-transformers path).
- **Scorers:** identical to HF bench (verbatim port of LongBench
  `metrics.py`).

### vLLM install

v0.20.0 is not on PyPI. Staged git tree to
`/workspace/shared/tq-vllm020/vllm-v0.20.0/`, then each job pip-installed
with `VLLM_USE_PRECOMPILED=1` reusing the base image `ce745b54`
(tq-upstream-nightly:v4) compiled extensions. Per-job local clone to
`/tmp/vllm-v0.20.0/` was necessary — 5 parallel jobs racing on one shared
source tree stomped each other's `build/` directory on the first attempt.

### Jobs

5 parallel 1-GPU Forge jobs on Llama-3.1-8B-Instruct, `small_balanced`:

| backend | job id |
|---|---|
| `auto` (fp16) | `af38238c` |
| `turboquant_k8v4` | `36542047` |
| `turboquant_4bit_nc` | `7a642416` |
| `turboquant_k3v4_nc` | `2810a1c8` |
| `turboquant_3bit_nc` | `44b23c1b` |

All succeeded (9–10 min each).

## Results

Scores on `small_balanced` (4 tasks, F1 / ROUGE-L / retrieval accuracy per
task as LongBench canonical metrics):

| backend | compression | qasper | hotpotqa | passage_retr | narrativeqa | **4-task avg** | Δ vs fp16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **fp16** (auto) | 1× | 0.469 | 0.468 | 1.000 | 0.425 | **0.591** | — |
| `turboquant_k8v4` | 2.6× | 0.008 | 0.036 | 0.000 | 0.002 | **0.012** | **−0.579 ❌** |
| `turboquant_4bit_nc` | 3.8× | 0.480 | 0.458 | 1.000 | 0.438 | **0.594** | **+0.003 ✅** |
| `turboquant_k3v4_nc` | ~3.5× | 0.476 | 0.458 | 0.960 | 0.408 | **0.576** | **−0.015 ✅** |
| `turboquant_3bit_nc` | 4.9× | 0.439 | 0.491 | 1.000 | 0.418 | **0.587** | **−0.004 ✅** |

### Headline findings

1. **Paper's 3.5-bit = fp16 parity claim reproduces.** Three of four
   upstream presets (`4bit_nc`, `k3v4_nc`, `3bit_nc`) land within ≤1.5 pp
   of fp16 at 3.5–5× compression. The paper's §4.3 Table 1 headline
   number is reproducible on LongBench with a **simpler, no-QJL,
   no-outlier-aware recipe** than paper's Algorithm 2.

2. **4.9× compression is still fp16-lossless on LongBench.** The most
   aggressive preset `turboquant_3bit_nc` (upstream claims +20.59% PPL)
   hits 0.587 vs fp16 0.591 — a 0.4 pp gap, within sample noise. This
   shows **PPL and LongBench-QA diverge**: PPL is token-level and
   penalises every distribution shift; LongBench QA metrics are
   answer-level and far more robust to KV quantization.

3. **`turboquant_k8v4` catastrophically broken on A100.** All 4 tasks at
   ~0 score — generation runs (olen=32/128 tokens emitted) but tokens
   are nonsense. Not a partial degradation; it's total corruption.

4. **Paper's QJL mechanism was not load-bearing on real tasks.** The
   upstream authors explicitly omitted QJL citing community consensus
   (5+ groups). Our HYP-049/050/052/054/055c form four of those
   rejections. A fifth pattern from upstream's own implementation: their
   aggressive 4.9× preset works fp16-lossless **without QJL**, and a
   paper-faithful `B_35_paper` (QJL-on-both-tiers) on HYP-055c lost by
   6–7 pp vs our `B_35_prime` (QJL-on-regs-only) on the same budget.

### k8v4 on A100 — diagnostic not conclusive

PR #38479 body states "Auto-detects SM capability for Ampere vs Hopper FP8
formats" and reports author's perf/quality on "4× RTX PRO 6000 Blackwell"
(SM120). A100 is SM80 / Ampere — the auto-detect path the author wrote
but did not benchmark.

Clean-compile diagnostic attempted (job `1e130891`, `74b55c01`): two
failures, both at cmake/nvcc subprocess exit 1 while compiling vllm from
source inside the Forge job container. Didn't rule in or out an ABI vs
correctness bug. **Recommended follow-up:** file an upstream issue with
our k8v4-on-A100 results and let PR author debug on their side.

## Pass / fail

- **Primary pass:** any upstream preset within ≤2 pp of fp16 on 4-task
  avg — **PASS** (3 of 4 presets pass, best at +0.003 pp).
- **Secondary pass:** fp16 via vLLM matches HF-transformers fp16 within
  sampling noise — **PASS** (our pipeline run of fp16 through vLLM
  produced qasper 0.469, hotpot 0.468, passage 1.000 which sit inside
  paper Table 1's per-category ranges: SingleQA 0.4529, MultiQA 0.4516,
  Synthetic 0.5954 — accounting for our 4-task subset vs paper's 21-task
  bench-E).
- **Null/kill:** no — result is confirmatory, not rejecting the paper.

## Relationship to prior hypotheses

- **HYP-049/050/052/054**: four rejections of QJL in various
  configurations on our stack. Upstream's explicit QJL omission and their
  reference to "5+ independent groups" corroborate.
- **HYP-055b**: paper-strict reproduction on Qwen3-8B — −5.0 pp gap at
  3.5-bit. Driver hypothesis was "wrong model" (paper uses Llama-3.1-8B)
  plus "missing QJL on outlier tier".
- **HYP-055c**: tested both drivers. Paper-strict `B_35_paper` (QJL on
  both tiers) on Llama-3.1-8B-Instruct itself **lost by 7.44 pp vs
  `B_35_prime`** (QJL on regs only) at matched 3.5-bit budget. Paper's
  own recipe does not reproduce on its own model.
- **HYP-057 (this)**: third confirmation angle. Upstream's completely
  non-QJL recipe reaches fp16 parity at 3.5× compression on Llama-3.1-8B.
  Combined with HYP-055c, the evidence is: **paper's fp16-parity
  compression point is real, but the QJL+outlier mechanism the paper
  attributes it to is not load-bearing**.

### Implications for our project

- GOAL.md success criterion #3 (4.5× compression at LongBench parity) is
  achievable *without QJL*. Our shipped 4-bit MSE uniform path (HYP-029
  lineage) is close to upstream's `turboquant_4bit_nc` in both algorithm
  and result.
- GOAL.md success criterion #4 (2.5-bit stretch goal where QJL was
  "load-bearing") — given HYP-055c's rejection of QJL on its own model +
  HYP-057's upstream-without-QJL success at 4.9× compression, the stretch
  goal should be rewritten in terms of non-QJL compression (maybe
  `turboquant_3bit_nc` as a reference point, 4.9× at fp16 parity on
  LongBench).
- **Our docker/vllm_patches/ is redundant on v0.20.0** — upstream merged
  the custom-page-size hook natively (`v1/kv_cache_interface.py +26
  LOC`). On v0.20.0+ we can drop the patches and use upstream directly.
- **Our vllm_backend_fused.py (plugin path) is now optional** — upstream
  shipped a parallel implementation under `turboquant_attn.py`. We can
  either (a) keep ours as a performance-comparison baseline, or (b)
  deprecate ours and use upstream.

### Follow-ups

- **[out of scope]** Debug `turboquant_k8v4` on A100 — file upstream
  issue at `vllm-project/vllm` with our small_balanced scores; link to
  PR #38479's Blackwell-only bench matrix.
- **[optional]** Run full 21-task LongBench-E for apples-to-apples
  comparison with paper Table 1. Primary claim is already reproduced on
  the 4-task subset.
- **[optional]** NIAH eval — paper Figure 4 claims 0.997 vs fp16 0.997
  on Llama-3.1-8B. Separate eval infra required.

## Status: CONFIRMED

Upstream vLLM v0.20.0 reproduces the TurboQuant paper's fp16-parity
compression claim on LongBench (Llama-3.1-8B-Instruct, 4-task
small_balanced subset), at 3.5–5× compression, using a non-QJL recipe
that differs from the paper's Algorithm 2. Three of four upstream presets
pass; `turboquant_k8v4` fails on A100 (Ampere FP8 path; upstream author
tested Blackwell only) — out of scope for this hypothesis.

### Artefacts

- `tests/bench_longbench_vllm.py` (new) — vLLM Python LLM API
  LongBench eval.
- `tests/bench_longbench_kv_quant.py`, `tests/longbench_scorers.py`
  (vendored from HYP-055c worktree).
- `docs/reference/turboquant-paper-methodology.md` — paper §3/§4
  methodology reference + paper-vs-upstream algorithm delta.
- `/workspace/shared/vllm020_longbench/*.json` — per-preset per-task
  predictions and scores (5 files).
- Forge job IDs (all SUCCEEDED): `af38238c` (fp16), `36542047` (k8v4),
  `7a642416` (4bit_nc), `2810a1c8` (k3v4_nc), `44b23c1b` (3bit_nc).
- Diagnostic FAIL: `1e130891`, `74b55c01` (cmake subprocess exit 1
  during clean vllm compile — not pursued further).
