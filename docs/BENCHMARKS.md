# Benchmarks

> **Historical baseline (frozen 2026-04-22).** The 4-backend serving
> tables in §"Setup" through §"How to read this" below were captured
> with our pre-pivot custom-CUDA-kernel plugin
> (`turboquant/vllm_backend_fused.py` + `csrc/**`) labelled *ours*,
> compared against FA, FI, and the first-merged upstream
> `turboquant_4bit_nc`. They remain true for that code (now archived),
> but the project pivoted on 2026-04-23 to **improving upstream
> Triton TurboQuant via a vLLM plugin** — no custom CUDA. New HYP
> results land under
> §"Upstream TurboQuant optimization track" at the bottom.

End-to-end serving comparison of four attention-cache backends on the same hardware,
workload, and harness. Numbers come directly from `vllm bench serve`'s JSON output;
the aggregation script and raw results are in `results/v5_serve*`.

## Setup

- **Model**: `Qwen/Qwen3-8B`, fp16.
- **Hardware**: 1× A100-SXM4-40GB per run.
- **vLLM**: v0.19.0 base image for `FA`, `FI`, and *ours*; upstream nightly
  (`0.19.2rc1.dev21+g67ed01c35`) for *upstream*.
- **Harness**: `vllm serve` (OpenAI-compatible HTTP endpoint) +
  `vllm bench serve --dataset-name random --random-input-len S --random-output-len 128
  --num-prompts N --max-concurrency C`, see `tests/bench_serve_entry.sh` and
  `tests/bench_serve_upstream_entry.sh`.
- **Eager**: `--enforce-eager` for `FA` / `FI` / *ours* (A100 SM80 cannot
  torch.compile fp8e4nv); upstream runs with its default compiled path.

## Backends

| label | activation | kernels | notes |
|---|---|---|---|
| **FA** | vLLM auto → `FLASH_ATTN` (v2) | CUDA, tensor cores | stock vLLM 0.19 baseline |
| **FI** | `--attention-backend FLASHINFER` | CUDA, tensor cores | FlashInfer 0.2-ish |
| **ours** | `--attention-backend CUSTOM --kv-cache-dtype fp8` | hand-written CUDA (fused dequant + attention) | this repo; requires vLLM patches from [PR #39868](https://github.com/vllm-project/vllm/pull/39868) |
| **upstream** | `--kv-cache-dtype turboquant_4bit_nc` | Triton | upstream [`vllm.model_executor.layers.quantization.turboquant`](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/), merged into vLLM main 2026-04-15 |

## Median TTFT (ms) — lower is better

| seq × conc |     FA |     FI | **ours** | **upstream** | up/FA | ours/FA |
|------------|-------:|-------:|---------:|-------------:|------:|--------:|
|    1024 ×  8 |   299 |   351 | **  240** |   464 | 1.55× | **0.80×** |
|    2048 × 32 |   852 |   809 | **  821** |   882 | 1.04× | 0.96× |
|    8192 ×  8 |  1788 |  1544 | ** 1315** |  1993 | 1.11× | **0.74×** |
|   16384 ×  8 |  4005 |  3731 | ** 1833** |  3173 | 0.79× | **0.46×** |
|   32768 ×  4 |  9187 |  8328 | ** 3356** |  7716 | 0.84× | **0.37×** |
|   32768 ×  8 | 31758 | 29101 | ** 3194** |  7500 | 0.24× | **0.10×** |

## Median TPOT (ms) — lower is better

| seq × conc |   FA |   FI | **ours** | **upstream** | up/FA | ours/FA |
|------------|-----:|-----:|---------:|-------------:|------:|--------:|
|    1024 ×  8 | 22.7 | 22.0 | **29.5** | **19.2** | **0.84×** | 1.30× |
|    2048 × 32 | 52.0 | 49.6 | **62.3** | **73.2** | 1.41× | 1.20× |
|    8192 ×  8 | 49.5 | 47.9 | **56.8** | **71.8** | 1.45× | 1.15× |
|   16384 ×  8 | 97.9 | 90.9 | **91.6** |**145.1** | 1.48× | 0.94× |
|   32768 ×  4 | 87.5 | 80.3 | **82.9** |**157.9** | 1.80× | 0.95× |
|   32768 ×  8 | 87.7 | 80.4 |**138.5** |**334.2** | 3.81× | 1.58× |

## Output throughput (tok/s) — higher is better

| seq × conc |    FA |    FI | **ours** | **upstream** | up/FA | ours/FA |
|------------|------:|------:|---------:|-------------:|------:|--------:|
|    1024 ×  8 |  235 |  244 | **  198** | **  253** | **1.07×** | 0.84× |
|    2048 × 32 |  339 |  348 | **  303** | **  265** | 0.78× | 0.90× |
|    8192 ×  8 |  112 |  116 | **  106** | **   82** | 0.74× | 0.94× |
|   16384 ×  8 | 60.0 | 64.2 | ** 68.9** | ** 44.4** | 0.74× | **1.15×** |
|   32768 ×  4 | 24.1 | 26.3 | ** 36.4** | ** 18.5** | 0.77× | **1.51×** |
|   32768 ×  8 | 24.1 | 26.2 | ** 45.4** | ** 19.9** | 0.83× | **1.89×** |

## With CUDA graphs enabled

The eager tables above match the Dockerfile default (`--enforce-eager`). We
re-ran FA, FI, and ours with `--compilation-config cudagraph_mode:FULL` to
measure what CUDA graphs buy on the same hardware.

- **ours**: `--compilation-config '{"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'`.
  `mode:0` keeps inductor off — A100 SM80 cannot torch.compile fp8e4nv — but
  graph capture is independent of compilation. Our decode / quantize-write
  ops are dispatcher-routed (`torch.ops.turboquant_v5.*`,
  `torch.ops.turboquant_write.*`) so they survive vLLM's KV-cache storage
  swap during `profile_cudagraph_memory`; see HYP-051.
- **FA / FI**: dropped `--enforce-eager`, narrowed capture list to the same
  `[1,2,4,8,16,32,64]` for a clean cross-backend comparison.
- All three use `--max-num-seqs 64 --gpu-memory-utilization 0.85
  PYTORCH_ALLOC_CONF=expandable_segments:True`. The narrower
  `max_num_seqs` is needed so the graph pool fits alongside fp8 KV cache at
  `0.85` util (the default `256` overflows). Never hit in our bench since
  max concurrency here is 32.
- Upstream is not re-listed — it already ran compiled + graphs in the eager
  tables (that's upstream's default path).

### Median TTFT (ms)

| seq × conc |    FA |    FI | **ours** | ours/FA |
|------------|------:|------:|---------:|--------:|
|    1024 ×  8 |   474 |   460 | **  427** | 0.90× |
|    2048 × 32 |   756 |   743 | **  713** | 0.94× |
|    8192 ×  8 |  1752 |  1712 | ** 1201** | 0.69× |
|   16384 ×  8 |  3978 |  3735 | ** 1861** | 0.47× |
|   32768 ×  4 |  9094 |  8278 | ** 3345** | 0.37× |
|   32768 ×  8 | 31209 | 28807 | ** 3199** | **0.10×** |

### Median TPOT (ms)

| seq × conc |   FA |   FI | **ours** | ours/FA |
|------------|-----:|-----:|---------:|--------:|
|    1024 ×  8 | 15.3 | 15.2 | **19.6** | 1.28× |
|    2048 × 32 | 52.1 | 49.4 | **59.6** | 1.14× |
|    8192 ×  8 | 47.1 | 46.0 | **55.7** | 1.18× |
|   16384 ×  8 | 96.6 | 90.2 | **91.8** | 0.95× |
|   32768 ×  4 | 86.5 | 79.3 | **81.6** | 0.94× |
|   32768 ×  8 | 86.0 | 79.4 |**138.2** | 1.61× |

### Output throughput (tok/s)

| seq × conc |    FA |    FI | **ours** | ours/FA |
|------------|------:|------:|---------:|--------:|
|    1024 ×  8 | 292.5 | 293.4 | **255.2** | 0.87× |
|    2048 × 32 | 343.0 | 352.2 | **313.8** | 0.91× |
|    8192 ×  8 | 115.7 | 117.9 | **108.1** | 0.93× |
|   16384 ×  8 |  60.4 |  64.9 | ** 68.4** | 1.13× |
|   32768 ×  4 |  24.4 |  26.6 | ** 36.9** | **1.51×** |
|   32768 ×  8 |  24.5 |  26.5 | ** 45.5** | **1.86×** |

### What graphs change

Graphs only pay off in the short-context decode regime where Python launch
overhead is a large fraction of each step. At long context, the kernel is
compute-bound and graphs do nothing visible.

| metric | backend | s1024×c8 (short) | s32768×c8 (long) |
|---|---|---:|---:|
| **TPOT, eager → graphs** | FA   | 22.7 → 15.3 ms (**1.48×**) | 87.7 → 86.0 ms (1.02×) |
|                          | FI   | 22.0 → 15.2 ms (**1.45×**) | 80.4 → 79.4 ms (1.01×) |
|                          | ours | 29.5 → 19.6 ms (**1.50×**) | 138.5 → 138.2 ms (1.00×) |
| **throughput, eager → graphs** | FA | 235 → 293 (1.24×) | 24.1 → 24.5 (1.02×) |
|                                 | FI | 244 → 293 (1.20×) | 26.2 → 26.5 (1.01×) |
|                                 | ours | 198 → 255 (1.29×) | 45.4 → 45.5 (1.00×) |

TTFT at short ctx *regresses* ~1.4-1.8× across all three. Prefill compute
doesn't benefit from graphs, and the narrower `max_num_seqs=64` changes
the scheduler's chunked-prefill behaviour. TTFT at long ctx is unchanged —
there prefill cost dominates the scheduling delta, and the
"ours avoids preemption" story from the eager tables is preserved (ours
still 0.10× FA at s32768×c8).

**Net: the regime verdicts from the eager tables hold.** Graphs give every
backend the same ~1.5× short-ctx decode bump; they don't shift who wins
where.

Raw data in `results/v5_serve_graphs/`; regenerate with
`uv run python results/v5_serve_graphs/aggregate.py`.

## KV cache memory (same `gpu_memory_utilization=0.85`, from engine logs)

| backend | KV tokens budget | vs FA |
|---|---:|---:|
| FA (fp16) | 126,416 | 1.00× |
| ours (TQ fp8 + PR #39868) | **404,544** | **3.20×** |
| upstream (turboquant_4bit_nc) | ~480,000 | ~3.8× |

## How to read this

Three regimes, three winners:

### ≤ 2k context, short prompts (chat-style)

**upstream wins.** Its Triton dequant is free on lightly-loaded decode — matches or
beats FA on TPOT (19.2 ms vs 22.7 ms at s1024×8) and hits 253 tok/s vs FA's 235. This
is the regime upstream was designed for.

### 4k–8k context, medium load

**Stock FA / FI win.** Both TurboQuant variants pay a decode-step tax that
outweighs the cache savings at this size. Ours and upstream sit at 0.85–0.94× FA
throughput here.

### ≥ 16k context, batch ≥ 4 (long-doc, agentic, RAG)

**Ours wins decisively.** At s32768 × c8 we deliver **1.89× FA's throughput**
(45.4 vs 24.1 tok/s) and **10× faster TTFT** (3.2 s vs 31.8 s). The gap vs
upstream is 2.3× throughput.

The decisive factor at this regime is not the decode kernel — it's the
scheduler. See [Why ours wins TTFT at long ctx](#why-ours-wins-ttft-at-long-ctx).

## Why ours wins TTFT at long ctx

At long context, TTFT is ~entirely prefill cost. Three compounding effects:

1. **~3.8× smaller KV write** during prefill. TQ writes 4-bit nibbles + per-tile
   norms (~68 B/head/token) vs FA's fp16 (256 B/head/token). Pure HBM bandwidth.

2. **~3.8× smaller KV read in chunked prefill.** vLLM chunks long prompts at
   `max_num_batched_tokens=8192`. Each chunk's attention has to read all prior
   cache for its request. At the last chunk of a 32k prompt, that's ~24k tokens
   of K+V per request — multiplied by batch size, this is the hot loop.

3. **Avoided preemption under memory pressure.** This is the nonlinear effect:

   | seq × conc | FA TTFT | ours TTFT | FA scale vs prior row |
   |---|---:|---:|---:|
   | s32768 × c4 | 9.2 s | 3.4 s | — |
   | s32768 × c8 | **31.8 s** | **3.2 s** | **3.5×** |

   Going from batch=4 to batch=8 (2×) should roughly double TTFT. But FA jumped
   3.5× because at batch=8 the fp16 cache (19 GB for 8× 32k contexts) doesn't
   fit the 17 GB budget. vLLM preempts 2–3 requests and re-prefills them — the
   median TTFT explodes.

   TQ at 3.2× compression keeps all 8 requests on-GPU. No preemption, TTFT
   stays roughly linear.

Upstream only gets to ~4× FA (not 10×) at s32768×c8 because its page-size
accounting isn't as deeply wired into vLLM's scheduler as our PR #39868 path,
so some preemption still happens.

## Why ours loses TPOT

Per-decode-step compute. The TQ kernel must dequantize before attention on
every decode step. On A100 SM80:

- FA / FI: tensor-core attention at 312 TFLOPS effective
- ours: scalar-FMA dequant + attention at ~20 TFLOPS

At short ctx the dequant work is tiny; at long ctx × large batch it scales
with `batch × ceil(seq / num_splits)`. Upstream's Triton kernel has the same
scaling but higher constants → it loses TPOT faster than ours at long ctx
(334 ms/step at s32768×c8 vs our 138.5).

This is a known architectural ceiling: A100 has no async `ldmatrix`
instruction, so the smem→mma pipeline stall can't hide the dequant. H100 has
it; we expect the TPOT gap to largely close on H100 (untested as of this
commit).

## Raw data

- `results/v5_serve/` — short/medium configs, ours vs FA/FI
- `results/v5_serve_vs_upstream/` — first 3 configs × 4 backends
- `results/v5_serve_long/` — 3 long-ctx configs × 4 backends

Each has an `aggregate.py` you can re-run locally to regenerate the tables.

## Reproducing

1. Build the image locally:

   ```bash
   git clone https://github.com/RRoundTable/turboquant.git
   cd turboquant
   docker build -t vllm-turboquant .
   ```

2. Start the server:

   ```bash
   docker run --gpus all -p 8000:8000 vllm-turboquant \
     --model Qwen/Qwen3-8B \
     --gpu-memory-utilization 0.85 --max-model-len 32896
   ```

   (Entrypoint defaults to `--attention-backend CUSTOM --kv-cache-dtype fp8
   --enforce-eager` — see the bottom of `Dockerfile`.)

3. Drive it with `vllm bench serve` from any machine with network access:

   ```bash
   vllm bench serve --backend vllm --model Qwen/Qwen3-8B \
     --host <host> --port 8000 --endpoint /v1/completions \
     --dataset-name random \
     --random-input-len 32768 --random-output-len 128 \
     --num-prompts 64 --max-concurrency 8 --ignore-eos
   ```

For the Forge-based full sweep, see `tests/bench_serve_entry.sh` and
`tests/bench_serve_upstream_entry.sh`.

---

# Upstream TurboQuant optimization track

Active development as of 2026-04-23. We ship optimizations to upstream
vLLM v0.20.0's Triton TurboQuant kernels via a vLLM plugin (no source
fork). Entries below land per confirmed HYP from
`docs/ROADMAP.md` Phase 3+. See `docs/GOAL.md` for the SHA-256 parity
gate and out-of-scope list.

## Setup (current track)

- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (paper's model).
- **Hardware**: 1× A100-SXM4-40GB on Forge, `--security-profile profiling-debug`.
- **vLLM**: v0.20.0 (commit `579602aa4be6`), staged at
  `/workspace/shared/tq-vllm020/vllm-v0.20.0/`.
- **Plugin**: `pip install -e /workspace/shared/turboquant-plugin/`
  before bench launch.
- **Accuracy harness**: `tests/bench_longbench_vllm.py --preset small_balanced`,
  greedy decoding, identical to HYP-057.
- **Perf harness**: `tests/bench_serve_upstream_entry.sh` adapted to
  v0.20.0; bench grid `{4bit_nc, k3v4_nc, 3bit_nc, fp16}` ×
  `seq ∈ {1024, 8192}` × `concurrency ∈ {1, 8}` = 16 cells.

## Baseline — HYP-057 (Llama-3.1-8B-Instruct, small_balanced)

LongBench `small_balanced` 4-task accuracy (qasper / hotpotqa /
passage_retr / narrativeqa, F1·ROUGE-L·acc):

| preset | compression | 4-task avg | Δ vs fp16 |
|---|---:|---:|---:|
| `auto` (fp16) | 1× | 0.591 | — |
| `turboquant_4bit_nc` | 3.82× | 0.594 | +0.003 ✅ |
| `turboquant_k3v4_nc` | 4.34× | 0.576 | −0.015 ✅ |
| `turboquant_3bit_nc` | 5.02× | 0.587 | −0.004 ✅ |
| `turboquant_k8v4` | 2.61× | 0.012 | −0.579 ❌ (A100 broken; out of scope) |

Forge job IDs (all SUCCEEDED): `af38238c` (fp16), `7a642416`
(`4bit_nc`), `2810a1c8` (`k3v4_nc`), `44b23c1b` (`3bit_nc`),
`36542047` (`k8v4`). Raw JSONs at
`/workspace/shared/vllm020_longbench/`.

## Per-HYP results

Each row records the confirmed HYP, the kernel(s) it patches, the
parity gate it cleared, and the median TPOT delta vs the HYP-058
baseline at the bench cell where the HYP wins biggest. *TBD* rows are
filled in as Phase 3+ HYPs land.

| HYP | patches | parity gate | best cell (preset × seq × conc) | TPOT before → after | Δ TPOT |
|---|---|---|---|---:|---:|
| HYP-058 (baseline lock) | none | byte-exact 340/340 | reference (table below) | — | — |
| HYP-062 (joint launch retune) **REJECTED** | `_tq_decode_stage1` | SHA-256 85/85 ✓ | `4bit_nc × 8192 × 1` | 17.50 → 17.46 (best of 27) | **+0.20 %** (noise; below 5 % gate) |
| HYP-063 (smem centroid pre-stage) | `_tq_decode_stage1` | SHA-256 | TBD | TBD | TBD |
| HYP-064 (midpoints pre-load) | `_tq_fused_store_mse` | SHA-256 | TBD | TBD | TBD |
| HYP-065 (adaptive `NUM_KV_SPLITS`) | `TurboQuantMetadataBuilder.build` | mean ±0.002 pp (opt-out) | `3bit_nc × 8192 × 8` (83.4 ms ref) | TBD | TBD |
| HYP-066 (`tl.dot` QK fp16 acc) | `_tq_decode_stage1` | mean ±0.002 pp (opt-out) | TBD | TBD | TBD |
| HYP-067 (`tl.dot` V acc + TMA) | `_tq_decode_stage1` | mean ±0.002 pp (opt-out) | TBD | TBD | TBD |

## HYP-058 — measured baseline (the reference for every Phase 3+ row above)

Llama-3.1-8B-Instruct, vLLM v0.20.0 (`579602aa4be6`), 1× A100-SXM4-40GB
under default Forge profile, eager mode, `gpu-memory-utilization=0.85`,
output_len=128. Forge job `fb2e708a` (succeeded 2026-04-23 13:27 UTC).
Raw JSONs at `results/hyp058/perf_*.json` and on Forge NFS at
`/workspace/shared/hyp058_phase1/perf_grid/`.

### Median TPOT (ms) — lower is better

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 12.9 | 15.8 | 13.7 | 48.2 |
| `turboquant_4bit_nc` | 14.2 | 20.0 | 17.5 | 68.8 |
| `turboquant_k3v4_nc` | 14.2 | 20.7 | 17.9 | 76.9 |
| `turboquant_3bit_nc` | 14.3 | 21.2 | 18.6 | 83.4 |

### Median TTFT (ms)

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 93.9 | 466.0 | 702.3 | 1546.4 |
| `turboquant_4bit_nc` | 95.7 | 464.4 | 745.7 | 1747.8 |
| `turboquant_k3v4_nc` | 95.2 | 370.6 | 725.0 | 1871.1 |
| `turboquant_3bit_nc` | 98.1 | 372.3 | 746.8 | 1822.4 |

### Output throughput (tok/s)

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 66.0 | 412.5 | 49.0 | 132.5 |
| `turboquant_4bit_nc` | 60.8 | 265.2 | 39.1 | 91.2 |
| `turboquant_k3v4_nc` | 60.8 | 330.5 | 38.6 | 86.7 |
| `turboquant_3bit_nc` | 60.3 | 323.0 | 37.5 | 81.8 |

### Accuracy parity

LongBench `small_balanced` (qasper × 25 + hotpotqa × 25 +
passage_retrieval_en × 25 + narrativeqa × 10 = 85 samples per preset).
SHA-256 of greedy-decode prediction strings: **340 / 340 byte-exact
match vs HYP-057 baseline** (max |Δscore| = 0.0000 across all 4
in-scope presets). `turboquant_k8v4` is out-of-scope per `docs/GOAL.md`
(Ampere FP8 path broken).

## Profiling artefacts

Each confirmed HYP archives paired nsys + ncu traces at
`/workspace/shared/hypNNN/`:

- `before/trace.nsys-rep`, `before/decode.ncu-rep` — baseline (plugin off).
- `after/trace.nsys-rep`, `after/decode.ncu-rep` — patched (plugin on).
- `delta-warpstall.md` — short writeup of the dominant warp-stall class
  shift, must match the HYP's predicted shift.

No HYP merges to main without both halves on disk.
