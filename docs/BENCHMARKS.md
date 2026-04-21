# Benchmarks

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
