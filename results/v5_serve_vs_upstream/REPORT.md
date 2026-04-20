# Upstream vllm turboquant vs our TurboQuant-CUDA vs FA / FI

- Upstream turboquant: vllm nightly (0.19.2rc1.dev21+g893611813), `--kv-cache-dtype turboquant_4bit_nc`, Triton kernels.
- Ours: vllm v0.19.0 + docker/vllm_patches (PR #39868), `--attention-backend CUSTOM --kv-cache-dtype fp8`, hand-written CUDA kernel.
- FA / FI: vllm v0.19.0 baselines from results/v5_serve/.
- Qwen/Qwen3-8B fp16, A100-40GB. `vllm bench serve --dataset-name random`.

## Median TTFT (ms) — lower is better

| seq × conc |  FA | FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|------------|----:|---:|---------:|-------------:|--------:|------:|--------:|
|  1024 ×  8 | 299 | 351 | ** 240** | ** 464** |  0.80x | 1.55x |  1.93x |
|  2048 × 32 | 852 | 809 | ** 821** | ** 882** |  0.96x | 1.04x |  1.07x |
|  8192 ×  8 | 1788 | 1544 | **1315** | **1993** |  0.74x | 1.11x |  1.52x |

## Median TPOT (ms) — lower is better

| seq × conc |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|------------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|
|  1024 ×  8 | 22.7 | 22.0 | ** 29.5** | ** 19.2** |  1.30x | 0.84x |  0.65x |
|  2048 × 32 | 52.0 | 49.6 | ** 62.3** | ** 73.2** |  1.20x | 1.41x |  1.17x |
|  8192 ×  8 | 49.5 | 47.9 | ** 56.8** | ** 71.8** |  1.15x | 1.45x |  1.26x |

## Output throughput (tok/s) — higher is better

| seq × conc |     FA |     FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|------------|-------:|-------:|---------:|-------------:|--------:|------:|--------:|
|  1024 ×  8 |  235.3 |  243.5 | ** 197.7** | ** 253.0** |  0.84x | 1.07x |  1.28x |
|  2048 × 32 |  338.7 |  347.6 | ** 303.2** | ** 265.1** |  0.90x | 0.78x |  0.87x |
|  8192 ×  8 |  111.9 |  116.0 | ** 105.7** | **  82.3** |  0.94x | 0.74x |  0.78x |

## Read

Two different performance envelopes:

- **Upstream wins at short-ctx low-load** (s1024×8): TPOT 19.2 ms (vs our 29.5,
  FA 22.7) and 253 tok/s (vs our 198, FA 235). Its Triton dequant matches FA on
  lightly-loaded decode.
- **Ours wins at long-ctx high-load** (s8192×8, s2048×32): upstream falls to
  ~30 % slower than ours and ~25 % slower than FA. Our fused dequant+attention
  CUDA kernel beats upstream's separate-dequant Triton path once the decode
  hot loop dominates.
- **TTFT** at long ctx: ours beats FA (1315 vs 1788 at s8192×8); upstream is
  *worse* than FA (1993) — its Triton write path adds prefill overhead the
  CUDA path doesn't.
- **Neither reliably beats FA at throughput** except upstream at s1024×8.

Upstream and ours solve different points on the curve: portability + short-ctx
vs A100-specialized + long-ctx / high-batch.
