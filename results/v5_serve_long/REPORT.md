# Long-context serving: FA / FI / ours vs upstream turboquant

Qwen/Qwen3-8B fp16 on A100-40GB. output_len=128, num_prompts=64.
- FA / FI / ours: stock vllm v0.19.0 + docker/vllm_patches (PR #39868).
- upstream: vllm nightly 0.19.2rc1.dev21+g893611813, `--kv-cache-dtype turboquant_4bit_nc`.

## Median TTFT (ms) — lower is better

| seq × c |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|---------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|
| 16384 ×  8 | 4005 | 3731 | ** 1833** | ** 3173** |  0.46x | 0.79x |  1.73x |
| 32768 ×  4 | 9187 | 8328 | ** 3356** | ** 7716** |  0.37x | 0.84x |  2.30x |
| 32768 ×  8 | 31758 | 29101 | ** 3194** | ** 7500** |  0.10x | 0.24x |  2.35x |

## Median TPOT (ms) — lower is better

| seq × c |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|---------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|
| 16384 ×  8 | 97.9 | 90.9 | ** 91.6** | **145.1** |  0.94x | 1.48x |  1.58x |
| 32768 ×  4 | 87.5 | 80.3 | ** 82.9** | **157.9** |  0.95x | 1.80x |  1.91x |
| 32768 ×  8 | 87.7 | 80.4 | **138.5** | **334.2** |  1.58x | 3.81x |  2.41x |

## Output throughput (tok/s) — higher is better

| seq × c |    FA |    FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |
|---------|------:|------:|---------:|-------------:|--------:|------:|--------:|
| 16384 ×  8 |  60.0 |  64.2 | **  68.9** | **  44.4** |  1.15x | 0.74x |  0.64x |
| 32768 ×  4 |  24.1 |  26.3 | **  36.4** | **  18.5** |  1.51x | 0.77x |  0.51x |
| 32768 ×  8 |  24.1 |  26.2 | **  45.4** | **  19.9** |  1.89x | 0.83x |  0.44x |
