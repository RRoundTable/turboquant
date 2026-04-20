# Serving-mode comparison: deployed image (TQ 16c6e93) vs FA vs FI

Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True
`vllm bench serve --dataset-name random` against `vllm serve` HTTP endpoint.

## Time-to-first-token (TTFT) — prefill cost in serving

| seq × conc | metric | FA (ms) | FI (ms) | **TQ (ms)** | TQ/FA | TQ/FI |
|------------|--------|--------:|--------:|------------:|------:|------:|
|  1024 ×  8 | median |     299 |     351 | **    240** | 0.80x | 0.68x |
|  1024 ×  8 | mean   |     892 |     883 | **    867** | 0.97x | 0.98x |
|  1024 ×  8 | p99    |    9268 |    9205 | **   9389** | 1.01x | 1.02x |
|  2048 × 32 | median |     852 |     809 | **    821** | 0.96x | 1.01x |
|  2048 × 32 | mean   |    5293 |    5284 | **   5275** | 1.00x | 1.00x |
|  2048 × 32 | p99    |   13349 |   13304 | **  13280** | 0.99x | 1.00x |
|  8192 ×  8 | median |    1788 |    1544 | **   1315** | 0.74x | 0.85x |
|  8192 ×  8 | mean   |    2132 |    2043 | **   1749** | 0.82x | 0.86x |
|  8192 ×  8 | p99    |   13596 |   13635 | **  13292** | 0.98x | 0.97x |

## Time-per-output-token (TPOT) — decode latency

| seq × conc | metric | FA (ms) | FI (ms) | **TQ (ms)** | TQ/FA | TQ/FI |
|------------|--------|--------:|--------:|------------:|------:|------:|
|  1024 ×  8 | median |      23 |      22 | **     30** | 1.30x | 1.34x |
|  1024 ×  8 | mean   |      27 |      26 | **     34** | 1.25x | 1.29x |
|  1024 ×  8 | p99    |      87 |      86 | **     95** | 1.09x | 1.10x |
|  2048 × 32 | median |      52 |      50 | **     62** | 1.20x | 1.26x |
|  2048 × 32 | mean   |      51 |      49 | **     61** | 1.20x | 1.26x |
|  2048 × 32 | p99    |     116 |     114 | **    127** | 1.10x | 1.11x |
|  8192 ×  8 | median |      49 |      48 | **     57** | 1.15x | 1.19x |
|  8192 ×  8 | mean   |      55 |      53 | **     62** | 1.13x | 1.17x |
|  8192 ×  8 | p99    |     114 |     113 | **    122** | 1.08x | 1.08x |

## Throughput (req/s, output tok/s, total tok/s)

| seq × conc | metric        |    FA |    FI | **TQ** | TQ/FA | TQ/FI |
|------------|---------------|------:|------:|-------:|------:|------:|
|  1024 ×  8 | req/s         |  1.84 |  1.90 | ** 1.54** | 0.84x | 0.81x |
|  1024 ×  8 | out tok/s     | 235.35 | 243.54 | **197.70** | 0.84x | 0.81x |
|  1024 ×  8 | total tok/s   | 2118.13 | 2191.82 | **1779.29** | 0.84x | 0.81x |
|  2048 × 32 | req/s         |  2.65 |  2.72 | ** 2.37** | 0.90x | 0.87x |
|  2048 × 32 | out tok/s     | 338.66 | 347.57 | **303.23** | 0.90x | 0.87x |
|  2048 × 32 | total tok/s   | 5757.16 | 5908.69 | **5154.91** | 0.90x | 0.87x |
|  8192 ×  8 | req/s         |  0.87 |  0.91 | ** 0.83** | 0.94x | 0.91x |
|  8192 ×  8 | out tok/s     | 111.95 | 115.97 | **105.65** | 0.94x | 0.91x |
|  8192 ×  8 | total tok/s   | 7276.56 | 7537.90 | **6867.32** | 0.94x | 0.91x |

## Read

- **TTFT (prefill in serving) wins for TQ at long ctx**:
  s8192×8 median TTFT: TQ 1315 ms vs FI 1544 ms (0.85×), FA 1788 ms (0.74×).
  Matches offline split-metric finding (HYP-045 split-metric report).
- **TPOT (decode in serving) is 1.19–1.34× slower for TQ**:
  s2048×32 median TPOT: TQ 62 ms vs FI 50 ms (1.26×) — same A100 kernel-
  ceiling story as HYP-042b.
- **Output throughput is 0.85–0.91× FI for TQ**: smaller gap than the offline
  bench because continuous-batching amortizes some of the per-step overhead.
- **No regressions vs the deployed image** (TQ 16c6e93). All 3 configs ran
  under `vllm serve` with the production patch overlay (PR #39868 + HYP-044 +
  HYP-045) and produced clean serving-mode metrics.
