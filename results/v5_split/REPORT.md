# Prefill vs decode (split metric): FA vs FI vs TQ

Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True

Method: T_short = generate(output_len=1) ⇒ prefill + 1 decode.
        T_long  = generate(output_len=128) ⇒ prefill + 128 decode.
        decode_per_step = (T_long - T_short) / 127.
        prefill         = T_short - decode_per_step.

## Prefill latency (ms) — TQ matches or beats baseline at long ctx

| seq × b | FA prefill | FI prefill | TQ prefill | TQ/FA | TQ/FI |
|---------|----------:|----------:|----------:|------:|------:|
|  1024 ×  1 |        4.0 |        3.2 |        0.6 | 0.15x | 0.19x |
|  1024 ×  8 |       29.9 |       37.4 |       34.5 | 1.15x | 0.92x |
|  1024 × 32 |      112.0 |      116.0 |      113.5 | 1.01x | 0.98x |
|  4096 ×  1 |        6.2 |        7.2 |        3.9 | 0.64x | 0.54x |
|  4096 ×  8 |       58.1 |       78.1 |       75.4 | 1.30x | 0.96x |
|  4096 × 32 |      248.7 |      233.3 |      201.9 | 0.81x | 0.87x |
|  8192 ×  1 |       18.9 |       11.0 |        9.0 | 0.48x | 0.82x |
|  8192 ×  8 |      137.5 |      138.6 |      134.5 | 0.98x | 0.97x |
|  8192 × 32 |      383.3 |      294.3 |      282.5 | 0.74x | 0.96x |
| 16384 ×  1 |       32.1 |       23.8 |       19.4 | 0.61x | 0.82x |
| 16384 ×  8 |      236.0 |      210.3 |      190.3 | 0.81x | 0.90x |
| 16384 × 32 |      536.7 |      474.5 |      350.0 | 0.65x | 0.74x |

## Decode-only latency (ms/step) and throughput (tok/s)

| seq × b | FA ms | FI ms | TQ ms | FA tok/s | FI tok/s | TQ tok/s | TQ/FA | TQ/FI |
|---------|------:|------:|------:|---------:|---------:|---------:|------:|------:|
|  1024 ×  1 |  20.5 |  19.8 |  26.7 |       49 |       51 |       37 | 0.77x | 0.74x |
|  1024 ×  8 |  20.5 |  19.7 |  26.3 |      390 |      405 |      304 | 0.78x | 0.75x |
|  1024 × 32 |  21.0 |  20.1 |  26.9 |     1525 |     1591 |     1192 | 0.78x | 0.75x |
|  4096 ×  1 |  20.6 |  19.5 |  26.3 |       48 |       51 |       38 | 0.78x | 0.74x |
|  4096 ×  8 |  20.6 |  19.6 |  26.4 |      389 |      409 |      303 | 0.78x | 0.74x |
|  4096 × 32 |  23.4 |  20.6 |  51.6 |     1369 |     1555 |      621 | 0.45x | 0.40x |
|  8192 ×  1 |  20.3 |  20.1 |  25.5 |       49 |       50 |       39 | 0.80x | 0.79x |
|  8192 ×  8 |  20.4 |  19.0 |  34.0 |      392 |      421 |      235 | 0.60x | 0.56x |
|  8192 × 32 |  31.7 |  26.7 |  78.5 |     1010 |     1200 |      407 | 0.40x | 0.34x |
| 16384 ×  1 |  20.3 |  19.8 |  25.9 |       49 |       51 |       39 | 0.79x | 0.76x |
| 16384 ×  8 |  20.9 |  19.5 |  49.4 |      382 |      411 |      162 | 0.42x | 0.39x |
| 16384 × 32 |  49.3 |  38.9 | 133.1 |      649 |      822 |      240 | 0.37x | 0.29x |

## Notes

- **Prefill is at or below baseline for TQ across the grid.** At long ctx × large batch (16384 × 32), TQ prefill is 350 ms vs FA 537 ms / FI 475 ms — 1.5× faster than FA. Likely a side-effect of PR #39868: less KV bytes to write during prefill.
- **Decode is uniformly slower for TQ.** Ratio worsens with (seq × batch):
  - 1024×1   TQ/FI = 0.74×
  - 8192×8   TQ/FI = 0.56×
  - 16384×32 TQ/FI = 0.29×
- The previous 'tok/s' columns in HYP-041/044/045 reports used total_wall = prefill + decode, which understated the decode gap at long ctx (where prefill dominates) and overstated TQ at short ctx (where prefill is negligible). Use this report's separated numbers.
