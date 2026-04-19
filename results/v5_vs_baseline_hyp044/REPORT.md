# HYP-044 end-to-end: HYP-041 sweep with chunk-cap heuristic

Model: Qwen/Qwen3-8B (fp16) | output_len=128 | A100-40GB | enforce_eager=True

v0 = HYP-041 (old _choose_num_splits). v1 = HYP-044 patched.

| seq  | b  | base v0 | base v1 | tq v0  | tq v1  | Δ tq   | tq/base v0 | tq/base v1 | Δ ratio |
|-----:|---:|--------:|--------:|-------:|-------:|-------:|-----------:|-----------:|--------:|
| 1024 |  1 |    48.4 |    48.0 |   38.4 |   38.8 |    +1% |      0.79x |      0.81x |    +2pp |
| 1024 |  8 |   376.8 |   389.1 |  291.6 |  300.4 |    +3% |      0.77x |      0.77x |    +0pp |
| 1024 | 32 |  1415.6 |  1471.2 |  961.2 | 1138.3 |   +18% |      0.68x |      0.77x |    +9pp |
| 4096 |  1 |    47.2 |    47.7 |   38.1 |   38.6 |    +1% |      0.81x |      0.81x |    +0pp |
| 4096 |  8 |   373.9 |   383.4 |  291.3 |  296.3 |    +2% |      0.78x |      0.77x |    -1pp |
| 4096 | 32 |  1268.9 |  1271.3 |      — |      — |      — |          — |          — |       — |
| 8192 |  1 |    47.6 |    48.3 |   37.3 |   37.9 |    +2% |      0.78x |      0.79x |    +1pp |
| 8192 |  8 |   380.3 |   376.2 |  215.1 |  227.7 |    +6% |      0.57x |      0.61x |    +4pp |
| 8192 | 32 |   922.0 |   921.5 |      — |      — |      — |          — |          — |       — |
| 16384 |  1 |    48.2 |    47.2 |   38.2 |   37.9 |    -1% |      0.79x |      0.80x |    +1pp |
| 16384 |  8 |   351.3 |   351.6 |      — |      — |      — |          — |          — |       — |
| 16384 | 32 |   600.5 |       — |      — |      — |      — |          — |          — |       — |

Notes:
- tq v1/v0 = kernel speedup from HYP-044 (chunk_size cap at 256 tokens,
  no batch divisor in split count); microbench predicted 0.79–0.84× kernel latency
  at batch≥8, translating to ~15% end-to-end because attention is ~91% of per-step Δ.
- Configs marked — are HYP-041 OOMs that persist (workspace allocation, not split-K);
  HYP-045 (pre-allocated workspace) is the fix.
