# HYP-044 — batch-aware split-K microbench

Standalone v5 paged kernel (no vLLM), A100-40GB, seq=8192, Qwen3-8B
shape (num_kv_heads=8, num_qo_heads=64, head_dim=128), CUDA-graph
captured timing. Old heuristic from `vllm_backend_fused.py:147`
(`splits_by_sm = (4 * sm_count) // (batch * num_kv_heads)`, divisor
snap); proposed new heuristic caps chunk at 256 tokens and does not
divide by batch.

## Results

| batch | old us | new us | new/old | new ÷ (new@b=1) | old ÷ (old@b=1) |
|------:|-------:|-------:|--------:|----------------:|----------------:|
|   1   |  127.1 |  126.6 | 1.00×   | 1.00×           | 1.00×           |
|   2   |  207.6 |  173.3 | 0.84×   | 1.37×           | 1.63×           |
|   4   |  306.9 |  281.9 | 0.92×   | 2.23×           | 2.41×           |
|   8   |  576.8 |  482.7 | **0.84×** | 3.81×         | 4.54×           |
|  16   | 1120.0 |  912.2 | 0.81×   | 7.20×           | 8.81×           |
|  32   | 2209.9 | 1747.9 | 0.79×   | 13.80×          | 17.39×          |

## Read

- **Chunk cap is worth 15–21% at batch ≥ 8.** Real speedup, ship it.
- **But does NOT flatten batch scaling.** New heuristic is 13.8× at
  batch=32 vs 17.4× with the old one — the linear-in-batch trend
  survives.
- Per-decode-step wall Δ (HYP-041/HYP-042b) at seq=8192 × b=8 should
  drop by ~16% of the attention component: 20.7 ms → ~17.3 ms ⇒
  per-step total CUDA 35.8 ms → ~32.4 ms, TQ / baseline ratio
  2.01× → ~1.82×. End-to-end tok/s at seq=8192 × b=8 should rise from
  215 → ~240 (still 0.63× of baseline 380; recovers ~15% of the gap).

## Why batch scaling survives

At seq=8192, chunk=256, batch=32 the new heuristic launches
`32 × 32 × 8 = 8192` blocks — 76× the A100 SM count. Grid is
saturated many waves over; wall time tracks total work
(batch × chunk_size × num_kv_heads) not grid count. Per-token compute
cost in TQ's scalar-FMA decode is the binding constraint (confirms
HYP-035 / HYP-037 / HYP-040 / HYP-042b — the A100 kernel ceiling is
compute-density-limited, not split-K-limited).

## Decision

- **HYP-044 partially confirmed.** Ship the chunk-cap heuristic
  (patch `_choose_num_splits` in both `vllm_backend_fused.py` and
  `tests/bench_v5_graph.py`). Expected end-to-end: ~15% tok/s gain at
  batch ≥ 8, no change at batch = 1.
- **The "kernel ratio back to 2.7× at batch=8" prediction is rejected.**
  That would require a compute-density change (tensor cores, denser
  dequant), which HYP-037 / HYP-040 showed is not reachable on SM80.
  Revisit on H100 in HYP-046.
- Still the best A100-side lever available. HYP-045 (workspace
  pre-alloc) and HYP-046 (H100) remain the other pending items.

Raw data: `results/hyp044/results.csv`, `results/hyp044/run.log`.
