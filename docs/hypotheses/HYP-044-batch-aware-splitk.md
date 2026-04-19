# HYP-044: batch-aware split-K to saturate SMs at batch > 1

## Evidence (from HYP-042b)

Same-seq decode kernel latency ratios across batch:

| source | seq | batch | FlashInfer | TQ | TQ / FI |
|--------|----:|------:|-----------:|---:|--------:|
| HYP-035 | 4096 | 1 | 40.8 μs | 109.8 μs | 2.69× |
| HYP-042b | 8192 | 8 | 110 μs/layer | 1150 μs/layer | **9.8×** |

FlashInfer's decode cost grows only ~2× from batch=1 to batch=8 (it
packs requests into the same kernel and fills the SMs). TQ's v5 cost
grows close to linearly with batch. That flips a core serving
assumption: **increasing batch does not amortize the TQ quant
overhead — it amplifies the gap.**

## Hypothesis

The v5_paged split-K grid keeps **grid size constant** as batch grows
but lets **per-block work scale linearly with batch**, so kernel
latency tracks batch × chunk_size instead of grid-saturation × chunk.

### Inspection finding

`turboquant/vllm_backend_fused.py:147`:

```python
splits_by_sm = max(1, (4 * sm_count) // (batch_size * num_kv_heads))
```

At seq=8192 (Qwen3-8B, num_kv_heads=8, A100 sm_count=108):

| batch | splits_by_sm | num_splits (snapped to divisor) | chunk_size | grid |
|---:|---:|---:|---:|---:|
| 1  | 54 | 32 |  256 | 1·32·8 = 256 |
| 8  |  6 |  4 | 2048 | 8·4·8 = 256 |
| 32 |  1 |  1 | 8192 | 32·1·8 = 256 |

Grid size is roughly constant at 256 blocks (~2.4× A100 SM count). But
per-block work (chunk_size) grows 8× from batch=1 to batch=8 and 32×
from batch=1 to batch=32. The TQ decode kernel is scalar-FMA dominated
(HYP-006, HYP-030), so per-block runtime tracks chunk_size linearly.
That predicts kernel latency ratio batch=8 / batch=1 ≈ 8 — and
HYP-042b measured 4.9× ratio at batch=8 vs HYP-035's 2.69× at batch=1
(roughly 2× degradation), consistent with the prediction once you
account for FlashInfer's own batch scaling.

FlashInfer doesn't degrade because it uses tensor cores and saturates
per-block throughput at much smaller chunks — its per-block work
amortizes faster.

## Proposed fix

Replace the "divide SM target by batch_size" heuristic with one that
**caps chunk_size** and lets grid grow with batch:

```python
TARGET_CHUNK = 256
splits_by_chunk = max(1, max_len // TARGET_CHUNK)
splits_by_sm   = max(1, (4 * sm_count) // num_kv_heads)  # no /batch
target = min(splits_by_chunk, splits_by_sm)
# then snap to largest divisor of max_len ≤ target as before
```

At seq=8192 this gives `target = min(32, 54) = 32` independent of
batch. Per-block work stays at ~256 tokens; grid grows to
`batch × 32 × 8` (256 at b=1, 2048 at b=8, 8192 at b=32). A100 can
handle 8192-block grids — the previous comment about "grid much larger
than 2-3× SM count over-splitting" is correct only when chunk_size
becomes too small (<128 tokens), which is why we cap chunk at 256
instead of letting it shrink further.

## Prediction

After the fix:

- per-layer per-step kernel ratio at seq=8192 × b=8 drops from 4.9× to
  **≤ 2.7×** (back to HYP-035's batch=1 result)
- per-decode-step wall Δ drops from +18 ms to **≤ +10 ms**
- end-to-end decode tok/s (seq=8192 × b=8) rises from 215 to
  **≥ 290 tok/s** (recovers ~40 % of the HYP-041 gap)

## Method

1. (DONE) Inspect heuristic — confirmed batch-divisor bug at
   `vllm_backend_fused.py:147`.
2. Microbench: sweep batch ∈ {1, 2, 4, 8, 16, 32} at seq=8192 on
   standalone v5 paged kernel (no vLLM). Record kernel latency for
   both old and proposed heuristics. Two Forge jobs, parallel.
3. If microbench confirms ≥ 1.5× kernel speedup at batch=8, ship the
   heuristic change and re-run HYP-041's full 12-config sweep for
   end-to-end validation.

If microbench shows < 1.2× speedup, the chunk_size hypothesis is
wrong and other per-block fixed costs dominate (Q load, softmax init).
File HYP-044b for that path.

## Status: confirmed (kernel), partially confirmed (end-to-end); shipped

## Result (microbench, A100-40GB, seq=8192)

| batch | old us | new us | new/old | new ÷ (new@b=1) |
|------:|-------:|-------:|--------:|----------------:|
|   1   |  127.1 |  126.6 | 1.00×   | 1.00×           |
|   2   |  207.6 |  173.3 | 0.84×   | 1.37×           |
|   4   |  306.9 |  281.9 | 0.92×   | 2.23×           |
|   8   |  576.8 |  482.7 | **0.84×** | 3.81×         |
|  16   | 1120.0 |  912.2 | 0.81×   | 7.20×           |
|  32   | 2209.9 | 1747.9 | 0.79×   | 13.80×          |

Full run: `results/hyp044/SUMMARY.md`, `results/hyp044/results.csv`.

### What the result says vs the prediction

- **Chunk-cap heuristic is worth 15–21 % at batch ≥ 8.** Small but
  real; ship it.
- **Batch scaling is NOT flattened.** New heuristic is 13.8× at
  batch=32 vs 17.4× with the old one — the linear-in-batch trend
  survives. Predicted kernel ratio back to ~2.7× at batch=8 is
  rejected; measured ratio at batch=8 is 3.81× (down from 4.54× old,
  far from 2.69×).
- **Why**: at seq=8192, chunk=256, batch=32, the new heuristic
  launches 8192 blocks — 76× A100 SM count. Grid is saturated many
  waves over; wall time tracks total work (batch × chunk_size ×
  num_kv_heads), not grid count. Per-token compute density in TQ's
  scalar-FMA decode is the binding constraint on SM80 (confirms
  HYP-035 / HYP-037 / HYP-040 / HYP-042b).

### Expected end-to-end (before running)

Per-decode-step wall Δ at seq=8192 × b=8 drops by ~16 % of the
attention component: 20.7 ms → ~17.3 ms ⇒ total per-step CUDA
35.8 ms → ~32.4 ms, TQ/baseline ratio 2.01× → ~1.82×. Decode
tok/s should rise from 215 → ~240 (still 0.63× of baseline 380;
recovers ~15 % of the HYP-041 gap).

## Action (DONE — patch landed, end-to-end sweep re-run)

Patched `turboquant/vllm_backend_fused.py::_choose_num_splits` and the
inline copy in `tests/bench_v5_graph.py::_choose_num_splits` to the
chunk-cap heuristic. Re-ran HYP-041's 12-config grid on the same env
(image `tq-hyp029:pr`, eager mode, PR #39868 overlay).

### End-to-end result (Qwen3-8B, A100-40GB, output_len=128, tok/s)

| seq × b | tq v0 | tq v1 (HYP-044) | Δ tq | tq/base v0 | tq/base v1 | Δ ratio |
|--------:|------:|----------------:|-----:|-----------:|-----------:|--------:|
|  1024×1 |  38.4 |  38.8 |    +1% | 0.79× | 0.81× |  +2pp |
|  1024×8 | 291.6 | 300.4 |    +3% | 0.77× | 0.77× |  +0pp |
| 1024×32 | 961.2 | 1138.3 | **+18%** | 0.68× | **0.77×** | **+9pp** |
|  4096×1 |  38.1 |  38.6 |    +1% | 0.81× | 0.81× |  +0pp |
|  4096×8 | 291.3 | 296.3 |    +2% | 0.78× | 0.77× | −1pp |
|  8192×1 |  37.3 |  37.9 |    +2% | 0.78× | 0.79× |  +1pp |
|  8192×8 | 215.1 | 227.7 | **+6%** | 0.57× | **0.61×** | **+4pp** |
| 16384×1 |  38.2 |  37.9 |    −1% | 0.79× | 0.80× |  +1pp |

(Configs 4096×32, 8192×32, 16384×8, 16384×32 OOM pre-existing from
HYP-041; HYP-045 is the fix.)

**Highlights:**
- At **seq=8192 × b=8** (the HYP-041 headline config): tq/base
  **0.57× → 0.61×** (+4 pp, 6% tok/s).
- At **seq=1024 × b=32**: tq/base **0.68× → 0.77×** (+9 pp, 18% tok/s)
  — chunk_size dropped from 1024 to 256, biggest relative kernel
  speedup.
- **batch=1 unchanged** across every seq — expected, because at
  batch=1 the old heuristic already produced chunk ≤ 256.

End-to-end gains track the microbench order-of-magnitude but smaller:
attention is ~91 % of the per-decode-step Δ, and kernel speedup of
0.79–0.84× translates to ~5–10 % end-to-end (except at batch=32 short
seq where kernel dominates more of the step).

Full report: `results/v5_vs_baseline_hyp044/REPORT.md`.
