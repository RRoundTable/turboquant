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

The v5_paged split-K grid is sized for a single request; when batch
grows, the kernel launches more blocks per-request but does not rebalance
across requests. This leaves SMs underutilized at small per-request
work but over-subscribed at large batch, so wall time scales with
batch × seq rather than with `ceil(batch × seq / num_sms)`.

## Prediction

Batch-aware split-K that picks `num_splits = ceil(batch × seq × num_kv_heads / target_blocks_per_sm × num_sms)` will:

- flatten the batch scaling: ratio at batch=8 drops from 9.8× to
  **≤ 3.5×** (same order as HYP-035 at batch=1)
- improve per-decode-step tok/s at seq=8192 × b=8 from 215 to **≥ 280**
  (compounds with HYP-043 if both land)

## Method

1. Inspect the grid computation in `csrc/src/decode_v5_tc_binding.cu`
   (or wherever `num_splits` is picked). Confirm whether the grid
   accounts for `batch_size`.
2. Microbench: sweep batch ∈ {1, 2, 4, 8, 16, 32} at seq=8192 on
   standalone kernel (no vLLM). Plot kernel latency.
3. If the grid ignores batch, change the split-selection heuristic to
   `target_blocks_per_sm = 2` across `batch × num_kv_heads × splits`.
4. Re-microbench; if flat or improved, retest end-to-end under
   HYP-041's sweep grid.

## Status: pending
