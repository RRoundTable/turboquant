# HYP-022: Fuse combine into decode kernel to save 5μs/layer

## Hypothesis
The split-KV combine is a separate kernel launch adding ~5μs per layer.
Fusing it into the decode kernel (last block does the combine) saves this.
28 layers × 5μs = 140μs total savings on TPOT.

## Prediction
- Per-layer: 46μs → ~41μs
- TPOT (CUDA graphs): 1.48ms → ~1.34ms (9% improvement)

## Method
After the decode kernel writes partial results to `partition_o/partition_lse`,
the last split block to finish does the combine reduction. Two approaches:

**A. Atomic counter**: Each block atomically increments a counter. The last block
(counter == num_splits-1) reads all partials and writes final output.

**B. Two-pass in same kernel**: First grid does decode. Then a second small grid
(batch × qo_heads blocks) does the combine. Both in one kernel launch via
cooperative groups `grid.sync()`. But cooperative launch has restrictions.

Approach A (atomic counter) is simpler and works without cooperative launch.

## Status: pending
