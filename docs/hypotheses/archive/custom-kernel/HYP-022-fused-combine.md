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

## Results

Correctness: perfect (cos=1.0, max_diff=0.0). But **slower**:

| seq | Separate | Fused | Delta |
|-----|---------|-------|-------|
| 512 | 49.7 μs | 53.3 μs | -3.6 μs (7% slower) |
| 1024 | 49.0 μs | 53.0 μs | -4.0 μs (8% slower) |
| 2048 | 57.8 μs | 62.4 μs | -4.5 μs (8% slower) |

__threadfence() blocks ALL warps in the block (~3-4μs), which exceeds the
~3μs kernel launch overhead it was supposed to save. The atomic counter
approach adds overhead that the separate combine kernel doesn't have.

## Status: rejected
Fused combine is 7-8% slower due to __threadfence + atomic overhead.
