# HYP-001: bdz parallelism will scale linearly with thread count

## Hypothesis
Increasing bdz (number of KV token tile groups per block) from 1→16 will proportionally reduce kernel latency because the bottleneck is serial token processing.

## Prediction
~8-16× speedup from bdz=1 to bdz=16. Latency should scale as O(1/bdz).

## Method
Sweep bdz ∈ {1, 2, 4, 8, 16} with standalone kernel, seq=1024, Qwen3 config.

## Results
| bdz | Threads | Latency (μs) | Speedup |
|-----|---------|-------------|---------|
| 1   | 16      | 856         | 1.0×    |
| 2   | 32      | 476         | 1.8×    |
| 4   | 64      | 278         | 3.1×    |
| 8   | 128     | 176         | 4.9×    |
| 16  | 256     | 142         | 6.0×    |

Diminishing returns after bdz=8 (6× not 16×). Bottleneck shifts from token parallelism to compute/memory at high bdz.

## Status: confirmed (partially)
Speedup confirmed but sub-linear. 6× at bdz=16, not the predicted 8-16×. Compute-bound regime reached.
