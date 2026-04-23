# HYP-014: --maxrregcount to increase occupancy

## Hypothesis
Capping registers from 96→64→48 will increase occupancy from 1→2+ blocks/SM,
giving the warp scheduler more warps to hide instruction latency.

## Prediction
20-40% latency reduction from doubled occupancy.

## Results
Zero effect. All register caps produce identical latency (within noise).
No spills at any level. Correctness: cos=1.000000.

| seq | default (96) | r64 | r48 |
|-----|-------------|-----|-----|
| 1024 | 102.7 μs | 102.9 μs | 103.0 μs |

## Analysis
With only 8 grid blocks and 108 SMs, most SMs get 0 blocks. The few that get 1
block can't benefit from higher occupancy because there's no second block to schedule.

Register reduction helps when blocks compete for SM slots. With 8 blocks / 108 SMs,
occupancy is irrelevant — the bottleneck is grid-level parallelism, not SM-level.

This confirms: **split-KV is the only path to lower batch=1 latency.**

## Status: rejected
Occupancy improvement is invisible when grid blocks << SM count.
