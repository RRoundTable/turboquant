# HYP-023: CUDA graph capture of TQ decode kernel

## Hypothesis
CUDA graph replay eliminates kernel launch overhead (~3μs per launch).
Split-KV has 2 launches per layer (decode + combine). 28 layers × 2 × 3μs = 168μs savings.

## Results (A100, Qwen3-1.7B, batch=1)

| seq | Eager | CUDA Graph | Speedup | Save/call |
|-----|-------|-----------|---------|-----------|
| 128 | 21.8 μs | 25.6 μs | 0.85× (slower) | -3.8 μs |
| 256 | 31.8 μs | 35.7 μs | 0.89× (slower) | -3.9 μs |
| **512** | 46.5 μs | **35.5 μs** | **1.31×** | **+11.0 μs** |
| **1024** | 46.9 μs | **37.3 μs** | **1.26×** | **+9.6 μs** |
| 2048 | 60.6 μs | 57.6 μs | 1.05× | +3.0 μs |

seq≤256 (nosplit): graph adds overhead (single small kernel, capture cost > launch cost).
seq=512-1024 (split-KV): graph saves 10-11μs (multiple launches benefit from replay).

## 28-layer TPOT projection

| seq | Eager 28L | Graph 28L | Save |
|-----|----------|----------|------|
| 512 | 1.31 ms | 0.99 ms | +0.32 ms |
| 1024 | 1.31 ms | **1.04 ms** | **+0.27 ms** |
| 2048 | 1.69 ms | 1.61 ms | +0.08 ms |

Projected full model TPOT (seq=1024):
  FP16 + CUDA graph: 1.20ms
  TQ + CUDA graph:   0.19ms + 1.04ms = **1.23ms (2.5% overhead)**

## Status: confirmed
CUDA graph capture reduces TQ decode overhead from 23% to 2.5% at seq=1024.
Only effective for split-KV (seq≥512). Use adaptive: eager at seq≤256, graph at seq≥512.
