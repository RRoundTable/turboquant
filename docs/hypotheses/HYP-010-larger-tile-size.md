# HYP-010: Larger tile_size_per_bdx will amortize per-tile overhead

## Hypothesis

Currently tile_size_per_bdx=4 (each thread processes 4 KV token rows per iteration).
Increasing to 8 or 16 will:
1. Reduce number of iterations (fewer block.sync() calls)
2. Amortize per-tile fixed costs (cp_async setup, norm precompute) over more tokens
3. Improve instruction-level parallelism (longer unrolled inner loops)

The tradeoff: more registers per thread (s[tile_size] and the loop body), more smem
per tile (staging scales with tile_tokens = tile_size_per_bdx × bdy × bdz).

## Prediction

tile_size_per_bdx=8: 10-20% speedup over tile_size_per_bdx=4.
tile_size_per_bdx=16: possibly slower (register pressure, smem overflow).

## Quantitative estimate (seq=1024, bdz=16, bdy=1, head_dim=128)

| tile_per_bdx | tile_tokens | iters | syncs/iter | total syncs | staging smem |
|-------------|-------------|-------|------------|-------------|-------------|
| 4           | 64          | 16    | 4          | 64          | 4 KB        |
| 8           | 128         | 8     | 4          | 32          | 8 KB        |
| 16          | 256         | 4     | 4          | 16          | 16 KB       |

Each sync costs ~0.5μs. Halving syncs from 64→32 saves ~16μs (89→73μs predicted).

**Register pressure check:**
- s[tile_size] array: tile_size = bdy × tile_per_bdx = 1 × N
  - tile_per_bdx=4: s[4] = 4 floats = 4 regs
  - tile_per_bdx=8: s[8] = 8 floats = 8 regs
  - tile_per_bdx=16: s[16] = 16 floats = 16 regs
- A100 has 65536 regs per SM, 256 per thread at full occupancy.
  Current usage ~50 regs. s[16] adds 12 regs → 62 regs. Still fits.

**Smem check (bdz=16, bdy=1, head_dim=128):**
- staging = tile_tokens × 64 bytes
  - t=8: 128 × 64 = 8192 bytes
  - t=16: 256 × 64 = 16384 bytes
- sync_state = bdz × bdy × (HD+2) × 4 = 16 × 130 × 4 = 8320 bytes
- At t=8: max(8192+norms, 8320) ≈ 9KB. Fine.
- At t=16: max(16384+norms, 8320) ≈ 17KB. Still under 48KB default.

## Method

1. Compile v4 kernel with tile_size_per_bdx ∈ {4, 8, 16}
2. Benchmark seq ∈ {128, 512, 1024, 2048} for each
3. Check for register spilling with -Xptxas -v

## Status: pending
