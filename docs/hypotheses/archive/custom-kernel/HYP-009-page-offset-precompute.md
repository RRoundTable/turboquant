# HYP-009: Precompute page offsets — now feasible with v4's low smem

## Hypothesis
v4's paging overhead is 32μs (89μs paged vs 57μs contiguous at bdz=16, seq=1024).
HYP-002 rejected page offset precomputation because it added 2.2KB smem on top of
v2's 8.2KB, hurting occupancy. v4 uses only ~1.2KB for staging — plenty of headroom.

Precomputing page_idx and entry_idx once per tile, shared across cp_async and norms,
will eliminate redundant divmod and __ldg calls, reducing the 32μs paging overhead.

## Prediction
10-15μs reduction (89→74-79μs). Won't eliminate all 32μs because scattered page
access pattern still hurts memory coalescing.

## Method
Modify v4 kernel:
1. Add smem arrays: page_idx[tile_tokens] and entry_idx[tile_tokens]
2. Phase 0: cooperatively compute divmod + __ldg(indices) for all tile rows
3. cp_async_packed_tile reads from smem page_idx/entry_idx (no divmod)
4. precompute_norms reads from smem page_idx/entry_idx (no divmod)

Extra smem: tile_tokens * (4 + 4) = 64 * 8 = 512 bytes (at bdz=16). Negligible.

## Results

| seq | Before (μs) | After (μs) | Diff |
|-----|-------------|------------|------|
| 128 | 35.9 | 34.2 | -5% |
| 512 | 52.9 | 53.5 | +1% |
| 1024 | 89.0 | 91.3 | +3% |
| 4096 | 253.2 | 285.3 | **+13%** |

**Net negative at long sequences.** The extra `block.sync()` for Phase 0 adds ~0.5μs
per iteration. At seq=4096 with 64 iterations: +32μs overhead > divmod savings.

The original design (redundant divmod in cp_async and norms) is actually better because
both run without an intervening sync — the divmod is interleaved with other work.

This is the SAME conclusion as HYP-002 but for a different reason:
- HYP-002 (v2): failed due to smem pressure reducing occupancy
- HYP-009 (v4): failed due to extra sync overhead exceeding divmod savings

**Lesson:** block.sync() is expensive (~0.5μs). Adding syncs to "save" compute that
costs less than 0.5μs is counterproductive.

## Status: rejected
Reverted. Extra sync cost > divmod savings.
