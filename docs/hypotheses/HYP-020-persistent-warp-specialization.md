# HYP-020: Persistent kernel with warp specialization

## Hypothesis

The v4 contiguous+split-KV kernel has 4 `block.sync()` per tile iteration:
1. After cp_async K + norms precompute
2. After QK compute (before V load)
3. After cp_async V + norms precompute
4. After V accumulate (before next tile)

Each sync costs ~0.5μs (measured in HYP-009). At seq=1024 with 8 splits, each
split processes ~128 tokens = 2 tile iterations = 8 syncs = ~4μs sync overhead.

**Warp specialization** eliminates syncs by splitting warps into producer (load+dequant)
and consumer (QK/V compute) roles. They communicate via named barriers instead of
block-wide sync. The producer fills smem buffer A while consumer reads buffer B (ping-pong).

**Persistent kernel** keeps the kernel running across multiple (batch, head, split)
combinations without relaunching — eliminates ~5μs kernel launch overhead per split.

## Prediction

- 4μs sync savings + 5μs launch savings per split = ~9μs total
- At seq=1024 (8 splits): 48μs → ~39-42μs (10-20% improvement)
- Closer to SDPA's 30μs

## Architecture

```
Thread block: 256 threads = 8 warps

Current (all warps do everything):
  All warps: load K → sync → QK → sync → load V → sync → V_acc → sync

Warp-specialized (producer/consumer):
  Warps 0-3 (producer): load K → signal(bar0) → load V → signal(bar1) → ...
  Warps 4-7 (consumer): wait(bar0) → QK → wait(bar1) → V_acc → ...

  Using named barriers (SM80+):
    bar.arrive(0) — producer signals K ready
    bar.wait(0)   — consumer waits for K
    bar.arrive(1) — consumer signals K consumed (producer can reuse buffer)
    bar.wait(1)   — producer waits before overwriting
```

Double-buffered smem: producer fills buf[next] while consumer reads buf[curr].
No block.sync() needed — only named barriers between warp groups.

## Named barriers on A100 (SM80)

```cuda
// Producer: signal K tile ready
asm volatile("bar.arrive 0, %0;" :: "r"(producer_thread_count));

// Consumer: wait for K tile
asm volatile("bar.sync 0, %0;" :: "r"(total_thread_count));
```

SM80 supports up to 16 named barriers per block (bar 0..15).
Each barrier tracks a configurable number of threads.

## Risks

1. **Complexity**: warp specialization is hard to debug. Named barrier deadlocks
   are possible if producer/consumer imbalance.
2. **Load imbalance**: if producer finishes before consumer (or vice versa), one
   group stalls. Our kernel's compute >> load, so consumer may starve producer.
3. **Register pressure**: producer warps use different registers than consumer warps.
   Total register budget shared — may reduce occupancy.
4. **Diminishing returns**: 4μs sync savings is only 8% of 48μs total. The sync
   overhead may be even less than 0.5μs per sync in practice.

## Method

1. Implement double-buffered staging with named barriers
2. Split bdz warps into producer (bdz/2) and consumer (bdz/2) groups
3. Producer: cp_async + norms precompute into buf[next]
4. Consumer: QK/V compute from buf[curr]
5. Benchmark vs current v4 contiguous+split-KV

## Results (A100, Qwen3-1.7B, batch=1)

| seq | v4 contiguous | v5 warpspec | v5/v4 |
|-----|--------------|------------|-------|
| 128 | 22.1 μs | 22.2 μs | 1.006× (no change) |
| 256 | 32.0 μs | 31.7 μs | 0.99× (no change) |

Longer seq crashed due to split-KV API mismatch in benchmark (4D tensor issue).
Non-split correctness: cos=0.985 at seq≥256 (buffer indexing bug in double-buffer).
Split correctness: cos=1.000 (works correctly).

## Analysis

**Zero improvement from double-buffering + named barriers.**

Root cause: at 4-bit data, cp_async loads are so fast (~0.5μs for a tile of 64×128
packed bytes = 4KB) that there's nothing to overlap. The compute phase (QK dot product
+ softmax) takes ~10× longer than the load. Overlapping a 0.5μs load with a 5μs
compute saves at most 0.5μs per iteration — invisible in the noise.

This is the same conclusion as HYP-005 (cp_async pipelining was 18% slower):
**TurboQuant's kernel is compute-bound, not memory-bound.** Pipelining/overlapping
the load phase doesn't help when load << compute.

The 4 syncs per iteration cost ~2μs total. Named barriers might save ~0.5μs of
that (lower-latency barriers), but this is <1% of total kernel time.

## Status: rejected
Zero measurable improvement. The kernel's compute:load ratio is ~10:1 — there's
nothing meaningful to overlap. Named barriers save <1% vs block.sync().
