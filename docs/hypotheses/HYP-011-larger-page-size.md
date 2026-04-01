# HYP-011: Larger page_size will reduce page table overhead

## Hypothesis

Current page_size=16 means frequent page boundary crossings. Each crossing requires
a divmod + __ldg(indices) to resolve the page table. With page_size=16 and seq=1024:
64 pages, each crossing costs divmod + scattered memory access.

Larger page_size means:
1. Fewer pages → fewer page table lookups
2. More contiguous memory within each page → better coalescing
3. Less divmod overhead (divmod is computed per token, but fewer boundary effects)

The tradeoff: larger pages = more internal fragmentation (wasted space at end of
last page), and vLLM's page allocator may be tuned for specific page sizes.

## Prediction

| page_size | pages (seq=1024) | Expected speedup |
|-----------|-----------------|-----------------|
| 16        | 64              | baseline (89μs) |
| 32        | 32              | 5-10% (80-85μs) |
| 64        | 16              | 10-15% (76-80μs) |
| 128       | 8               | 12-18% (73-78μs) |
| 256       | 4               | ~same as 128 (diminishing returns) |

The speedup comes from:
- Fewer divmod operations per tile (page_size divmod is per-token)
- Better memory locality within pages (sequential tokens in same page are contiguous)
- Less page table __ldg overhead

**The divmod itself is not the main cost.** FlashInfer uses uint_fastdiv which is a
multiply+shift (~4 cycles). The real cost is the page table __ldg (global memory load
of page index, ~200 cycle latency). Larger pages reduce the FREQUENCY of page boundary
crossings, but within a tile, EVERY token still does a divmod (to find its page and entry).

So the benefit is primarily from memory locality, not divmod count.

## Method

1. Run v4 bdz=16 benchmark with page_size ∈ {16, 32, 64, 128}
2. Same KV data, just repacked into different page sizes
3. seq ∈ {256, 512, 1024, 2048}
4. Also measure: compare v4 paged vs contiguous at each page_size to isolate
   the page overhead at different granularities

## Prior knowledge

From HYP-008: contiguous (no paging) at bdz=16 was 57μs vs paged 89μs at seq=1024.
The 32μs gap is ALL paging overhead. If larger pages recover half of this (16μs), we'd
reach 73μs — a meaningful improvement.

vLLM default page_size=16 tokens. FlashInfer supports arbitrary page sizes.
Larger pages may conflict with vLLM's memory allocator assumptions.

## Results

Correctness: **all page sizes produce identical output** (cos=1.0 vs ps=16).

| seq | ps=16 | ps=32 | ps=64 | ps=128 | ps=64/ps=16 |
|-----|-------|-------|-------|--------|-------------|
| 256 | 36.0 | 36.0 | 35.8 | 36.0 | 0.99× |
| 512 | 52.6 | 52.4 | 52.1 | 52.3 | 0.99× |
| 1024 | 86.2 | 85.8 | 85.3 | 85.4 | 0.99× |
| 2048 | 150.5 | 159.4 | 152.2 | 152.3 | 1.01× |

**Page size has NO measurable effect** — all within 1% noise.

## Analysis

The 32μs paging overhead (from HYP-008: 89μs paged vs 57μs contiguous at seq=1024)
is NOT from page boundary crossings or page size. It's from the page table indirection
itself: every token requires divmod + __ldg(indices[page_iter]) regardless of page size.

With page_size=128 and seq=1024: only 8 pages, but still 1024 divmod operations
(one per token per tile iteration). The divmod cost is per-token, not per-page.

The paging overhead is fundamentally from:
1. **divmod per token** (~4 cycles via uint_fastdiv, but serialized per thread)
2. **__ldg(indices) per token** (~200 cycle latency, partially hidden by warp scheduling)
3. **Scattered global memory access** (different pages at different VRAM addresses)

None of these improve with larger page size.

## Status: rejected
Page size has zero effect on performance. The paging overhead is per-token
(divmod + indirect load), not per-page-boundary.
