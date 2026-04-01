# HYP-002: Precomputing page offsets in smem will reduce divmod overhead

## Hypothesis
Page table lookup (divmod + __ldg per token) is repeated across K/V loads and dim chunks. Caching the results in smem once per tile will eliminate redundant work.

## Prediction
5-10% latency reduction from amortized page table lookup.

## Method
Add 2.2KB static smem for page offset cache. Precompute divmod once per tile, reuse for K/V loads.

## Results
373 μs → 416 μs (**11% slower**, net negative).

Root cause: the 2.2KB extra smem reduced occupancy. Fewer concurrent blocks per SM → worse latency hiding. The divmod cost was lower than the occupancy penalty.

## Status: rejected
Smem pressure outweighed the computation savings. On DGX Spark (SM121), register pressure and smem limits make this approach counterproductive.
