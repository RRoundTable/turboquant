# HYP-007a: WMMA tensor cores for QK dot product — feasibility test

## Hypothesis
Replacing the scalar warp-shuffle QK dot product with WMMA `mma_sync` (fp16 tensor cores)
will reduce QK compute time, even for the rank-1 decode case (single query per head).

## Prediction
- WMMA QK: 2-4× faster than scalar FMA for the QK phase
- Main concern: rank-1 (bdy=1) underutilizes the 16-row M dimension
- GQA (bdy=4+) should see better utilization

## Method
Microbenchmark on A100: fp16 Q [M, D] × fp16 K^T [D, N] using WMMA vs scalar warp-shuffle,
where M=1..16, N=16, D={64,128}. Single warp, 1000 iterations.

## Results

| M (bdy) | D=64 scalar | D=64 WMMA | Speedup | D=128 scalar | D=128 WMMA | Speedup |
|----------|------------|-----------|---------|-------------|------------|---------|
| 1        | 8.0 μs     | 8.5 μs    | **0.94×** | 9.6 μs      | 13.8 μs    | **0.70×** |
| 4        | 21.6 μs    | 8.5 μs    | **2.5×**  | 25.9 μs     | 13.8 μs    | **1.9×**  |
| 8        | 34.0 μs    | 7.3 μs    | **4.7×**  | 40.7 μs     | 12.1 μs    | **3.4×**  |
| 16       | 64.8 μs    | 8.2 μs    | **7.9×**  | 73.2 μs     | 12.0 μs    | **6.1×**  |

Correctness: cos=1.000000 for all configs (both scalar and WMMA).

## Analysis

**WMMA time is nearly constant across M** (~8μs for D=64, ~13μs for D=128). The tensor
core does the full 16×16 matmul regardless of how many M rows are valid. Scalar time
scales linearly with M.

**Critical finding: WMMA only helps when bdy ≥ 4 (GQA ratio ≥ 4:1).**

| GQA ratio | bdy | D=128 speedup | Models |
|-----------|-----|---------------|--------|
| 1:1       | 1   | 0.70× (SLOWER) | — |
| 2:1       | 2   | ~1.1× (marginal) | Qwen3-1.7B (16qo/8kv) |
| 4:1       | 4   | 1.9×  | Llama-3-8B (32qo/8kv) |
| 8:1       | 8   | 3.4×  | Llama-3-70B (64qo/8kv) |
| 16:1      | 16  | 6.1×  | — |

**For Qwen3-1.7B (our primary target, bdy=2): tensor cores give marginal improvement.**
The rank-1/rank-2 decode case underutilizes the 16×16 M dimension.

## Implications for HYP-007

The tensor core path (Phase 2) is **model-dependent**:
- For GQA-heavy models (Llama-3, Mixtral): 2-6× speedup from tensor cores. Worth pursuing.
- For our Qwen3-1.7B target (bdy=2): marginal. Other optimizations may be higher impact.

Alternative approaches for low-bdy models:
1. **Batch across requests**: fill the M dimension with different batch items (FlashDecoding)
2. **Wider tiles**: process more KV tokens per warp (increase N beyond 16)
3. **INT4 tensor cores**: `mma.m16n8k64.s4.s4.s32` processes 4× more K-elements per
   instruction, potentially better for rank-1 case (higher k-dimension parallelism)

## Status: confirmed (partially)
WMMA works correctly. Speedup confirmed for bdy≥4 but NOT for bdy=1-2.
Tensor cores alone won't close the gap for Qwen3-1.7B.
