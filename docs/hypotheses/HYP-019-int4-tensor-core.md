# HYP-019: INT4 tensor cores for QK dot product

## Hypothesis
The 40% of the 1.6× gap comes from scalar FMA vs tensor cores. A100 supports
`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` which processes INT4×INT4→INT32
at 1248 TOPS (4× FP16 tensor core throughput).

If we keep K as 4-bit indices and quantize Q to 4-bit on the fly, we can use INT4
tensor cores directly for the QK dot product, applying codebook scaling post-matmul.

**Mathematical basis (uniform quantization only):**
For uniform codebook: `centroid[i] = scale * (i - offset)`
Then: `sum(q_j * centroid[k_j]) = scale * (sum(q_j * k_j) - offset * sum(q_j))`
       = `scale * INT4_matmul_result - scale * offset * sum(q)`

This is exact — no approximation. The INT4 matmul gives the raw index dot product,
and we correct with a post-matmul scaling.

**For Lloyd-Max codebook: NOT exact.** The centroids are non-uniform, so
`sum(q * centroid[k]) ≠ scale * sum(q * k_index)`. Would need to switch from
Lloyd-Max to uniform quantization (slightly higher MSE).

## Prediction
- QK compute: 12μs → 3-4μs (3-4× from INT4 TC throughput)
- Total at seq=1024: 48μs → ~38-42μs (closing half the remaining gap to SDPA)
- Quality trade-off: uniform quant has ~20-30% higher MSE than Lloyd-Max

## Method
1. Implement uniform 4-bit quantization (replace Lloyd-Max codebook)
2. Write QK kernel using PTX `mma.sync.m16n8k64.s4.s4.s32`
3. Apply post-matmul correction: `score = scale * raw_score - offset_correction`
4. Benchmark vs scalar FMA and vs SDPA
5. Measure quality impact (PPL with uniform vs Lloyd-Max)

## Results (A100, Triton INT8 TC, seq=1024)

| Approach | Time | Speedup | Cosine |
|----------|------|---------|--------|
| Scalar FMA (reference) | 20 μs | 1.0× | 1.000 |
| INT8 elementwise (Triton) | 245 μs | 0.08× | 0.984 |
| INT8 tl.dot (tensor core) | 290 μs | 0.07× | 0.984 |
| FP16 dequant (Triton) | 87 μs | 0.22× | 1.000 |

**INT8 tensor cores are 15× SLOWER than scalar FMA.**

Root cause: decode attention is a rank-1 matmul (1 query × N keys).
Tensor cores are optimized for large matrix-matrix products (M≥16, N≥16, K≥32).
At rank-1, the 16×8 output tile is mostly wasted (15/16 rows unused).
Triton's tl.dot launch overhead + INT8 expansion dominate.

This confirms HYP-007a's finding: tensor cores only help at bdy≥4 (GQA ratio ≥ 4:1).
For Qwen3-1.7B (bdy=2), tensor cores are counterproductive.

Cosine=0.984 shows Lloyd-Max codebook index arithmetic loses ~1.6% accuracy.
Would need uniform quantization for exact math, but with worse MSE.

## Status: rejected
Tensor cores don't help for rank-1 decode attention at bdy≤2.
The 15× overhead from MMA tile underutilization exceeds any compute gain.
