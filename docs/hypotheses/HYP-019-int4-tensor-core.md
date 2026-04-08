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

## Status: pending
