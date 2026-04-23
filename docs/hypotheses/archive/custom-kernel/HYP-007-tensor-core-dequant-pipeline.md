# HYP-007: Rotated-domain attention + dequant→fp16→tensor cores will match SDPA

## Hypothesis

The 3-7× gap between v4 and SDPA is entirely from scalar FMA vs tensor core HMMA.
By (1) computing attention in the rotated domain (eliminating FWHT) and (2) dequanting
4-bit codebook indices to fp16 in registers and feeding them to fp16 tensor cores via
`mma.sync.m16n8k16.f32.f16.f16.f32`, we can close this gap.

### Why this should work

1. **Rotated-domain attention eliminates FWHT:**
   `Q·K^T = (RQ)·(RK)^T` — rotate Q once (one FWHT per head, ~1μs), then all
   attention scores are computed in the rotated space. Never inverse-rotate K/V.
   Dequant simplifies to: `codebook[index] × norm` — no cross-dimension dependency.

2. **fp16 tensor cores give 16× compute throughput:**
   A100 fp16 tensor cores: 312 TFLOPS. Our scalar FMA: ~20 TFLOPS effective.
   Even with dequant overhead, BitDecoding achieves 4.8× speedup on A100 using
   this exact approach (CUDA cores dequant + tensor cores matmul, software-pipelined).

3. **Register-resident codebook lookup is cheap:**
   16 fp16 centroids fit in 8 registers (16 × 2B = 32B). 4-bit index → 1 `prmt`
   or LUT instruction → fp16 centroid. Multiply by fp16 norm → done.
   This is simpler than Marlin's `lop3` trick (which handles uniform quant only).

4. **BitDecoding (HPCA 2026) validates the architecture:**
   They achieve 4.8× on A100 with: load quantized KV → dequant on CUDA cores →
   fp16 tensor core matmul, all software-pipelined. 8.9× on H100.
   Our setup is similar (4-bit KV, paged cache, decode attention).

## Prediction

- Latency at seq=1024: **40-80 μs** (vs v4=296μs, SDPA=60μs)
- Ratio to SDPA: **0.7-1.3×** (within striking distance, possibly faster due to 3.76× less memory)
- At long sequences (4K+): **faster than SDPA** (memory-bandwidth savings dominate)

## Method

### Phase 1: Rotated-domain attention (no tensor cores yet)
Validate that computing attention in the rotated domain produces correct results.
- Rotate Q via FWHT before the kernel
- Remove inverse rotation of V output from the kernel
- Dequant K/V as: `codebook[index] × norm` (no FWHT)
- Should produce identical cosine to current approach

### Phase 2: Dequant→fp16→tensor core matmul
Replace scalar QK/V loops with tensor core `mma.sync`:
- Use CUTLASS or raw PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- Tile Q×K^T as: Q [1×head_dim] × K^T [head_dim×tile_tokens] → scores [1×tile_tokens]
- Pipeline: dequant tile N+1 on CUDA cores while tensor cores compute tile N

### Phase 3: Software pipeline (BitDecoding pattern)
Overlap three stages:
1. Load quantized KV from VRAM (cp_async or global load)
2. Dequant to fp16 on CUDA cores (codebook LUT + norm multiply)
3. Tensor core matmul (mma.sync on fp16 data)

Stages 2 and 3 use different execution units (INT/FP ALU vs tensor cores) —
they can run concurrently within the same warp.

## Prior art

- **BitDecoding** (Du et al., HPCA 2026, arXiv:2503.18773) — 4.8× on A100.
  CUDA core dequant + tensor core matmul, fragment-aware layout, software pipeline.
- **Marlin** (Frantar et al., arXiv:2408.11743) — near-4× for W4A16 GEMM.
  `lop3` bit trick for uniform INT4→fp16. Register-level pipeline interleaving.
- **SageAttention2** (Zhang et al., ICML 2025) — INT4 tensor cores for QK.
  Per-thread quantization aligned with MMA fragment layout.
- **QServe** (Lin et al., MLSys 2025) — W4A8KV4 with INT8 tensor cores.
- **KVQuant** (Hooper et al., NeurIPS 2024) — non-uniform codebook LUT, no tensor cores. 1.7×.

## Risks

1. **MMA fragment layout**: Tensor core operands must be in specific register layouts
   (m16n8k16 fragments). Dequantized fp16 values must land in the right registers.
   BitDecoding solves this with layout-aware dequantization — needs careful study.

2. **Decode attention is rank-1**: Q is a single vector (not a matrix), so the
   Q×K^T "matmul" is really a matrix-vector product. Tensor cores are optimized
   for matrix-matrix — rank-1 may underutilize them. FlashInfer handles this by
   batching across GQA heads (bdy > 1) to create a small matrix.

3. **Codebook LUT vs linear dequant**: Marlin's `lop3` trick works for uniform quant
   only. Our Lloyd-Max codebook needs a true LUT. Register-resident LUT (16 entries)
   should work but needs verification of instruction count.

4. **Paged KV with tensor cores**: The paged layout (scattered pages) may break
   coalescing assumptions needed for efficient tensor core feeding. May need to
   restructure the tile loading.

## Status: pending
