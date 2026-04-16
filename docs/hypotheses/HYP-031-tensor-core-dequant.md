# HYP-031: Tensor-core dequant for decode kernel

## Hypothesis

TurboQuant's decode kernel currently dequantizes 4-bit Lloyd-Max codes using scalar FMA instructions, which peak at ~20 TFLOPS on A100. FlashInfer's decode kernel uses fp16 tensor cores (mma.sync), which peak at 312 TFLOPS. At short sequences (seq<=256), TQ wins because its 80 bytes/token memory footprint dominates over FlashInfer's 256 bytes/token -- the kernel is memory-bound and TQ moves 3.2x less data. But as sequence length grows beyond 512, the arithmetic intensity of attention rises and the kernel transitions from memory-bound to compute-bound. At that point, FlashInfer's 15.6x compute advantage (312/20 TFLOPS) crushes TQ's 3.2x bandwidth advantage, resulting in TQ being 12x slower at seq=4096.

Replacing TQ's scalar-FMA dequant with a Marlin-style dequant-to-fp16-then-tensor-core path will bring TQ's decode latency curve from linear-in-seq to flat (like FlashInfer), while preserving the 3.2x memory advantage from 4-bit quantized KV storage.

## Prediction

- At seq=4096, TQ kernel latency should drop from 2.6ms to ~0.3-0.5ms, matching FlashInfer's ~0.2ms within 2x.
- At seq=256, TQ should remain faster than or equal to FlashInfer (bandwidth-dominated regime unchanged).
- Memory per token stays at 80 bytes (4-bit quantized KV), unchanged from current TQ.
- The crossover point where FlashInfer becomes faster should shift from seq~512 to seq~4096 or beyond.

## Method

1. **Dequant 4-bit nibbles to fp16 via bitwise ops (Marlin pattern):**
   Load packed INT4 bytes from global memory via cp_async into shared memory. In registers, extract nibbles using bitwise shifts and masks. Apply the Lloyd-Max codebook lookup: `fp16_val = codebook_fp16[nibble] * norm_fp16`. The codebook is 16 entries of fp16 values (32 bytes total), loaded once into shared memory or constant memory at kernel launch.

2. **Accumulate using mma.sync.m16n8k16:**
   After dequant, K and V values are in fp16 registers in the layout expected by mma.sync. Use `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for QK^T accumulation and attention-weighted V accumulation. This replaces the current scalar FMA loop over K/V entries.

3. **Pipeline: cp_async -> smem -> dequant -> tensor cores:**
   - Stage 0: cp_async loads next tile of packed 4-bit KV from global memory to shared memory (double-buffered).
   - Stage 1: Warp threads cooperatively dequant the current smem tile to fp16 registers using warp-shuffle + bitwise ops.
   - Stage 2: Feed fp16 register fragments to mma.sync for QK^T and PV accumulation.
   - This three-stage pipeline overlaps global memory loads with dequant and compute.

4. **Lloyd-Max codebook constraint:**
   Lloyd-Max codebook has exactly 16 entries (4-bit quantization). At 2 bytes/entry (fp16), the full codebook is 32 bytes -- fits trivially in shared memory, constant memory, or even registers. Dequant is a 16-entry LUT lookup followed by a single fp16 multiply (by the per-group norm), which is negligible compared to the mma.sync it feeds.

5. **Key implementation details:**
   - Tile size: m16n8k16 tiles, processing 16 K-vectors per mma instruction.
   - Register pressure: dequant adds ~8 fp16 registers per warp for codebook + nibble extraction temporaries. Must verify no register spill (compile with `-Xptxas -v`).
   - Split-K: Retain existing split-K parallelism (HYP-013/HYP-018) for long sequences. Each split processes a contiguous chunk of KV, dequants locally, and reduces via the existing combine kernel.
   - Softmax: Online softmax (FlashAttention-style) remains in fp32 scalar registers between mma.sync tiles -- this is the standard pattern.

## Status: pending

## References

- HYP-030 seq_len sweep data (motivation: 12x slower than FlashInfer at seq=4096)
- Marlin (arXiv:2408.11743) -- dequant-to-fp16 register pattern for 4-bit weights
- BitDecoding (HPCA 2026) -- INT4 tensor core direct path (rejected in HYP-019 for rank-1 decode, but the dequant-to-fp16 intermediate step is viable)
- SageAttention2 (ICML 2025) -- mixed-precision attention with quantized KV, validates fp16 mma for attention
- FlashInfer's decode kernel -- uses mma.sync for QK/V accumulate, this is the target to match
- Phase 9b in docs/ROADMAP.md -- this is that item, now with profiling data from HYP-030 to justify it
- HYP-007 (tensor-core dequant pipeline) -- earlier exploration of this idea, before seq_len sweep data existed
- HYP-019 (INT4 tensor core) -- rejected direct INT4 path; this hypothesis uses fp16 intermediate instead
