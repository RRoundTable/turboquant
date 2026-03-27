# Roadmap

## Now

### Phase 1: Architecture and Memory Layout Design (C++ / CPU)

Define the C++/CUDA struct for 3.5-bit compressed data stored in VRAM. Verify bit-exact correctness and memory alignment without GPU.

- [x] Define `TurboQuantTile` struct (16 tokens × 64 dims, 480 bytes, 16-byte aligned)
- [x] Implement 4-bit nibble packing and 3-bit group-of-8 packing (GGML-compatible)
- [x] Implement signed quantization (MSB sign + magnitude index)
- [x] CPU mock tests: bit-exact roundtrip, pack/unpack, FP16 conversion
- [x] `static_assert` alignment check: sizeof must be multiple of 16 bytes (128-bit)
- [ ] GPU-native Python core (codebook, Hadamard, QJL) — done, needs integration with tile layout

**Verification gate:** Bit-exact C++ unit test + compile-time alignment check.

### Phase 2: Write Kernel (Prefill Quantization Fusion)

Kernel that takes model KV output, compresses it, writes to VRAM. Tested independently of Phase 3.

- [ ] Write `apply_rope_and_quantize_kv` kernel (RoPE + Hadamard rotation + scalar quantize + pack → VRAM write)
- [ ] PyTorch reference comparison: CUDA kernel output == PyTorch reference (bit-exact)
- [ ] NCU profiling: memory throughput > 80% of peak VRAM bandwidth, tensor cores idle (0%)

**Verification gate:** Zero bit-wise error vs PyTorch reference + write bandwidth profiling.

### Phase 3: Read Kernel (Decode Dequant + Attention Fusion)

FlashInfer integration. Load `TurboQuantTile` from VRAM, inline dequantize in SRAM/registers, compute attention. Tested with dummy data (not Phase 2 output) to isolate variables.

- [ ] Implement dequant + attention kernel (or L2 ping-pong pipeline)
- [ ] Dummy injection test: pre-compressed tiles → attention output within atol=1e-3 of `scaled_dot_product_attention`
- [ ] NCU profiling: zero register spilling (no local memory traffic). If spilling, reduce tile size or redistribute threads.

**Verification gate:** Accuracy vs SDPA reference + zero register spill.

### Phase 4: End-to-End System Integration

Integrate Phase 2 (write) + Phase 3 (read) into vLLM/SGLang backend via PagedAttention binding.

- [ ] Python/C++ binding with vLLM KV cache manager
- [ ] TTFT measurement: Phase 2 write overhead vs BF16 baseline
- [ ] TPOT measurement: decode speed improvement from Phase 3 read fusion (target: 3-4× faster)
- [ ] Max batch size: measure concurrent users before OOM (proves VRAM savings)
- [ ] Model quality: LongBench, WikiText perplexity, NIAH test (100k+ tokens)

**Verification gate:** Serving metrics (TTFT, TPOT, max batch) + task accuracy (PPL, NIAH).

## Next

- [ ] Outlier calibration pipeline — automatic outlier channel detection from calibration data
- [ ] Multi-model support — validate across Llama, Mistral, Qwen architectures

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node tensor parallelism
