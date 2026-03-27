# Roadmap

## Now

### Phase 1: Architecture and Memory Layout Design (C++ / CPU) — DONE

- [x] TurboQuantTile struct (16 tokens × 64 dims, now 544 bytes at uniform 4-bit)
- [x] 4-bit nibble packing, Lloyd-Max codebook
- [x] CPU mock tests, static_assert alignment
- [x] GPU-native Python core (codebook, Hadamard, QJL)

### Phase 2: Write Kernel — DONE

- [x] CUDA write kernel: L2 normalize → codebook quantize → bit-pack → VRAM write
- [x] Bit-exact vs C++ CPU reference
- [x] PyTorch write kernel reference (`write_kernel.py`)

Note: Hadamard rotation and RoPE not fused into CUDA kernel yet (done in Python simulation).

### Phase 3: Fused Decode Kernel — DONE (standalone, NOT integrated into vLLM)

Built a standalone CUDA decode kernel that fuses dequantization into the attention loop.
Uses FlashInfer headers for math/types but does NOT modify FlashInfer source code.

#### 3a. Standalone kernel — DONE
- [x] `paged_kv_turbo_t` struct: quantized KV + norms + page table (`csrc/include/turboquant/page_turbo.cuh`)
- [x] `decode_turboquant.cuh`: standalone attention kernel with inline dequant (`csrc/include/turboquant/decode_turboquant.cuh`)
- [x] Dummy injection test: cosine=1.000000 vs CPU reference

#### 3b. Profiling — DONE
- [x] No register spilling (50 regs, 0 local memory)
- [x] Parallel dequant: 1.72× speedup, 1.02 tokens/μs

#### 3c. Python JIT wrapper — DONE
- [x] `TurboQuantDecoder` via torch.utils.cpp_extension JIT compile (`turboquant/decode_kernel.py`)
- [x] cosine=1.0 from Python

**NOT done:** FlashInfer was not modified. The fused kernel is standalone, not integrated into FlashInfer's JIT system or vLLM's attention dispatch.

### Phase 4: vLLM Integration — IN PROGRESS

#### What works:
- [x] `TurboQuantBackend` registered in vLLM (FlashAttention subclass)
- [x] `attention_backend="TURBOQUANT"` selects our backend
- [x] Python quantize-dequant simulation in `do_kv_cache_update`
- [x] Qwen3-1.7B generates correct text (7/8 factual accuracy matches baseline)
- [x] TTFT/TPOT benchmarked: 1.84× overhead from Python simulation

#### What does NOT work:
- [ ] **Kernel fusion not applied** — vLLM uses Python quantize-dequant + FlashAttention, not the fused CUDA kernel
- [ ] **No memory savings** — KV cache stores fp16 (same size as baseline), quantize-dequant is a simulation pass
- [ ] **No compressed cache allocator** — vLLM's allocator needs modification to allocate smaller int8 buffers
- [ ] Max batch size measurement
- [ ] Perplexity eval (WikiText, LongBench, NIAH)

#### To achieve real kernel fusion + memory savings:
1. Modify vLLM's `_allocate_kv_cache` to allocate compressed int8 buffers
2. Override `get_kv_cache_shape` to return compressed dimensions
3. Replace `do_kv_cache_update` with CUDA write kernel (quantize → store packed bytes)
4. Replace `forward` with fused CUDA decode kernel (read packed bytes → dequant → attention)
5. Benchmark actual VRAM reduction and TPOT improvement

### Phase 5: Fused Kernel Integration — NOW

Wire the standalone fused CUDA decode kernel into vLLM's attention backend. Store quantized bytes in paged cache, use fused kernel for decode attention.

- [ ] Store quantized bytes (not fp16) in vLLM's paged KV cache
- [ ] Call fused CUDA decode kernel from vLLM's `forward()` for decode
- [ ] Prefill fallback: eager dequant → FlashAttention
- [ ] Benchmark: TPOT with fused kernel vs eager simulation
- [ ] Max batch size: measure VRAM savings from compressed cache

## Next

- [ ] Outlier calibration pipeline — automatic outlier channel detection
- [ ] Multi-model support — validate across Llama, Mistral, Qwen architectures

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node tensor parallelism
