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

### Phase 5: Fused Kernel Integration — DONE

Fused CUDA decode kernel running in vLLM. Qwen3-1.7B generates coherent text.

- [x] Store quantized bytes in separate HND tensors alongside vLLM cache
- [x] Call fused CUDA decode kernel from vLLM `forward()` for decode
- [x] Prefill fallback: eager dequant → FlashAttention
- [x] Fix 8 bugs (chunk size, dequant split, GQA dispatch, bound check, HEAD_DIM, output dims, rotation, layout)
- [x] 8/8 standalone kernel tests at cosine=1.0
- [ ] Benchmark: TPOT with fused kernel vs eager simulation
- [ ] Max batch size: measure VRAM savings

### Phase 6: Benchmarks and Optimization — DONE (partial)

#### 6a. Kernel benchmark — DONE

Standalone kernel comparison (seq_len=1024, Qwen3 config):

| Kernel | Latency | vs SDPA |
|--------|---------|---------|
| FP16 SDPA (FlashAttention) | 20.5 μs | 1.0× |
| TQ fused (bdz=1, 16 threads) | 856 μs | 41.6× slower |
| TQ fused (bdz=16, 256 threads) | 142 μs | 6.9× slower |

Memory: 3.76× compression confirmed (512 → 136 bytes/token/head).

#### 6a-opt. Step-by-step optimization — DONE

| Step | What | Result |
|------|------|--------|
| More threads (bdz sweep) | bdz 1→16 | 856→142 μs (**6× speedup**) |
| Pipeline analysis | Kernel is compute-bound | Double-buffer won't help |
| FWHT in kernel | Shuffle-based FWHT | Broken with multi-warp layout |
| bdz>1 merge | Cross-tz softmax merge | Broken (tile index interaction with GQA) |

**Conclusion: our standalone kernel proves correctness (cosine=1.0) but is 7-42× slower than FlashAttention. Closing this gap requires modifying FlashInfer's optimized decode kernel directly — replacing its `cast_load` KV path with a dequant path. This avoids reimplementing tensor cores, pipelining, and warp specialization from scratch.**

#### 6b-6d. Not started

Blocked on kernel performance. The fused kernel is too slow for meaningful serving benchmarks.

### Phase 7: Modify FlashInfer decode kernel — NOW

Follow FlashInfer's existing architecture. Add TurboQuant as a new KV dtype alongside FP8/FP16. Only change the KV loading path — everything else stays FlashInfer's optimized code.

Working tree: `~/workdir/flashinfer` on DGX Spark.

#### What FlashInfer already does for FP8:
1. `cp_async` loads FP8 bytes from VRAM → shared memory
2. `cast_load` converts FP8 → float in registers (hardware type cast)
3. QK dot product, softmax, V accumulate — all optimized with tensor cores

#### What we change for TurboQuant 4-bit:
1. Replace `cp_async` with: load 4-bit packed bytes → **codebook lookup** → write fp16 to smem
2. Keep `cast_load` (reads fp16 from smem → float registers, same as FP16 path)
3. Keep QK, softmax, V accumulate **completely unchanged**

#### Done:
- [x] `flashinfer_dequant_load.cuh`: dequant_load_to_smem (replaces cp_async per element)
- [x] `flashinfer_decode_turbo.cuh`: FlashInfer-style kernel with dequant load
- [x] Correctness: cosine=1.0 (head_dim=64/128, GQA, 1-64 tokens)
- [x] Benchmark: 1739 μs (84× vs SDPA) — correct but slow

#### Optimization steps (see `docs/reference/optimization-plan.md`):
- [x] **7a.** Fix bdz>1 merge → 373 μs (4.7× speedup, 18× vs SDPA)
- [x] **7b.** Precompute page offsets → skipped (net negative from smem pressure)
- [ ] **7c.** In-kernel FWHT → eliminate 203 μs Python overhead
- [ ] **7d.** Modify FlashInfer source directly → ~25-30 μs (production path)

Current: 373 μs kernel + 203 μs Python FWHT = 576 μs total decode.
Theoretical lower bound: ~18 μs (faster than SDPA for memory-bound decode).

## Next

- [ ] Outlier calibration pipeline — automatic outlier channel detection
- [ ] Multi-model support — validate across Llama, Mistral, Qwen architectures

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node tensor parallelism
