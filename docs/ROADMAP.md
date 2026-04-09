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
- [x] **7d.** Inject dequant into FlashInfer decode — cosine=1.0, 9/9 configs pass (A100)
  - Uses FlashInfer's compute_qk, update_local_state, sync_state directly
  - head_dim={64,128}, GQA={1:1,2:1,4:1}, batch={1,2}, seq_len={16..256}
  - Tested on Forge A100-SXM4-40GB
- [x] **7e.** Benchmark v2 kernel vs SDPA on A100

v2 benchmark results (Qwen3 config: 12 heads, head_dim=128, batch=1):

| seq_len | SDPA (μs) | TQ v2 (μs) | Ratio |
|---------|-----------|------------|-------|
| 128     | 22        | 59         | 2.7×  |
| 256     | 31        | 101        | 3.3×  |
| 512     | 30        | 185        | 6.2×  |
| 1024    | 31        | 351        | 11.4× |
| 2048    | 30        | 635        | 21.1× |
| 4096    | 34        | 1352       | 39.8× |

**Analysis:** SDPA uses cp_async pipelining (load/compute overlap). Our v2 kernel has
zero pipelining — dequant is synchronous, each tile blocks until load completes. The
compute path (compute_qk/update_local_state) is FlashInfer's optimized code, but the
load path kills performance because it serializes memory and compute.

**Root cause:** `cp_async` is a HW DMA that overlaps with compute. Our dequant-load
requires ALU work (codebook lookup) during the load, so it can't use `cp_async`.
The entire tile load → sync → compute → sync pattern is serial.

#### Software pipelining (closing the gap with SDPA):
- [x] **7f.** cp_async staged pipeline — **net negative** (18% slower than v2).
  cp_async packed bytes to smem staging, dequant from staging to fp16. Correct (cos=1.0)
  but extra syncs + staging overhead > overlap benefit. Root cause: compute phases are
  too short (0.5μs) to hide VRAM load (1.5μs). cp_async only helps when compute ≈ load.
- [x] **7g.** Fused inline dequant (v4) — **22-33% faster than v2**
  Eliminated fp16 smem buffer, dequant inline during QK/V compute.
  cp_async packed bytes → staging, precompute norms → smem, inline dequant to float.
  No FlashInfer function reuse (custom QK/V loops). ~7× less smem than v2.
  Results (12 heads, hd=128, batch=1):

  | seq | SDPA | v2 | v4 | v4/v2 |
  |-----|------|----|----|-------|
  | 512 | 53μs | 206μs | 159μs | 0.78× |
  | 1024 | 60μs | 415μs | 296μs | 0.71× |
  | 2048 | 67μs | 755μs | 503μs | 0.67× |

  Still 3-7× slower than SDPA at bdz=4. Remaining gap: occupancy + page table overhead.
- [x] **7h.** Increase bdz to 16 — **3.3× speedup** (HYP-008 confirmed)
  v4 at bdz=16: 89μs at seq=1024 (was 296μs at bdz=4). 256 threads = 8 warps.
  Correctness: cos=1.0, 6/6 configs. Faster than SDPA at short sequences.

#### Contiguous + split-KV optimization:
- [x] **7i.** Contiguous KV layout (HYP-017) — **beats SDPA at seq≤256**
  No paging overhead: 16μs at seq=128 (vs SDPA 22μs), 24μs at seq=256 (vs SDPA 30μs)
- [x] **7j.** Contiguous + split-KV combined (HYP-018) — **flat 48μs at seq=128-1024**
  Adaptive: nosplit at seq≤256 (22-32μs), split at seq≥512 (48-59μs)

#### Current best (A100, Qwen3-1.7B, batch=1):

| seq | Best TQ | Config | vs SDPA | Memory |
|-----|---------|--------|---------|--------|
| 128 | **22 μs** | contiguous nosplit | **1.0× (matches SDPA)** | 3.8× less |
| 256 | **32 μs** | contiguous nosplit | 1.05× | 3.8× less |
| 512 | **48 μs** | contiguous split-4 | 1.6× | 3.8× less |
| 1024 | **48 μs** | contiguous split-8 | 1.6× | 3.8× less |
| 2048 | **59 μs** | contiguous split-16 | 2.0× | 3.8× less |

Kernel evolution: 856μs → 37μs (CUDA graph) = **23× total speedup** (23 hypotheses tested).

#### Hypothesis record (23 total, 10 confirmed, 13 rejected):
See `docs/hypotheses/` for all experiment records.

### Phase 8: Evaluation and Integration — DONE

- [x] **8a.** Perplexity — **0.01% degradation** (14.91 → 14.91 PPL on WikiText-2)
- [x] **8b.** Memory — **3.76× compression**, 3.8× more concurrent requests
- [x] **8c.** vLLM E2E with v4 contiguous+split-KV kernel — backend updated
- [x] **8d.** Multi-model — **6/6 models pass** on A100
- [x] **8e.** Max batch — **3.8× more requests** (71→268 at seq=4K on A100-40GB)
- [x] **8f.** Throughput — **TQ beats SDPA at batch≥64** (1.1-1.2× higher tok/s)
- [x] **8g.** Correctness — **100% exact token match** (12/12 prompts, Qwen3-1.7B + 8B)
- [x] **8h.** FlashInfer comparison — **TQ beats FlashInfer at seq≤256** (0.68× faster)
- [x] **8i.** CUDA write kernel (HYP-021) — **prefill TTFT overhead: 3.7%** (was 44%)
- [x] **8j.** CUDA graph capture (HYP-023) — **decode overhead: 2.5%** (was 23% eager)
- [x] **8k.** Fused combine (HYP-022) — rejected (7-8% slower, __threadfence overhead)

#### Complete E2E Performance (A100, Qwen3-1.7B, CUDA graphs, batch=1)

| Phase | FP16 baseline | TQ 4-bit | Overhead |
|-------|--------------|----------|----------|
| Prefill write (2K tok) | 1.2 ms (memcpy) | 3.1 ms (CUDA quantize) | **+3.7% of TTFT** |
| Decode TPOT (seq≤256) | 1.2 ms | **0.81 ms (eager)** | **33% faster** |
| Decode TPOT (seq=1024) | 1.2 ms | **1.23 ms (CUDA graph)** | **2.5% slower** |
| Memory | 1.0× | **0.27×** | **3.76× less** |
| **Max batch throughput** | 1.0× | **~3.6×** | **3.6× gain** |

Adaptive dispatch: eager at seq≤256 (TQ faster), CUDA graph at seq≥512 (26% kernel speedup).

#### Correctness

| Test | Result |
|------|--------|
| Kernel cosine (all configs) | 1.000000 |
| WikiText-2 PPL | 14.91 → 14.91 (0.01% loss) |
| Exact token match (12 prompts) | 100% on Qwen3-1.7B and 8B |
| Factual accuracy | FP16 = TQ on every prompt |

### Architecture gap: TurboQuant vs FlashInfer (baseline)

**Current: 1.6× at seq=1024 (48μs vs 30μs). Origin of the gap:**

```
FlashInfer FP16 decode pipeline:
  VRAM [fp16] ──cp_async──► SMEM [fp16] ──cast_load──► Regs [float]
       └── pipelined: load N+1 overlaps compute N (2-3 stages) ──┘
  Compute: tensor core mma.sync (312 TFLOPS)
  Grid: batch × kv_heads × num_splits (fills all SMs)

TurboQuant v4 contiguous+split pipeline:
  VRAM [4-bit] ──cp_async──► SMEM staging [uint8] ──dequant──► Regs [float]
       └── NO overlap: dequant is ALU, can't pipeline with compute ──┘
  Compute: scalar FMA + warp shuffle (20 TFLOPS)
  Grid: batch × kv_heads × num_splits (same structure)
```

| Aspect | FlashInfer | TurboQuant | Gap factor |
|--------|-----------|------------|-----------|
| QK/V compute | Tensor core (312 TFLOPS) | Scalar FMA (20 TFLOPS) | **~1.5× on QK phase** |
| Pipelining | 2-3 stages overlap | Single-stage serial | **~1.2× (4 syncs vs 2)** |
| Data per token | 512 bytes (fp16) | 136 bytes (4-bit) | **0.27× (TQ wins)** |
| Grid parallelism | Same split-KV | Same split-KV | 1.0× |
| Smem per block | ~32 KB | ~7 KB | **0.22× (TQ wins)** |

**The 1.6× gap decomposition:**
- 40%: scalar FMA vs tensor cores (irreducible without INT4 TC or BitDecoding)
- 30%: rank-2 QK underutilizes M16 MMA dimension at bdy=2
- 15%: sequential dequant ALU (codebook lookup per element)
- 10%: SM fill (64 blocks / 108 SMs at batch=1)
- 5%: extra syncs + instruction overhead

### Phase 9: Closing the gap — kernel architecture improvements

#### 9a. INT4 tensor core matmul (BitDecoding pattern)
Feed packed 4-bit data directly to `mma.sync.m16n8k64.s32.s4.s4.s32`.
No dequant step — apply codebook scaling post-matmul.
Requires: uniform quantization (not Lloyd-Max), Q quantized to INT4.
**Expected: 2-3× on QK/V compute → close to FlashInfer at seq≥512.**
Ref: BitDecoding (HPCA 2026), SageAttention2 (ICML 2025).

#### 9b. Dequant-to-fp16 + tensor core (Marlin pattern)
Dequant 4-bit → fp16 in registers via bitwise LUT, feed to fp16 tensor cores.
Compatible with Lloyd-Max codebook (non-uniform). Pipeline: dequant on CUDA cores
while tensor cores compute previous tile.
**Expected: 1.5-2× on compute, keeps codebook quality.**
Ref: Marlin (arXiv:2408.11743), BitDecoding warp-layout-aware dequant.

#### 9c. Persistent kernel with warp specialization
Dedicate warp groups: load warps (dequant + cp_async) vs compute warps (QK/V).
Eliminates block.sync() between load and compute phases.
Requires named barriers (SM80+) for producer-consumer handoff.
**Expected: 1.2× from reduced sync overhead.**

#### 9d. Batch-level SM saturation
At batch≥4, grid = 4 × 8 × 8 = 256 blocks → all SMs busy.
Single-request latency can't improve, but throughput scales.
Already measured (8f): TQ beats SDPA at batch≥64 (1.1-1.2× higher tok/s).

### Phase 9 results:
- [x] **9a.** INT4 tensor cores (HYP-019) — **rejected** (15× slower at rank-1 decode)
- [x] **9c.** Warp specialization (HYP-020) — **rejected** (0% improvement, compute:load=10:1)
- [x] **9e.** Fused combine (HYP-022) — **rejected** (7-8% slower, __threadfence overhead)
- [x] **9f.** CUDA graph capture (HYP-023) — **confirmed** (26% kernel speedup at seq=1024)

### Phase 10: Deployment — IN PROGRESS

- [x] **10a.** vLLM plugin system — entry_points registration (`turboquant/vllm_plugin.py`)
- [x] **10b.** Dockerfile — nvidia/cuda + vLLM + FlashInfer + TurboQuant
- [x] **10c.** ECR setup — `847366387031.dkr.ecr.us-east-1.amazonaws.com/vllm-turboquant`
- [ ] **10d.** Forge image build — building (`forge image build --context .`)
- [ ] **10e.** Tensor parallel support — validate kernel at TP=2,4,8
- [ ] **10f.** E2E serving test — vLLM serve with `--kv-cache-dtype turboquant`

### Phase 11: Tensor Parallelism — NOW

TP splits KV heads across GPUs. Each GPU runs TurboQuant on its local head subset.

```
Example: Qwen3-8B (32QO/8KV) at TP=4
  GPU0: 8QO/2KV heads → bdy=4, grid=1×2×splits
  GPU1: 8QO/2KV heads → same
  GPU2: 8QO/2KV heads → same
  GPU3: 8QO/2KV heads → same
```

**Key issues to validate:**
- [ ] **11a.** Kernel correctness at low KV head counts (1-2 heads per GPU)
  - At TP=8, each GPU has 1 KV head → grid=1×1×splits (very few blocks)
  - Split-KV becomes critical for SM utilization
  - Test configs: num_kv_heads ∈ {1, 2, 4} with various GQA ratios

- [ ] **11b.** TP-aware KV cache allocation in vLLM backend
  - vLLM passes `num_kv_heads // tp_size` to each GPU's backend
  - Contiguous layout: `[batch, local_kv_heads, max_seq, bytes]`
  - Verify `kernel_config.py` dispatch handles local head counts

- [ ] **11c.** Multi-GPU benchmark
  - Compare TP=1 vs TP=2 vs TP=4 on Qwen3-8B / Mistral-7B
  - Measure: TPOT per TP config, throughput scaling
  - Verify: KV compression ratio stays 3.76× per GPU

- [ ] **11d.** Split-KV tuning for TP
  - At TP=4 with 2 KV heads: grid=2 blocks without split
  - Need aggressive splitting to fill SMs
  - Optimal: num_splits = max(SM_count / (batch × local_kv_heads), 1)

## Next

- [ ] **3-bit quantization** — GOAL.md target: ≥5× compression with <1% PPL
- [ ] **FWHT in write kernel** — fuse Hadamard rotation into CUDA quantize kernel
- [ ] **LongBench / NIAH** — quality evaluation at long contexts (4K-32K)
- [ ] **Tensor cores for GQA≥4** — WMMA gives 1.9× at bdy=4 (Llama-3, Mistral)

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node distributed (TP across nodes via NCCL)
- [ ] Continuous batching with dynamic KV cache growth
