# Architecture

## Overview

TurboQuant compresses LLM KV cache from FP16 to 4-bit (3.76× compression) with
near-zero quality loss. The system has two CUDA kernels (write + decode) and a
vLLM serving backend.

```
Prefill:                                    Decode:
  K,V [fp16] ──CUDA write kernel──►          Q [fp16] + KV cache [4-bit]
    normalize → codebook quantize →            ──CUDA decode kernel──►
    nibble pack → store 4-bit                  cp_async → inline dequant →
                                               QK dot → softmax → V accum →
  KV cache: [batch, heads, seq, 68B/tok]       output [fp16]
  (vs FP16: 256B/tok = 3.76× less)
```

## Key Results (A100, 5 models validated)

| Metric | Value |
|--------|-------|
| KV compression | 3.76× (4-bit + fp16 norms) |
| TPOT overhead (eager) | 0-2% across 5 models |
| TPOT overhead (CUDA graphs, seq=1024) | 2.5% |
| TTFT overhead (CUDA write kernel) | 3.7% |
| Quality (PPL) | 14.91 → 14.91 (0.01% loss) |
| Correctness | 100% exact token match (12 prompts × 2 models) |
| Throughput at max batch | ~3.6× gain |
| Decode kernel | 856μs → 37μs (23× speedup, 23 hypotheses) |

## Tech Stack

| Layer | Choice |
|-------|--------|
| Decode kernel | CUDA C++ (v4 contiguous + split-KV) |
| Write kernel | CUDA C++ (fused normalize → quantize → pack) |
| Algorithm | Python/PyTorch (codebook, Hadamard, quantizer) |
| Serving | vLLM v0.19.0 backend |
| FlashInfer | Headers only (math utils, cp_async, state_t) |
| Testing | pytest (Python), JIT compile (CUDA) |
| Profiling | nsys (timeline), torch.profiler (timing) |
| GPU targets | A100-SXM4-40GB (primary), DGX Spark GB10 (legacy) |
| Experiment infra | Forge cluster (notebooks + jobs) |

## Directory Structure

```
turboquant/
├── csrc/
│   ├── include/
│   │   ├── turboquant/
│   │   │   ├── page_turbo.cuh              # Paged KV cache struct + codebook constants
│   │   │   ├── flashinfer_dequant_load.cuh # Dequant-load helpers (cp_async + LUT)
│   │   │   └── quantize_write_kernel.cuh   # CUDA write kernel (normalize→quantize→pack)
│   │   ├── flashinfer_decode_turboquant_v2.cuh  # FlashInfer-integrated decode (Params struct)
│   │   ├── flashinfer_decode_turboquant_v3.cuh  # cp_async staged (rejected)
│   │   ├── flashinfer_decode_turboquant_v4.cuh  # Inline dequant, no fp16 smem (BEST paged)
│   │   ├── flashinfer_decode_turboquant_v4_contiguous.cuh  # Contiguous KV (BEST overall)
│   │   ├── flashinfer_decode_turboquant_v5_warpspec.cuh    # Warp specialized (rejected)
│   │   ├── flashinfer_decode_turboquant_v5_tc.cuh         # Tensor-core WMMA (HYP-031)
│   │   └── flashinfer_decode_turboquant_combine.cuh        # Split-KV combine kernel
│   └── src/
│       ├── decode_v4_binding.cu            # Paged v4 + split-KV binding
│       ├── decode_v4_contiguous_binding.cu # Contiguous v4 + split-KV binding
│       ├── decode_v5_tc_binding.cu         # Tensor-core v5 + split-KV binding (HYP-031)
│       └── quantize_write_binding.cu       # Write kernel binding
├── turboquant/                             # Python package
│   ├── quantizer.py                        # TurboQuantMSE, TurboQuantProd
│   ├── codebook.py                         # Lloyd-Max codebook (4-bit, 16 levels)
│   ├── hadamard.py                         # FWHT + RandomHadamardRotation
│   ├── kernel_config.py                    # Model presets + kernel dispatch config
│   ├── decode_kernel.py                    # v1 standalone kernel wrapper (legacy)
│   ├── decode_kernel_v4.py                 # v4 kernel wrapper + contiguous dispatch
│   ├── triton_decode.py                    # Triton port (for KernelAgent)
│   ├── triton_decode_ref.py                # PyTorch reference for Triton
│   └── write_kernel.py                     # Python write kernel reference
├── vllm_backend_fused.py                   # vLLM attention backend (contiguous v4)
├── tests/
│   ├── test_v4_correctness.cu              # v4 paged kernel tests
│   ├── test_v4_contiguous.cu               # v4 contiguous + split-KV + fused combine
│   ├── test_triton_decode.py               # Triton kernel correctness
│   ├── bench_contiguous.py                 # Contiguous + split-KV benchmark
│   ├── bench_write_kernel.py               # Write kernel benchmark
│   ├── bench_int4_tc.py                    # INT4 tensor core experiment
│   └── test_v5_tc.py                       # v5 tensor-core kernel test (HYP-031)
├── docs/
│   ├── GOAL.md                             # Project goal + success criteria
│   ├── ROADMAP.md                          # Phase 1-9 status
│   ├── ARCHITECTURE.md                     # This file
│   ├── SPEC.md                             # Behavioral spec
│   ├── hypotheses/                         # 23 experiment records (HYP-001 to HYP-023)
│   └── reference/                          # Benchmarks, comparisons, analysis docs
└── CLAUDE.md                               # Dev workflow + profiling guide
```

## Quantization Algorithm

```
Write (prefill):
  1. L2 normalize: norm = ||kv||₂, x̂ = kv / norm
  2. (Optional) Hadamard rotate: x̃ = signs ⊙ FWHT(x̂)  →  ~N(0, 1/d)
  3. Lloyd-Max quantize: 16-level codebook for N(0,1), find nearest centroid
  4. Nibble pack: 2 × 4-bit indices per byte (hi << 4 | lo)
  5. Store: packed bytes + fp16 norm per token per 64-dim chunk

Read (decode):
  1. Load packed bytes from KV cache (cp_async or global load)
  2. Unpack nibbles: hi = (byte >> 4) & 0xF, lo = byte & 0xF
  3. Codebook lookup: val = codebook[index] × codebook_scale × norm
  4. Use in attention: QK dot product, softmax, V accumulate
```

Codebook (constant memory, 16 entries):
```
[-2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
  0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733]
```

Per token per head: 32 packed bytes + 2 norm bytes = 34 bytes per 64-dim chunk.
For head_dim=128: 68 bytes (vs FP16: 256 bytes = **3.76× compression**).

## Decode Kernel Architecture (v4 contiguous)

The production kernel. No fp16 smem buffer — dequant happens inline during compute.

```
Grid: (batch × kv_heads × num_splits,)  — adaptive split-KV
Block: (bdx=16, bdy=GQA_ratio, bdz=16)  — 256 threads

Per tile iteration (tile_tokens = bdx × bdy × bdz / bdx × tile_per_bdx):
  Phase 1: cp_async packed K bytes → staging smem (HW DMA)
           + precompute K norms → smem (overlapped with cp_async)
           + wait + sync
  Phase 2: QK with inline dequant
           - Each thread reads packed bytes from staging
           - Codebook lookup → float (no fp16 intermediate)
           - Dot product with Q, warp shuffle reduce
           - Online softmax update
           + sync
  Phase 3: cp_async packed V bytes → staging + precompute V norms
           + wait + sync
  Phase 4: V accumulate with inline dequant
           + sync

Cross-warp merge: sync_state (FlashInfer's online softmax merge)
Output: fp16 (normal) or float (split-KV partial)
```

**Shared memory: ~7 KB** (staging + norms). vs v2's ~32 KB (fp16 K+V buffers).

## Split-KV (FlashDecoding)

For long sequences, partition KV across multiple blocks per (batch, kv_head):

```
seq=1024, 8 splits: each block handles 128 tokens
Grid: 1 × 8 × 8 = 64 blocks (vs 8 without split)
Combine: separate kernel merges partial results via online softmax

Adaptive policy:
  seq ≤ 256: nosplit (TQ faster than FlashInfer!)
  seq 512-1024: 4-8 splits
  seq 2048+: 16-24 splits
  CUDA graph at seq ≥ 512: 26% kernel speedup
```

## Write Kernel

Fused CUDA kernel for prefill KV quantization:

```
Grid: (num_tokens, num_heads)
Block: 32 threads (single warp, no syncthreads)

Per (token, head):
  1. L2 norm via warp shuffle reduction
  2. Normalize
  3. Binary search on 15 boundaries → 4-bit index [0..15]
  4. Nibble pack: (idx_even << 4) | idx_odd
  5. Store packed bytes + fp16 norm
```

**41 μs per layer** (vs Python 424 μs = 10× faster, vs memcpy 21 μs = 2× overhead).

## Key Design Decisions (from 23 experiments)

| Decision | Rationale | Evidence |
|----------|-----------|---------|
| **Inline dequant (no fp16 smem)** | 4× less smem traffic, no half↔float conversion | HYP-006: 22-33% faster than v2 |
| **Contiguous KV layout** | Eliminates 32μs paging overhead (divmod + indirect load) | HYP-017: beats FlashInfer at seq≤256 |
| **Split-KV for long seq** | Distributes work across SMs (8 blocks → 64+) | HYP-018: flat 47μs at seq=1024-4096 |
| **bdz=16** | Occupancy is #1 bottleneck, not compute | HYP-008: 3.3× speedup |
| **CUDA graph capture** | 26% kernel speedup from eliminating launch overhead | HYP-023: 2.5% TPOT overhead |
| **Scalar FMA (no tensor cores)** | Rank-1 decode underutilizes M16 MMA tile | HYP-007a, HYP-019: TC slower at bdy≤2 |
| **Separate combine kernel** | __threadfence in fused version costs more than kernel launch | HYP-022: fused 8% slower |
| **No pipelining** | Compute:load ratio is 10:1, nothing to overlap | HYP-005, HYP-020: both rejected |

## Model Support

Generalized dispatch supports any (head_dim, GQA ratio) via `kernel_config.py`:

| Model | QO/KV | GQA | head_dim | Validated |
|-------|-------|-----|----------|-----------|
| Qwen3-0.6B | 16/8 | 2:1 | 64 | Correctness + TPOT |
| Qwen3-1.7B | 16/8 | 2:1 | 128 | Full E2E + PPL |
| Qwen3-4B | 32/8 | 4:1 | 80 | Correctness + TPOT |
| Qwen3-8B | 32/8 | 4:1 | 128 | Full E2E + PPL |
| Mistral-7B | 32/8 | 4:1 | 128 | Correctness + TPOT |
| Llama-2-7B | 32/32 | 1:1 | 128 | Kernel only |
| Llama-3-8B | 32/8 | 4:1 | 128 | Kernel only |
| Llama-3-70B | 64/8 | 8:1 | 128 | Kernel only |

## Constraints

- GPU only — no CPU device support
- head_dim must be multiple of 64 (padded if not)
- GQA ratio must be integer (bdy = num_qo_heads / num_kv_heads)
- Lloyd-Max codebook assumes Hadamard-rotated vectors follow ~N(0,1/d)
- Contiguous layout requires pre-allocated max_seq (no dynamic paging)
- CUDA graph capture requires fixed seq_len per graph (re-capture on change)
