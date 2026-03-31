# TurboQuant Kernel Optimization Plan

## Current State

The fused decode kernel is **correct** (cosine=1.0 across all configs) but **84× slower** than SDPA.

| Kernel | Latency (1024 tok) | vs SDPA | Threads | Pipeline | Tensor Cores |
|--------|-------------------|---------|---------|----------|-------------|
| SDPA (FlashAttention) | 20.5 μs | 1.0× | 128-512 | cp_async double-buffer | Yes (FP16 MMA) |
| TQ standalone (bdz=16) | 142 μs | 6.9× | 256 | Single-buffer | No |
| TQ FlashInfer-style | 1739 μs | 84× | 16 | None | No |

## Performance Gap Analysis

### Why 84× slower (FlashInfer-style kernel)

| Factor | SDPA | TurboQuant | Gap contribution |
|--------|------|------------|-----------------|
| **Threads per block** | 128-512 | 16 (bdx=8, bdy=2, bdz=1) | ~32× |
| **Tensor cores** | FP16 WMMA (128 ops/cycle) | Scalar FMA (1 op/cycle) | ~4× |
| **Memory pipeline** | cp_async double-buffer (load overlaps compute) | Synchronous load + syncthreads | ~2× |
| **Page lookup** | Precomputed `kv_offset_smem` | Per-token `__ldg(indices + page_iter)` | ~1.5× |
| **Dequant overhead** | None (raw type cast) | Codebook lookup + multiply per element | ~1.3× |
| **Combined** | | | ~84× (product of factors) |

### Time Breakdown (from bench_breakdown.py, standalone bdz=16)

| Component | Time | % of total |
|-----------|------|-----------|
| Memory read (4-bit, 1088 KB) | ~65 μs | 46% |
| Compute (QK + softmax + V) | ~77 μs | 54% |
| **Total kernel** | **142 μs** | **100%** |
| Python FWHT (Q rotation + output un-rotation) | 203 μs | N/A (outside kernel) |

After thread optimization (bdz=16), kernel is **compute-bound**, not memory-bound.

## Optimization Roadmap (ordered by impact × feasibility)

### 1. Increase bdz (threads) — Expected: 6× speedup

**Impact: HIGH | Effort: MEDIUM**

The standalone bdz sweep showed 6× speedup (856→142 μs). The FlashInfer-style kernel needs the same optimization but with a working cross-tz merge.

**Blocker:** The cross-tz softmax merge was broken due to:
- Score array indexing interaction with bdy (GQA)
- Token-to-smem-row mapping inconsistency between load and QK functions

**Fix approach:**
1. Each tz processes `tile_size_per_bdx` tokens independently
2. QK loop reads smem rows at `(tz * bdy + ty) * tile_size_per_bdx + j` offset
3. After all tile iterations, merge tz results via shared memory using standard online softmax merge: `m_new = max(m_self, m_other)`, `d_new = d_self * exp(m_self - m_new) + d_other * exp(m_other - m_new)`
4. Test: verify cosine=1.0 with bdz=2 first, then scale to 16

**Expected result:** 1739 → ~290 μs (6× from threads alone)

### 2. Precompute page offsets — Expected: 1.5× speedup

**Impact: MEDIUM | Effort: LOW**

Current: every dequant_load call does `page_size.divmod()` + `__ldg(indices)` per token.
FlashInfer precomputes all offsets into `kv_offset_smem` once, then reuses.

**Fix:** Add the `kv_offset_smem` precomputation from FlashInfer's original decode kernel. Each thread computes its token's page offset once per bdx-group, stores in smem. Subsequent chunk iterations reuse the offset.

**Expected result:** 290 → ~190 μs

### 3. Move FWHT into kernel — Expected: eliminates 203 μs Python overhead

**Impact: HIGH | Effort: MEDIUM**

The Python FWHT for Q rotation + output un-rotation costs 203 μs — more than the kernel itself. Moving it into the kernel eliminates this entirely.

**Approach A (simpler):** Shared memory FWHT
- After all tz merge, write o_acc to smem (128 floats per head)
- Run in-place FWHT on smem using all bdx threads
- Write fp16 result to global memory
- Need: bdx threads cooperate on FWHT butterfly (8 threads × 8 elements = 64 dims per chunk)

**Approach B (register):** Fix the register-based FWHT
- The `fwht_register` function uses `__shfl_xor_sync` which should work within bdx lanes
- Bug was likely in the `(tx, ty, tz)` → lane mapping with bdz>1
- Fix: use `__shfl_xor_sync` with mask restricted to bdx-sized subwarp

**Expected result:** Eliminates 203 μs, total pipeline ~190 μs (kernel only)

### 4. cp_async pipelining — Expected: 1.5× speedup

**Impact: MEDIUM | Effort: HIGH**

Current: synchronous load (dequant) → syncthreads → compute → syncthreads.
FlashInfer uses `cp_async` with double-buffered pipeline.

**Challenge:** `cp_async` does raw byte copy — cannot dequant during copy. But we can:
1. `cp_async` the 4-bit packed bytes (4× less data than fp16)
2. After cp_async completes, dequant from smem to a second smem buffer
3. Compute reads from the dequanted buffer

This is a 3-stage pipeline: copy_packed → dequant_to_smem → compute.
The copy is async (overlaps with previous iteration's compute).

**Expected result:** 190 → ~130 μs

### 5. Tensor cores — Expected: 2-4× speedup on QK compute

**Impact: HIGH | Effort: VERY HIGH**

SDPA uses WMMA/MMA instructions for the QK dot product (128 FP16 ops per instruction). Our kernel uses scalar FMA (1 op per instruction).

**Challenge:** Decode attention has query_len=1, so QK is a [1, d] × [d, tile_size] product — not the typical matrix-matrix multiply. FlashAttention still uses tensor cores by organizing the tile as a matrix and using `mma.m16n8k16` instructions.

**Approach:** Restructure the QK accumulation to use `wmma::mma_sync`:
- Query: [1, 128] replicated across MMA fragments
- Key tile: [tile_size, 128] loaded as MMA B operand
- Requires MMA-compatible memory layout in smem

This is the most complex optimization and may require significant kernel restructuring.

**Expected result:** 130 → ~50-65 μs (2-2.5× from tensor cores)

### 6. Modify FlashInfer source directly — Expected: near-SDPA performance

**Impact: HIGHEST | Effort: HIGH**

Instead of a standalone kernel, modify FlashInfer's actual `decode.cuh`:
- Add a new `DTypeKV` variant for TurboQuant 4-bit
- Replace only the `cp_async` load path with our `dequant_load_to_smem`
- Keep ALL existing FlashInfer optimizations (tensor cores, pipeline, warp scheduling)

**This is the nuclear option** — maximum performance, minimum custom code. The dequant-load is ~20 lines of code; everything else is FlashInfer's production kernel.

**Expected result:** ~25-30 μs (within 1.5× of SDPA, memory-bound on 4× less data)

## Recommended Path

```
Phase 7a: Fix bdz merge (step 1)          → 1739 → ~290 μs
Phase 7b: Precompute page offsets (step 2) → ~290 → ~190 μs
Phase 7c: In-kernel FWHT (step 3)         → eliminates 203 μs Python
Phase 7d: Modify FlashInfer source (step 6) → ~25-30 μs (production path)
```

Steps 4-5 (pipeline, tensor cores) are subsumed by step 6 — FlashInfer already has them. The optimization effort should focus on getting the dequant-load function correct and efficient, then inject it into FlashInfer's production kernel.

## Theoretical Lower Bound

TurboQuant reads 3.76× less data than FP16. For a **memory-bound** decode kernel:
- SDPA reads 4096 KB at 1024 tokens: 20.5 μs
- TurboQuant reads 1088 KB: ~5.5 μs (theoretical minimum)

But there's compute overhead for dequant (~12 μs codebook lookup). So the theoretical lower bound is **~18 μs** — potentially **faster than SDPA** for long sequences where decode is memory-bound.

This is only achievable by modifying FlashInfer's source (step 6).
