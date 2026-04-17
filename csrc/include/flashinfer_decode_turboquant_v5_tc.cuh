#pragma once
// TurboQuant decode v5: tensor-core variant (HYP-031).
//
// Replaces v4's scalar FMA QK/V dot products with WMMA tensor core operations.
// Pipeline:
//   1. cp_async packed 4-bit bytes to smem staging (same as v4)
//   2. Dequant nibbles to fp16 via codebook LUT, write to smem fp16 buffer
//   3. wmma::load_matrix_sync / wmma::mma_sync for QK dot product
//   4. Online softmax (scalar, in fp32 registers)
//   5. Scalar V accumulate (WMMA for V is future work)
//
// Target config: head_dim=128, bdy=4 (Qwen3-8B: 32qo/8kv).
// At bdy=4, the M dimension of the 16x16 MMA tile is 25% utilized.
// HYP-007a showed 1.9x speedup at bdy=4 for QK compute alone.
//
// Thread block layout:
//   (32, 1, num_warps)
//   - Each warp is 32 threads (required by WMMA)
//   - num_warps (bdz) controls parallelism over KV tokens
//   - bdy (GQA ratio) is a template param, not a thread dim
//   - Each warp processes ALL bdy query heads via the M dimension of WMMA
//
// WMMA tiles: m16n16k16 for fp16 -> fp32 accumulate.
//   A (Q fragment): [16, 16] row-major -- rows 0..bdy-1 are valid queries
//   B (K fragment): [16, 16] col-major -- 16 KV tokens
//   C (scores):     [16, 16] row-major -- QK^T results
//
// For head_dim=128: accumulate over 128/16 = 8 WMMA k-tiles.
// For tile_n=16: process 16 KV tokens per WMMA pass.

#include <cuda_fp16.h>
#include <mma.h>
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/cp_async.cuh"
#include "turboquant/page_turbo.cuh"
#include "turboquant/flashinfer_dequant_load.cuh"
#include "flashinfer_decode_turboquant_v4_contiguous.cuh"  // ContiguousTurboQuantDecodeParams

namespace flashinfer {

// Number of KV tokens processed per WMMA tile (N dimension)
static constexpr uint32_t V5_TILE_N = 16;
static constexpr uint32_t V5_WMMA_M = 16;
static constexpr uint32_t V5_WMMA_N = 16;
static constexpr uint32_t V5_WMMA_K = 16;

// --------------------------------------------------------------------------
// Cooperative dequant: packed 4-bit bytes in staging smem -> fp16 in kv_smem.
// 32 threads in a warp each handle total_bytes/32 packed bytes.
// For head_dim=128: 64 bytes/token, 2 bytes/thread = 4 fp16 outputs/thread.
//
// HYP-032: codebook is pre-loaded into a per-lane register (cb_reg). Lane i
// holds kCodebook4bit[i] for i in [0, 16); lanes 16..31 hold 0 (never read).
// __shfl_sync fetches the target entry in 1 cycle instead of 4-10 cycles via
// constant memory. `__shfl_sync` with idx in [0, 16) always hits a valid
// source lane, so no mask gymnastics needed.
// --------------------------------------------------------------------------
template <uint32_t head_dim>
__device__ __forceinline__ void dequant_row_to_fp16_v5(
    __half* kv_fp16_row,             // output: [head_dim] fp16 in smem
    const uint8_t* staging_row,      // input: packed bytes for this token in smem
    const float* smem_norms_row,     // input: [dim_chunks] norm*codebook_scale values
    uint32_t tx,                     // lane id 0..31
    float cb_reg                     // HYP-032: per-lane codebook entry
) {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t bytes_per_chunk = 32;  // 64 dims x 4 bits / 8
    constexpr uint32_t total_bytes = dim_chunks * bytes_per_chunk;
    constexpr uint32_t bytes_per_thread = total_bytes / 32;

    #pragma unroll
    for (uint32_t b = 0; b < bytes_per_thread; b++) {
        uint32_t byte_idx = tx * bytes_per_thread + b;
        uint32_t chunk = byte_idx / bytes_per_chunk;
        uint32_t byte_in_chunk = byte_idx % bytes_per_chunk;
        uint32_t dim_base = chunk * 64 + byte_in_chunk * 2;

        float ns = smem_norms_row[chunk];
        uint8_t packed = staging_row[byte_idx];

        uint32_t hi_idx = (packed >> 4) & 0x0F;
        uint32_t lo_idx = packed & 0x0F;

        float hi_val = __shfl_sync(0xFFFFFFFF, cb_reg, hi_idx) * ns;
        float lo_val = __shfl_sync(0xFFFFFFFF, cb_reg, lo_idx) * ns;

        kv_fp16_row[dim_base]     = __float2half(hi_val);
        kv_fp16_row[dim_base + 1] = __float2half(lo_val);
    }
}

// --------------------------------------------------------------------------
// v5 tensor-core contiguous decode kernel
//
// Thread block: (32, 1, num_warps)
// Each warp independently processes a range of KV tokens using WMMA for QK.
// Cross-warp merge at the end via shared memory online softmax.
//
// WMMA QK: C[16,16] = sum_kt A[16,16] * B[16,16]
//   A = Q matrix: rows 0..bdy-1 are query heads, rest zero-padded to 16
//   B = K^T matrix: 16 KV tokens, k=16 elements at a time
//   C = QK^T scores: [16, 16], rows are query heads, cols are KV tokens
// --------------------------------------------------------------------------
template <uint32_t tile_n,          // KV tokens per WMMA tile = 16
          uint32_t head_dim,        // must be multiple of 16
          uint32_t bdy,             // GQA ratio (template, not thread dim)
          uint32_t num_warps,       // number of parallel warps over KV
          bool paged,               // HYP-035: true = walk page table per tile
          typename AttentionVariant, typename Params>
__device__ __inline__ void TurboQuantContiguousDecodeDeviceV5TC(
    const Params& params,
    uint8_t smem_raw[],
    const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y
) {
    const uint32_t tx = threadIdx.x;  // lane id 0..31
    const uint32_t warp_id = threadIdx.z;  // which warp 0..num_warps-1
    using IdType = typename Params::IdType;

    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;
    constexpr uint32_t k_tiles = head_dim / V5_WMMA_K;

    // Block/thread mapping
    const uint32_t num_qo_heads = params.num_qo_heads;
    const uint32_t batch_idx = params.partition_kv ? params.request_indices[bx] : bx;
    const uint32_t kv_tile_idx = params.partition_kv ? params.kv_tile_indices[bx] : 0;
    const uint32_t kv_head_idx = by;
    const uint32_t seq_len = params.seq_len;

    if (params.block_valid_mask && !params.block_valid_mask[bx]) return;

    const uint32_t kv_chunk_size_val = params.partition_kv ? *(params.kv_chunk_size_ptr) : seq_len;
    // HYP-035 paged mode: the kernel reads page table indices directly, so we
    // must clamp to the real per-batch seq_len (no zero-padded workspace to
    // absorb out-of-range reads). In contiguous mode, workspace is pre-zeroed
    // beyond seq_lens[b], so clamping to `seq_len` = max_len is still correct.
    const uint32_t effective_seq_len = paged
        ? static_cast<uint32_t>(params.seq_lens_ptr[batch_idx])
        : seq_len;
    const uint32_t chunk_start = params.partition_kv ? kv_tile_idx * kv_chunk_size_val : 0;
    const uint32_t chunk_end = params.partition_kv
        ? min((kv_tile_idx + 1) * kv_chunk_size_val, effective_seq_len)
        : effective_seq_len;
    const uint32_t chunk_size = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0;

    float codebook_scale = rsqrtf(static_cast<float>(params.padded_dim));

    // HYP-032: register-resident codebook. Lane i holds kCodebook4bit[i] for
    // i in [0, 16); lanes 16..31 hold 0 (never read — nibbles are 4-bit).
    // Replaces per-nibble constant-memory lookups with 1-cycle warp shuffles.
    const float cb_reg = (tx < 16) ? turboquant::kCodebook4bit[tx] : 0.0f;

    size_t quant_base = (size_t)batch_idx * params.quant_stride_batch
                      + (size_t)kv_head_idx * params.quant_stride_head;
    size_t norm_base  = (size_t)batch_idx * params.norm_stride_batch
                      + (size_t)kv_head_idx * params.norm_stride_head;

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    // Section 1: Q matrix [16, head_dim] fp16 (shared across all warps)
    //   = 16 * 128 * 2 = 4096 bytes
    //
    // Section 2: Per-warp KV buffer [tile_n=16, head_dim] fp16
    //   = 16 * 128 * 2 = 4096 bytes per warp
    //
    // Section 3: Per-warp staging [tile_n, bytes_per_row] uint8
    //   = 16 * 64 = 1024 bytes per warp (aligned to 16)
    //
    // Section 4: Per-warp norms [tile_n, dim_chunks] float
    //   = 16 * 2 * 4 = 128 bytes per warp
    //
    // Section 5: Per-warp scores [16, 16] float (for WMMA store)
    //   = 16 * 16 * 4 = 1024 bytes per warp
    //
    // Section 6: Cross-warp merge buffer [num_warps, bdy, head_dim+2] float
    //   (reuses sections 2-5 after main loop)

    constexpr uint32_t q_smem_bytes = V5_WMMA_M * head_dim * sizeof(__half);
    constexpr uint32_t kv_fp16_bytes = tile_n * head_dim * sizeof(__half);
    constexpr uint32_t staging_bytes = ((tile_n * bytes_per_row + 15) / 16) * 16;
    constexpr uint32_t norms_bytes = tile_n * dim_chunks * sizeof(float);
    constexpr uint32_t scores_bytes = V5_WMMA_M * V5_WMMA_N * sizeof(float);
    constexpr uint32_t per_warp_bytes = kv_fp16_bytes + staging_bytes + norms_bytes + scores_bytes;

    constexpr uint32_t main_loop_bytes = q_smem_bytes + num_warps * per_warp_bytes;
    constexpr uint32_t merge_bytes = num_warps * bdy * (head_dim + 2) * sizeof(float);

    // Per-warp smem pointers
    __half* q_smem = (__half*)smem_raw;
    uint8_t* warp_base = smem_raw + q_smem_bytes + warp_id * per_warp_bytes;
    __half* kv_smem = (__half*)warp_base;
    uint8_t* staging = warp_base + kv_fp16_bytes;
    float* smem_norms = (float*)(staging + staging_bytes);
    float* scores_smem = (float*)((uint8_t*)smem_norms + norms_bytes);

    AttentionVariant variant(params, batch_idx, smem_raw);

    // =====================================================================
    // Load Q into shared memory: [16, head_dim] fp16, row-major
    // Rows 0..bdy-1 = query heads, rows bdy..15 = zero
    // =====================================================================
    {
        // Cooperative zero-init across all threads in the block
        uint32_t flat_tid = warp_id * 32 + tx;
        uint32_t total_elems = V5_WMMA_M * head_dim;
        for (uint32_t i = flat_tid; i < total_elems; i += 32 * num_warps) {
            q_smem[i] = __float2half(0.0f);
        }
        __syncthreads();

        // Load valid Q rows + optional fused Hadamard rotation.
        // Forward rotation: Q' = FWHT(signs × Q) × scale
        // Uses 32 threads × VEC=4 layout (natural for this kernel).
        if (warp_id == 0) {
            constexpr uint32_t VEC = head_dim / 32;  // 4 for hd=128
            for (uint32_t qh = 0; qh < bdy; qh++) {
                uint32_t qo_head_idx = kv_head_idx * bdy + qh;
                const __half* q_ptr = (const __half*)params.q
                    + batch_idx * params.q_stride_n
                    + qo_head_idx * params.q_stride_h;

                float v[VEC];
                #pragma unroll
                for (uint32_t i = 0; i < VEC; i++) {
                    uint32_t d = tx * VEC + i;
                    float qval = __half2float(q_ptr[d]);
                    v[i] = params.hadamard_signs ?
                        qval * params.hadamard_signs[d] : qval;
                }

                if (params.hadamard_signs) {
                    // Local FWHT stages
                    #pragma unroll
                    for (uint32_t h = 1; h < VEC; h <<= 1) {
                        #pragma unroll
                        for (uint32_t ii = 0; ii < VEC; ii += 2 * h) {
                            #pragma unroll
                            for (uint32_t k = 0; k < h; k++) {
                                float a = v[ii + k], b = v[ii + k + h];
                                v[ii + k] = a + b;
                                v[ii + k + h] = a - b;
                            }
                        }
                    }
                    // Cross-thread FWHT stages (all 32 threads)
                    #pragma unroll
                    for (uint32_t delta = 1; delta < 32; delta <<= 1) {
                        #pragma unroll
                        for (uint32_t i = 0; i < VEC; i++) {
                            float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], delta);
                            if (tx & delta)
                                v[i] = partner - v[i];
                            else
                                v[i] = v[i] + partner;
                        }
                    }
                    float had_scale = params.hadamard_scale;
                    #pragma unroll
                    for (uint32_t i = 0; i < VEC; i++) v[i] *= had_scale;
                }

                // Write to smem in row-major [16, head_dim] layout
                #pragma unroll
                for (uint32_t i = 0; i < VEC; i++) {
                    q_smem[qh * head_dim + tx * VEC + i] = __float2half(v[i]);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // =====================================================================
    // Per-query-head state: online softmax + output accumulator
    // Each warp maintains state for ALL bdy query heads.
    // =====================================================================
    constexpr uint32_t odims_per_thread = head_dim / 32;

    float m_vals[bdy];
    float d_vals[bdy];
    float o_accs[bdy][odims_per_thread];

    #pragma unroll
    for (uint32_t h = 0; h < bdy; h++) {
        m_vals[h] = -math::inf;
        d_vals[h] = 0.f;
        #pragma unroll
        for (uint32_t i = 0; i < odims_per_thread; i++) {
            o_accs[h][i] = 0.f;
        }
    }

    // =====================================================================
    // Main loop: iterate over KV tokens assigned to this warp
    // =====================================================================
    uint32_t tokens_per_warp = (chunk_size + num_warps - 1) / num_warps;
    uint32_t warp_token_start = warp_id * tokens_per_warp;
    uint32_t warp_token_end = min(warp_token_start + tokens_per_warp, chunk_size);
    uint32_t warp_tokens = (warp_token_end > warp_token_start)
                           ? (warp_token_end - warp_token_start) : 0;
    uint32_t num_tiles = (warp_tokens + tile_n - 1) / tile_n;

    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        uint32_t tile_start_local = tile * tile_n;
        uint32_t tile_start_abs = chunk_start + warp_token_start + tile_start_local;
        uint32_t tile_valid = min(tile_n, warp_tokens - tile_start_local);

        // -----------------------------------------------------------------
        // Phase 1: cp_async packed K bytes -> staging smem
        // -----------------------------------------------------------------
        {
            constexpr uint32_t total_segments = (tile_n * bytes_per_row) / 16;
            // With 32 threads: each thread handles total_segments/32 segments
            // For head_dim=128: total_segments = 16*64/16 = 64, each thread = 2
            for (uint32_t s = tx; s < total_segments; s += 32) {
                uint32_t row = s / (bytes_per_row / 16);
                uint32_t seg_in_row = s % (bytes_per_row / 16);
                bool valid = (row < tile_valid);

                const uint8_t* src = params.k_quant;  // default (invalid)
                if (valid) {
                    uint32_t token_idx = tile_start_abs + row;
                    if constexpr (paged) {
                        // HYP-035: walk page table per token. kv_slab layout:
                        // [num_blocks, block_size, num_kv_heads, ebs]. k_quant
                        // holds the K slab base pointer in paged mode.
                        uint32_t page_iter = params.indptr[batch_idx]
                                             + token_idx / params.block_size;
                        uint32_t entry_idx = token_idx % params.block_size;
                        uint32_t page_idx  = params.indices[page_iter];
                        src = params.k_quant
                              + (size_t)page_idx * params.block_size
                                    * params.num_kv_heads * params.ebs
                              + (size_t)entry_idx * params.num_kv_heads * params.ebs
                              + (size_t)kv_head_idx * params.ebs
                              + seg_in_row * 16;
                    } else {
                        src = params.k_quant + quant_base
                              + (size_t)token_idx * params.quant_stride_token
                              + seg_in_row * 16;
                    }
                }
                uint8_t* dst = staging + row * bytes_per_row + seg_in_row * 16;
                cp_async::pred_load<128, cp_async::PrefetchMode::kPrefetch,
                                    cp_async::SharedMemFillMode::kFillZero>(
                    dst, src, valid);
            }
            cp_async::commit_group();
        }

        // Precompute norms (overlap with cp_async)
        {
            constexpr uint32_t total_norms = tile_n * dim_chunks;
            for (uint32_t n = tx; n < total_norms; n += 32) {
                uint32_t row = n / dim_chunks;
                uint32_t chunk = n % dim_chunks;
                float ns = 0.0f;
                if (row < tile_valid) {
                    uint32_t token_idx = tile_start_abs + row;
                    if constexpr (paged) {
                        // Norms live at offset `qbytes` within each entry.
                        uint32_t page_iter = params.indptr[batch_idx]
                                             + token_idx / params.block_size;
                        uint32_t entry_idx = token_idx % params.block_size;
                        uint32_t page_idx  = params.indices[page_iter];
                        const __half* norm_ptr = (const __half*)(
                            params.k_quant
                            + (size_t)page_idx * params.block_size
                                  * params.num_kv_heads * params.ebs
                            + (size_t)entry_idx * params.num_kv_heads * params.ebs
                            + (size_t)kv_head_idx * params.ebs
                            + params.qbytes);
                        ns = codebook_scale * __half2float(norm_ptr[chunk]);
                    } else {
                        size_t n_off = norm_base
                                       + (size_t)token_idx * params.norm_stride_token
                                       + chunk;
                        ns = codebook_scale * __half2float(params.k_norms[n_off]);
                    }
                }
                smem_norms[row * dim_chunks + chunk] = ns;
            }
        }

        cp_async::wait_group<0>();
        __syncwarp();

        // -----------------------------------------------------------------
        // Phase 2: Dequant staging -> fp16 KV buffer
        // -----------------------------------------------------------------
        for (uint32_t row = 0; row < tile_n; row++) {
            if (row < tile_valid) {
                dequant_row_to_fp16_v5<head_dim>(
                    kv_smem + row * head_dim,
                    staging + row * bytes_per_row,
                    smem_norms + row * dim_chunks,
                    tx, cb_reg
                );
            } else {
                for (uint32_t d = tx; d < head_dim; d += 32) {
                    kv_smem[row * head_dim + d] = __float2half(0.0f);
                }
            }
        }
        __syncwarp();

        // -----------------------------------------------------------------
        // Phase 3: WMMA QK dot product
        // C[16,16] = sum_{kt} A[16,16] * B[16,16]
        // A = Q: [16, head_dim] row-major, loaded 16x16 at a time along K
        // B = K^T: kv_smem [16, head_dim] row-major, loaded as col_major = transposed
        // -----------------------------------------------------------------
        // Explicit ::nvcuda::wmma:: qualification — the enclosing
        // `using namespace nvcuda::wmma` is resolved differently by older
        // nvcc (NGC 24.01 / CUDA 12.3) vs newer, breaking compile on some
        // images. Fully-qualified calls are portable.
        ::nvcuda::wmma::fragment<::nvcuda::wmma::matrix_a, V5_WMMA_M, V5_WMMA_N, V5_WMMA_K,
                               __half, ::nvcuda::wmma::row_major> a_frag;
        ::nvcuda::wmma::fragment<::nvcuda::wmma::matrix_b, V5_WMMA_M, V5_WMMA_N, V5_WMMA_K,
                               __half, ::nvcuda::wmma::col_major> b_frag;
        ::nvcuda::wmma::fragment<::nvcuda::wmma::accumulator, V5_WMMA_M, V5_WMMA_N, V5_WMMA_K,
                               float> c_frag;
        ::nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (uint32_t kt = 0; kt < k_tiles; kt++) {
            // A: Q[0..15, kt*16..(kt+1)*16] from q_smem
            // q_smem is row-major [16, head_dim], leading dim = head_dim
            ::nvcuda::wmma::load_matrix_sync(a_frag, q_smem + kt * V5_WMMA_K, head_dim);

            // B: K^T -- kv_smem is [16, head_dim] row-major
            // Loading as col_major with ldm=head_dim gives us K^T
            ::nvcuda::wmma::load_matrix_sync(b_frag, kv_smem + kt * V5_WMMA_K, head_dim);

            ::nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store C to shared memory
        ::nvcuda::wmma::store_matrix_sync(scores_smem, c_frag, V5_WMMA_N,
                                        ::nvcuda::wmma::mem_row_major);
        __syncwarp();

        // -----------------------------------------------------------------
        // Phase 3b: Online softmax (scalar, each thread reads its rows)
        // scores_smem is [16, 16] row-major. Row h = query head h.
        // Only rows 0..bdy-1 are valid.
        // -----------------------------------------------------------------
        // All 32 threads in the warp read the SAME score data and update
        // their SAME per-head state. This is intentional -- the redundant
        // reads are cheap and avoid cross-lane communication for softmax.
        // HYP-036 addendum: warp-butterfly reduction was tested (1510 ns)
        // vs this scalar unroll (327 ns) — 4.6× slower. Kept serial unroll:
        // compiler schedules the 16 independent fmax/exp2 in parallel across
        // the warp SFUs; shuffle reductions add serial dependency chains.
        #pragma unroll
        for (uint32_t h = 0; h < bdy; h++) {
            float m_prev = m_vals[h];

            float local_scores[V5_TILE_N];
            float local_max = -math::inf;

            uint32_t qo_head_idx = kv_head_idx * bdy + h;

            #pragma unroll
            for (uint32_t j = 0; j < V5_TILE_N; j++) {
                float raw_s = scores_smem[h * V5_WMMA_N + j];

                // LogitsTransform + softmax scaling
                uint32_t pos = tile_start_abs + j;
                raw_s = variant.LogitsTransform(params, raw_s, batch_idx, 0,
                                                pos, qo_head_idx, kv_head_idx);
                if constexpr (AttentionVariant::use_softmax) {
                    raw_s *= variant.sm_scale_log2;
                }

                bool mask = variant.LogitsMask(params, batch_idx, 0,
                                                pos, qo_head_idx, kv_head_idx);
                raw_s = (j < tile_valid && mask) ? raw_s : -math::inf;

                local_scores[j] = raw_s;
                local_max = max(local_max, raw_s);
            }

            m_vals[h] = max(m_vals[h], local_max);

            // Online softmax rescale
            if constexpr (AttentionVariant::use_softmax) {
                float o_scale = math::ptx_exp2(m_prev - m_vals[h]);
                d_vals[h] *= o_scale;
                #pragma unroll
                for (uint32_t j = 0; j < V5_TILE_N; j++) {
                    local_scores[j] = math::ptx_exp2(local_scores[j] - m_vals[h]);
                    d_vals[h] += local_scores[j];
                }
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    o_accs[h][i] *= o_scale;
                }
            }

            // Write attention weights back for V accumulate
            #pragma unroll
            for (uint32_t j = 0; j < V5_TILE_N; j++) {
                scores_smem[h * V5_WMMA_N + j] = local_scores[j];
            }
        }
        __syncwarp();

        // -----------------------------------------------------------------
        // Phase 4: Load V + dequant, then scalar V accumulate
        // V accumulate: for each head h, O[h] += sum_j attn_weight[h,j] * V[j]
        // Scalar because V is not a large matmul (bdy * tile_n * head_dim/32).
        // -----------------------------------------------------------------
        {
            constexpr uint32_t total_segments = (tile_n * bytes_per_row) / 16;
            for (uint32_t s = tx; s < total_segments; s += 32) {
                uint32_t row = s / (bytes_per_row / 16);
                uint32_t seg_in_row = s % (bytes_per_row / 16);
                bool valid = (row < tile_valid);

                const uint8_t* src = params.v_quant;
                if (valid) {
                    uint32_t token_idx = tile_start_abs + row;
                    if constexpr (paged) {
                        uint32_t page_iter = params.indptr[batch_idx]
                                             + token_idx / params.block_size;
                        uint32_t entry_idx = token_idx % params.block_size;
                        uint32_t page_idx  = params.indices[page_iter];
                        src = params.v_quant
                              + (size_t)page_idx * params.block_size
                                    * params.num_kv_heads * params.ebs
                              + (size_t)entry_idx * params.num_kv_heads * params.ebs
                              + (size_t)kv_head_idx * params.ebs
                              + seg_in_row * 16;
                    } else {
                        src = params.v_quant + quant_base
                              + (size_t)token_idx * params.quant_stride_token
                              + seg_in_row * 16;
                    }
                }
                uint8_t* dst = staging + row * bytes_per_row + seg_in_row * 16;
                cp_async::pred_load<128, cp_async::PrefetchMode::kPrefetch,
                                    cp_async::SharedMemFillMode::kFillZero>(
                    dst, src, valid);
            }
            cp_async::commit_group();
        }

        // V norms
        {
            constexpr uint32_t total_norms = tile_n * dim_chunks;
            for (uint32_t n = tx; n < total_norms; n += 32) {
                uint32_t row = n / dim_chunks;
                uint32_t chunk = n % dim_chunks;
                float ns = 0.0f;
                if (row < tile_valid) {
                    uint32_t token_idx = tile_start_abs + row;
                    if constexpr (paged) {
                        uint32_t page_iter = params.indptr[batch_idx]
                                             + token_idx / params.block_size;
                        uint32_t entry_idx = token_idx % params.block_size;
                        uint32_t page_idx  = params.indices[page_iter];
                        const __half* norm_ptr = (const __half*)(
                            params.v_quant
                            + (size_t)page_idx * params.block_size
                                  * params.num_kv_heads * params.ebs
                            + (size_t)entry_idx * params.num_kv_heads * params.ebs
                            + (size_t)kv_head_idx * params.ebs
                            + params.qbytes);
                        ns = codebook_scale * __half2float(norm_ptr[chunk]);
                    } else {
                        size_t n_off = norm_base
                                       + (size_t)token_idx * params.norm_stride_token
                                       + chunk;
                        ns = codebook_scale * __half2float(params.v_norms[n_off]);
                    }
                }
                smem_norms[row * dim_chunks + chunk] = ns;
            }
        }

        cp_async::wait_group<0>();
        __syncwarp();

        // Dequant V -> fp16
        for (uint32_t row = 0; row < tile_n; row++) {
            if (row < tile_valid) {
                dequant_row_to_fp16_v5<head_dim>(
                    kv_smem + row * head_dim,
                    staging + row * bytes_per_row,
                    smem_norms + row * dim_chunks,
                    tx, cb_reg
                );
            } else {
                for (uint32_t d = tx; d < head_dim; d += 32) {
                    kv_smem[row * head_dim + d] = __float2half(0.0f);
                }
            }
        }
        __syncwarp();

        // Scalar V accumulate: each thread owns odims_per_thread output dims
        #pragma unroll
        for (uint32_t h = 0; h < bdy; h++) {
            #pragma unroll
            for (uint32_t j = 0; j < V5_TILE_N; j++) {
                float w = scores_smem[h * V5_WMMA_N + j];
                if (w > 0.0f) {
                    #pragma unroll
                    for (uint32_t i = 0; i < odims_per_thread; i++) {
                        uint32_t d = tx + i * 32;
                        o_accs[h][i] += w * __half2float(kv_smem[j * head_dim + d]);
                    }
                }
            }
        }
    }  // end main loop

    // =====================================================================
    // Cross-warp merge (online softmax)
    // =====================================================================
    __syncthreads();

    // Merge buffer: reuse smem_raw. Layout: [num_warps, bdy, head_dim + 2] float
    float* merge_buf = (float*)smem_raw;

    // Each warp stores its state
    #pragma unroll
    for (uint32_t h = 0; h < bdy; h++) {
        uint32_t slot = (warp_id * bdy + h) * (head_dim + 2);
        if (tx == 0) {
            merge_buf[slot + 0] = m_vals[h];
            merge_buf[slot + 1] = d_vals[h];
        }
        #pragma unroll
        for (uint32_t i = 0; i < odims_per_thread; i++) {
            uint32_t d = tx + i * 32;
            merge_buf[slot + 2 + d] = o_accs[h][i];
        }
    }
    __syncthreads();

    // Warp 0 merges across all warps
    if (warp_id == 0) {
        #pragma unroll
        for (uint32_t h = 0; h < bdy; h++) {
            float m_merged = -math::inf;
            float d_merged = 0.f;
            float o_merged[odims_per_thread];
            #pragma unroll
            for (uint32_t i = 0; i < odims_per_thread; i++) o_merged[i] = 0.f;

            for (uint32_t w = 0; w < num_warps; w++) {
                uint32_t slot = (w * bdy + h) * (head_dim + 2);
                float m_w = merge_buf[slot + 0];
                float d_w = merge_buf[slot + 1];

                // Read from smem -- all lanes read the same m_w/d_w (broadcast)
                // Each lane reads its own output dimensions
                if (tx == 0) {
                    // dummy to keep tx==0 sync -- all threads read though
                }

                if (m_w <= -1e29f) continue;

                if (m_merged <= -1e29f) {
                    m_merged = m_w;
                    d_merged = d_w;
                    #pragma unroll
                    for (uint32_t i = 0; i < odims_per_thread; i++) {
                        uint32_t d = tx + i * 32;
                        o_merged[i] = merge_buf[slot + 2 + d];
                    }
                    continue;
                }

                float m_new = fmaxf(m_merged, m_w);
                float scale_old = expf(m_merged - m_new);
                float scale_new = expf(m_w - m_new);

                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    uint32_t d = tx + i * 32;
                    float o_w = merge_buf[slot + 2 + d];
                    o_merged[i] = o_merged[i] * scale_old + o_w * scale_new;
                }
                d_merged = d_merged * scale_old + d_w * scale_new;
                m_merged = m_new;
            }

            // Normalize
            if (d_merged > 0.f) {
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    o_merged[i] /= d_merged;
                }
            }

            // Inverse Hadamard: FWHT(output) × scale × signs
            // o_merged is in strided layout [tx + i*32]. Repack to
            // contiguous [tx*VEC + i] for butterfly, then write out.
            if (params.hadamard_signs) {
                constexpr uint32_t VEC = head_dim / 32;
                float v[VEC];

                // Repack: strided → contiguous via smem
                float* tmp = (float*)smem_raw;  // reuse merge_buf
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    tmp[tx + i * 32] = o_merged[i];
                }
                __syncwarp();
                #pragma unroll
                for (uint32_t i = 0; i < VEC; i++) {
                    v[i] = tmp[tx * VEC + i];
                }

                // Local FWHT stages
                #pragma unroll
                for (uint32_t hh = 1; hh < VEC; hh <<= 1) {
                    #pragma unroll
                    for (uint32_t ii = 0; ii < VEC; ii += 2 * hh) {
                        #pragma unroll
                        for (uint32_t k = 0; k < hh; k++) {
                            float a = v[ii + k], b = v[ii + k + hh];
                            v[ii + k] = a + b;
                            v[ii + k + hh] = a - b;
                        }
                    }
                }
                // Cross-thread FWHT
                #pragma unroll
                for (uint32_t delta = 1; delta < 32; delta <<= 1) {
                    #pragma unroll
                    for (uint32_t i = 0; i < VEC; i++) {
                        float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], delta);
                        if (tx & delta)
                            v[i] = partner - v[i];
                        else
                            v[i] = v[i] + partner;
                    }
                }
                // Scale × signs, then write back to strided via smem
                float had_scale = params.hadamard_scale;
                #pragma unroll
                for (uint32_t i = 0; i < VEC; i++) {
                    tmp[tx * VEC + i] = v[i] * had_scale
                        * params.hadamard_signs[tx * VEC + i];
                }
                __syncwarp();
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    o_merged[i] = tmp[tx + i * 32];
                }
            }

            // Output
            uint32_t qo_head_idx = kv_head_idx * bdy + h;

            if (params.partition_kv && params.partition_o != nullptr) {
                float* out = params.partition_o
                    + (bx * num_qo_heads + qo_head_idx) * head_dim;
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    out[tx + i * 32] = o_merged[i];
                }
                if (params.partition_lse != nullptr && tx == 0) {
                    params.partition_lse[bx * num_qo_heads + qo_head_idx] =
                        m_merged + logf(d_merged);
                }
            } else {
                __half* out = (__half*)params.o
                    + (batch_idx * num_qo_heads + qo_head_idx) * head_dim;
                #pragma unroll
                for (uint32_t i = 0; i < odims_per_thread; i++) {
                    out[tx + i * 32] = __float2half(o_merged[i]);
                }
                if (params.lse != nullptr && tx == 0) {
                    params.lse[batch_idx * num_qo_heads + qo_head_idx] =
                        m_merged + logf(d_merged);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Shared memory size calculation
// --------------------------------------------------------------------------
template <uint32_t head_dim, uint32_t bdy, uint32_t num_warps>
constexpr uint32_t calc_smem_v5_tc() {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;

    constexpr uint32_t q_smem_bytes = V5_WMMA_M * head_dim * sizeof(__half);
    constexpr uint32_t kv_fp16_bytes = V5_TILE_N * head_dim * sizeof(__half);
    constexpr uint32_t staging_bytes = ((V5_TILE_N * bytes_per_row + 15) / 16) * 16;
    constexpr uint32_t norms_bytes = V5_TILE_N * dim_chunks * sizeof(float);
    constexpr uint32_t scores_bytes = V5_WMMA_M * V5_WMMA_N * sizeof(float);
    constexpr uint32_t per_warp_bytes = kv_fp16_bytes + staging_bytes + norms_bytes + scores_bytes;

    constexpr uint32_t main_loop_bytes = q_smem_bytes + num_warps * per_warp_bytes;
    constexpr uint32_t merge_bytes = num_warps * bdy * (head_dim + 2) * sizeof(float);
    constexpr uint32_t max_bytes = (main_loop_bytes > merge_bytes)
                                    ? main_loop_bytes : merge_bytes;
    return max_bytes + 128;  // alignment padding
}

// --------------------------------------------------------------------------
// Kernel wrapper
// --------------------------------------------------------------------------
template <uint32_t head_dim, uint32_t bdy, uint32_t num_warps,
          bool paged,
          typename AttentionVariant, typename Params>
__global__ void TurboQuantContiguousDecodeKernelV5TC(
    const __grid_constant__ Params params
) {
    extern __shared__ uint8_t smem[];
    TurboQuantContiguousDecodeDeviceV5TC<V5_TILE_N, head_dim, bdy, num_warps,
                                         paged, AttentionVariant, Params>(
        params, smem);
}

}  // namespace flashinfer
