#pragma once
// Modified FlashInfer paged decode kernel with TurboQuant 4-bit KV.
//
// This is a thin wrapper around FlashInfer's BatchDecodeWithPagedKVCacheDevice.
// The ONLY change: KV loading uses dequant_load_kv_tile instead of cp_async.
// All QK/softmax/V accumulation/output code is FlashInfer's original.
//
// Shared memory holds fp16 (same as FlashInfer FP16 path).
// DTypeKV = __half for the compute path.

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/math.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/state.cuh"

#include "turboquant/flashinfer_dequant_load.cuh"
#include "turboquant/page_turbo.cuh"

namespace turboquant {

namespace cg = cooperative_groups;
using flashinfer::vec_t;
using flashinfer::state_t;
using flashinfer::math::shfl_xor_sync;

// Reuse FlashInfer's compute_qk and update_local_state directly.
// These read fp16 from shared memory — identical to the FP16 path.
// (They're in an anonymous namespace in decode.cuh, so we reference them via the header.)

// Warp-safe FWHT for one 64-dim chunk using shared memory.
// Only the bdx threads (within the same warp) participate.
// Uses __syncwarp() instead of __syncthreads().
// data: [64] floats in smem. bdx=8 threads, each owns 8 consecutive values.
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ void warp_fwht_64(float* data, uint32_t tx, uint32_t warp_mask) {
    constexpr uint32_t d = bdx * vec_size;

    // Phase 1: intra-thread butterflies (h=1,2,4)
    #pragma unroll
    for (uint32_t h = 1; h < vec_size; h *= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
            uint32_t partner = i ^ h;
            if (partner > i) {
                float a = data[tx * vec_size + i];
                float b = data[tx * vec_size + partner];
                data[tx * vec_size + i] = a + b;
                data[tx * vec_size + partner] = a - b;
            }
        }
    }
    __syncwarp(warp_mask);

    // Phase 2: inter-thread butterflies (h=8,16,32)
    for (uint32_t h = vec_size; h < d; h *= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
            uint32_t gidx = tx * vec_size + i;
            uint32_t pidx = gidx ^ h;
            float my_val = data[gidx];
            float p_val = data[pidx];
            __syncwarp(warp_mask);
            if (gidx < pidx) {
                data[gidx] = my_val + p_val;
                data[pidx] = my_val - p_val;
            }
            __syncwarp(warp_mask);
        }
    }

    float inv_sqrt = rsqrtf(static_cast<float>(d));
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; i++)
        data[tx * vec_size + i] *= inv_sqrt;
    __syncwarp(warp_mask);
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          uint32_t tile_size_per_bdx, uint32_t num_stages_smem,
          typename IdType>
__global__ void FlashInferDecodeWithTurboQuantKV(
    // Query and output (same as FlashInfer)
    const __half* __restrict__ q,     // [batch, num_qo_heads, head_dim] (UNROTATED if signs != nullptr)
    __half* __restrict__ o,           // [batch, num_qo_heads, head_dim] (UNROTATED output)
    float* __restrict__ lse,
    const float* __restrict__ signs,  // [padded_dim] Hadamard rotation signs (nullptr = no rotation)

    // TurboQuant quantized KV (replaces paged_kv_t)
    const paged_kv_turbo_t<IdType> paged_kv,

    // FlashInfer batch decode metadata
    const IdType* __restrict__ request_indices,
    const IdType* __restrict__ kv_tile_indices,
    const bool* __restrict__ block_valid_mask,
    const uint32_t* __restrict__ kv_chunk_size_ptr,
    uint32_t padded_batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    bool partition_kv,
    float sm_scale
) {
    auto block = cg::this_thread_block();
    constexpr uint32_t head_dim = bdx * vec_size;  // e.g. 64

    // NOTE: For head_dim=128, this kernel handles one 64-dim chunk.
    // The caller launches multiple kernels (one per chunk) and merges.
    // Or we iterate chunks inside — same as our standalone kernel.
    // For simplicity, iterate over dim_chunks internally.

    uint32_t batch_idx = request_indices[blockIdx.x];
    uint32_t kv_tile_idx = kv_tile_indices[blockIdx.x];
    uint32_t kv_head_idx = blockIdx.y;
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    uint32_t qo_head_idx = kv_head_idx * bdy + ty;

    if (block_valid_mask && !block_valid_mask[blockIdx.x]) return;

    uint32_t kv_chunk_size = *kv_chunk_size_ptr;
    uint32_t kv_len = paged_kv.get_length(batch_idx);
    uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    uint32_t chunk_end = partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    uint32_t chunk_size = chunk_end - chunk_start;

    float sm_scale_log2 = sm_scale * flashinfer::math::log2e;
    uint32_t actual_head_dim = paged_kv.head_dim;
    uint32_t dim_chunks = paged_kv.dim_chunks;
    constexpr uint32_t CHUNK_DIMS = 64;

    // Shared memory: fp16 tile for K and V (same layout as FlashInfer FP16)
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    extern __shared__ uint8_t smem[];
    __half* k_smem = (__half*)smem;
    __half* v_smem = (__half*)(smem + tile_tokens * CHUNK_DIMS * sizeof(__half));

    // Load query and optionally apply Hadamard rotation in-kernel.
    // Each ty is a different QO head — needs its own FWHT.
    // Use smem scratch: bdy buffers of 64 floats each, parallel across ty.
    constexpr uint32_t MAX_CHUNKS = 4;
    float q_reg[MAX_CHUNKS][vec_size];
    float* fwht_scratch = reinterpret_cast<float*>(smem);  // [bdy, 64] floats

    for (uint32_t c = 0; c < dim_chunks && c < MAX_CHUNKS; c++) {
        uint32_t q_dim = c * CHUNK_DIMS + tx * vec_size;

        // tz=0 loads Q for each ty into its own scratch buffer
        if (tz == 0) {
            float* my_scratch = fwht_scratch + ty * CHUNK_DIMS;
            if (q_dim < actual_head_dim) {
                vec_t<float, vec_size> qv;
                qv.cast_load(q + batch_idx * num_qo_heads * actual_head_dim
                               + qo_head_idx * actual_head_dim + q_dim);
                for (uint32_t i = 0; i < vec_size; i++)
                    my_scratch[tx * vec_size + i] = qv[i];
            } else {
                for (uint32_t i = 0; i < vec_size; i++)
                    my_scratch[tx * vec_size + i] = 0.f;
            }
        }
        block.sync();

        // Apply rotation per-ty using warp-safe FWHT (only bdx threads participate per warp)
        if (signs != nullptr && tz == 0) {
            float* my_scratch = fwht_scratch + ty * CHUNK_DIMS;
            // signs * x
            for (uint32_t i = 0; i < vec_size; i++)
                my_scratch[tx * vec_size + i] *= signs[c * CHUNK_DIMS + tx * vec_size + i];
            // Compute warp mask: only the bdx lanes for this ty in tz=0
            // Lane ID = tx + ty * bdx + tz * bdx * bdy. For tz=0: lane = tx + ty*bdx.
            // We want mask for lanes [ty*bdx .. ty*bdx+bdx-1]
            uint32_t warp_mask = ((1u << bdx) - 1u) << (ty * bdx);
            warp_fwht_64<vec_size, bdx>(my_scratch, tx, warp_mask);
        }
        block.sync();

        // All threads read from their ty's scratch
        float* my_scratch = fwht_scratch + ty * CHUNK_DIMS;
        for (uint32_t i = 0; i < vec_size; i++)
            q_reg[c][i] = my_scratch[tx * vec_size + i];
        block.sync();
    }

    // Output accumulators (per chunk)
    float o_acc[MAX_CHUNKS][vec_size];
    for (uint32_t c = 0; c < MAX_CHUNKS; c++)
        for (uint32_t i = 0; i < vec_size; i++)
            o_acc[c][i] = 0.f;
    float m_global = -flashinfer::math::inf;
    float d_global = 0.f;

    // Page iteration
    uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
    IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    // Each tz group loads bdy * tile_size_per_bdx tokens cooperatively (ty helps load).
    // All ty threads within the same tz read ALL bdy * tile_size_per_bdx tokens for QK.
    // smem row base for this tz: tz * bdy * tile_size_per_bdx
    uint32_t my_tz_smem_base = tz * bdy * tile_size_per_bdx;
    constexpr uint32_t tokens_per_tz = bdy * tile_size_per_bdx;

    for (uint32_t iter = 0; iter < flashinfer::ceil_div(chunk_size, tile_tokens); ++iter) {
        // === QK: accumulate across dim chunks ===
        float s[tokens_per_tz];
        for (uint32_t j = 0; j < tokens_per_tz; j++) s[j] = 0.f;

        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            dequant_load_kv_tile<tile_size_per_bdx, vec_size, bdx, bdy, bdz>(
                k_smem, paged_kv, paged_kv.k_quant, paged_kv.k_norms,
                kv_head_idx, chunk,
                packed_page_iter_base + iter * tile_tokens,
                chunk_size - iter * tile_tokens,
                last_indptr, tx, ty, tz
            );
            block.sync();

            // Each tz reads ALL its tokens (bdy * tile_size_per_bdx) from smem
            #pragma unroll
            for (uint32_t j = 0; j < tokens_per_tz; ++j) {
                vec_t<float, vec_size> k_vec;
                k_vec.cast_load(k_smem + ((my_tz_smem_base + j) * bdx + tx) * vec_size);
                float partial = 0.f;
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    partial += q_reg[chunk][i] * k_vec[i];
                }
                #pragma unroll
                for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
                    partial += shfl_xor_sync(partial, offset);
                }
                s[j] += partial;
            }
            block.sync();
        }

        // === Online softmax — each tz processes its token range ===
        uint32_t my_token_base = iter * tile_tokens + my_tz_smem_base;
        float m_prev = m_global;
        #pragma unroll
        for (uint32_t j = 0; j < tokens_per_tz; ++j) {
            s[j] *= sm_scale_log2;
            s[j] = (my_token_base + j < chunk_size) ? s[j] : -flashinfer::math::inf;
            m_global = max(m_global, s[j]);
        }
        // NaN-safe softmax: when all scores are -inf, skip accumulation
        if (m_global == -flashinfer::math::inf) {
            // All tokens invalid — o_acc stays zero, d_global stays 0
        } else {
            float o_scale = (m_prev == -flashinfer::math::inf) ? 0.0f
                          : flashinfer::math::ptx_exp2(m_prev - m_global);
            d_global *= o_scale;
            #pragma unroll
            for (uint32_t j = 0; j < tokens_per_tz; ++j) {
                s[j] = (s[j] == -flashinfer::math::inf) ? 0.0f
                      : flashinfer::math::ptx_exp2(s[j] - m_global);
                d_global += s[j];
            }
            for (uint32_t c = 0; c < dim_chunks; c++)
                for (uint32_t i = 0; i < vec_size; i++)
                    o_acc[c][i] *= o_scale;
        }

        // === V accumulation — each tz reads its own rows ===
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t v_dim = chunk * CHUNK_DIMS + tx * vec_size;
            if (v_dim >= actual_head_dim) continue;

            dequant_load_kv_tile<tile_size_per_bdx, vec_size, bdx, bdy, bdz>(
                v_smem, paged_kv, paged_kv.v_quant, paged_kv.v_norms,
                kv_head_idx, chunk,
                packed_page_iter_base + iter * tile_tokens,
                chunk_size - iter * tile_tokens,
                last_indptr, tx, ty, tz
            );
            block.sync();

            #pragma unroll
            for (uint32_t j = 0; j < tokens_per_tz; ++j) {
                vec_t<float, vec_size> v_vec;
                v_vec.cast_load(v_smem + ((my_tz_smem_base + j) * bdx + tx) * vec_size);
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    o_acc[chunk][i] += s[j] * v_vec[i];
                }
            }
            block.sync();
        }
    }

    // === Merge across bdz groups ===
    // Each (tz, ty, tx) thread has its own (m_global, d_global, o_acc).
    // m and d are the same across all tx in the same (tz, ty) — they're broadcast by shfl.
    // o_acc differs per tx — each tx holds a different vec_size slice of the output.
    //
    // Strategy: store (m, d) once per (tz, ty) from tx=0.
    // Store o_acc per (tz, ty, tx) — each tx writes its own slice.
    // Merge: tz=0 reads all tz groups, scales and accumulates.
    if constexpr (bdz > 1) {
        float* merge_smem = reinterpret_cast<float*>(smem);
        uint32_t tz_ty_idx = tz * bdy + ty;

        // Layout per (tz, ty): [m, d, o_per_tx[bdx][MAX_CHUNKS * vec_size]]
        // Size per entry: 2 + bdx * MAX_CHUNKS * vec_size
        constexpr uint32_t o_size_per_tx = MAX_CHUNKS * vec_size;
        constexpr uint32_t merge_entry_size = 2 + bdx * o_size_per_tx;

        // tx=0 writes m and d
        if (tx == 0) {
            merge_smem[tz_ty_idx * merge_entry_size + 0] = m_global;
            merge_smem[tz_ty_idx * merge_entry_size + 1] = d_global;
        }
        // Each tx writes its own o_acc slice
        for (uint32_t c = 0; c < dim_chunks; c++)
            for (uint32_t i = 0; i < vec_size; i++)
                merge_smem[tz_ty_idx * merge_entry_size + 2 + tx * o_size_per_tx + c * vec_size + i] = o_acc[c][i];
        block.sync();

        if (tz == 0) {
            for (uint32_t z = 1; z < bdz; z++) {
                uint32_t other = z * bdy + ty;
                float m_other = merge_smem[other * merge_entry_size + 0];
                float d_other = merge_smem[other * merge_entry_size + 1];

                if (m_other == -flashinfer::math::inf) continue;
                if (m_global == -flashinfer::math::inf) {
                    m_global = m_other;
                    d_global = d_other;
                    for (uint32_t c = 0; c < dim_chunks; c++)
                        for (uint32_t i = 0; i < vec_size; i++)
                            o_acc[c][i] = merge_smem[other * merge_entry_size + 2 + tx * o_size_per_tx + c * vec_size + i];
                    continue;
                }

                float m_new = max(m_global, m_other);
                float s_self = flashinfer::math::ptx_exp2(m_global - m_new);
                float s_other_scale = flashinfer::math::ptx_exp2(m_other - m_new);

                d_global = d_global * s_self + d_other * s_other_scale;
                for (uint32_t c = 0; c < dim_chunks; c++)
                    for (uint32_t i = 0; i < vec_size; i++) {
                        float o_other = merge_smem[other * merge_entry_size + 2 + tx * o_size_per_tx + c * vec_size + i];
                        o_acc[c][i] = o_acc[c][i] * s_self + o_other * s_other_scale;
                    }
                m_global = m_new;
            }
        }
    }

    // === Output normalization ===
    float d_rcp = (m_global != -flashinfer::math::inf) ? flashinfer::math::ptx_rcp(d_global) : 0.f;
    for (uint32_t c = 0; c < dim_chunks; c++)
        for (uint32_t i = 0; i < vec_size; i++)
            o_acc[c][i] *= d_rcp;

    // === Inverse Hadamard rotation on output (per chunk, in smem) ===
    if (signs != nullptr && tz == 0) {
        float* out_scratch = reinterpret_cast<float*>(smem) + ty * CHUNK_DIMS;
        uint32_t warp_mask = ((1u << bdx) - 1u) << (ty * bdx);

        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            // Write o_acc to smem
            for (uint32_t i = 0; i < vec_size; i++)
                out_scratch[tx * vec_size + i] = o_acc[chunk][i];
            __syncwarp(warp_mask);

            // Inverse: FWHT then signs
            warp_fwht_64<vec_size, bdx>(out_scratch, tx, warp_mask);
            for (uint32_t i = 0; i < vec_size; i++)
                out_scratch[tx * vec_size + i] *= signs[chunk * CHUNK_DIMS + tx * vec_size + i];
            __syncwarp(warp_mask);

            // Read back
            for (uint32_t i = 0; i < vec_size; i++)
                o_acc[chunk][i] = out_scratch[tx * vec_size + i];
            __syncwarp(warp_mask);
        }
    }

    // === Write output ===
    if (tz == 0) {
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t o_dim = chunk * CHUNK_DIMS + tx * vec_size;
            if (o_dim >= actual_head_dim) continue;

            __half* o_ptr = o + (blockIdx.x * num_qo_heads + qo_head_idx) * actual_head_dim + o_dim;
            #pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) {
                o_ptr[i] = __float2half(o_acc[chunk][i]);
            }
        }
        if (lse != nullptr) {
            lse[blockIdx.x * num_qo_heads + qo_head_idx] = m_global + __logf(d_global);
        }
    }
}

}  // namespace turboquant
