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

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          uint32_t tile_size_per_bdx, uint32_t num_stages_smem,
          typename IdType>
__global__ void FlashInferDecodeWithTurboQuantKV(
    // Query and output (same as FlashInfer)
    const __half* __restrict__ q,     // [batch, num_qo_heads, head_dim]
    __half* __restrict__ o,           // [batch, num_qo_heads, head_dim]
    float* __restrict__ lse,

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

    // Load query (iterate over chunks for head_dim > 64)
    constexpr uint32_t MAX_CHUNKS = 4;  // support up to head_dim=256
    float q_reg[MAX_CHUNKS][vec_size];
    for (uint32_t c = 0; c < dim_chunks && c < MAX_CHUNKS; c++) {
        uint32_t q_dim = c * CHUNK_DIMS + tx * vec_size;
        if (q_dim < actual_head_dim) {
            vec_t<float, vec_size> qv;
            qv.cast_load(q + batch_idx * num_qo_heads * actual_head_dim
                           + qo_head_idx * actual_head_dim + q_dim);
            for (uint32_t i = 0; i < vec_size; i++) q_reg[c][i] = qv[i];
        } else {
            for (uint32_t i = 0; i < vec_size; i++) q_reg[c][i] = 0.f;
        }
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

    for (uint32_t iter = 0; iter < flashinfer::ceil_div(chunk_size, tile_tokens); ++iter) {
        // === QK: accumulate across dim chunks ===
        float s[bdy * tile_size_per_bdx];
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; j++) s[j] = 0.f;

        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            // Load K tile: dequant from quantized cache → fp16 smem
            dequant_load_kv_tile<tile_size_per_bdx, vec_size, bdx, bdy, bdz>(
                k_smem, paged_kv, paged_kv.k_quant, paged_kv.k_norms,
                kv_head_idx, chunk,
                packed_page_iter_base + iter * tile_tokens,
                chunk_size - iter * tile_tokens,
                last_indptr, tx, ty, tz
            );
            block.sync();

            // QK dot product (identical to FlashInfer)
            #pragma unroll
            for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
                vec_t<float, vec_size> k_vec;
                k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
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

        // === Online softmax (identical to FlashInfer) ===
        float m_prev = m_global;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] *= sm_scale_log2;
            uint32_t token_idx = iter * tile_tokens + j;
            s[j] = (token_idx < chunk_size) ? s[j] : -flashinfer::math::inf;
            m_global = max(m_global, s[j]);
        }
        float o_scale = flashinfer::math::ptx_exp2(m_prev - m_global);
        d_global *= o_scale;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] = flashinfer::math::ptx_exp2(s[j] - m_global);
            d_global += s[j];
        }
        for (uint32_t c = 0; c < dim_chunks; c++)
            for (uint32_t i = 0; i < vec_size; i++)
                o_acc[c][i] *= o_scale;

        // === V accumulation (per chunk) ===
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
            for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
                vec_t<float, vec_size> v_vec;
                v_vec.cast_load(v_smem + (j * bdx + tx) * vec_size);
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    o_acc[chunk][i] += s[j] * v_vec[i];
                }
            }
            block.sync();
        }
    }

    // === Output normalization and write (identical to FlashInfer) ===
    float d_rcp = (m_global != -flashinfer::math::inf) ? flashinfer::math::ptx_rcp(d_global) : 0.f;

    if (tz == 0) {
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t o_dim = chunk * CHUNK_DIMS + tx * vec_size;
            if (o_dim >= actual_head_dim) continue;

            __half* o_ptr = o + (blockIdx.x * num_qo_heads + qo_head_idx) * actual_head_dim + o_dim;
            #pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) {
                o_ptr[i] = __float2half(o_acc[chunk][i] * d_rcp);
            }
        }
        if (lse != nullptr) {
            lse[blockIdx.x * num_qo_heads + qo_head_idx] = m_global + __logf(d_global);
        }
    }
}

}  // namespace turboquant
