#pragma once
// Forked FlashInfer decode attention kernel with fused TurboQuant dequantization.
//
// Changes from original BatchDecodeWithPagedKVCacheDevice:
//   1. Reads from paged_kv_turbo_t (quantized bytes + norms) instead of paged_kv_t (DTypeKV)
//   2. Replaces cp_async KV loads with sync load → dequant → smem store
//   3. Single-buffer pipeline (correctness first, double-buffer added in 3b if needed)
//   4. Shared memory holds fp16 after dequant (compute_qk/update_local_state unchanged)
//
// Everything after the KV load (QK dot product, softmax, V accumulate, output) is
// identical to the original FlashInfer kernel.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/math.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flashinfer/attention/state.cuh"

#include "turboquant/hadamard.cuh"
#include "turboquant/page_turbo.cuh"

namespace turboquant {

namespace cg = cooperative_groups;
using flashinfer::vec_t;
using flashinfer::state_t;

// ── Reuse FlashInfer's compute_qk and update_local_state ─────────────
// These read fp16 from shared memory and produce float results.
// They are identical to the original — copied here to avoid include tangles.

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void tq_compute_qk(
    const T* k_smem,
    const vec_t<float, vec_size>& q_vec,
    uint32_t iter_base, uint32_t iter_bound,
    float sm_scale_log2,
    float* s, state_t<vec_size>& st,
    uint32_t tx, uint32_t tz
) {
    float m_prev = st.m;
    #pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
        vec_t<float, vec_size> k_vec;
        k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);

        s[j] = 0.f;
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            s[j] += q_vec[i] * k_vec[i];
        }
        #pragma unroll
        for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
            s[j] += flashinfer::math::shfl_xor_sync(s[j], offset);
        }

        s[j] *= sm_scale_log2;
        s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -flashinfer::math::inf;
        st.m = max(st.m, s[j]);
    }

    float o_scale = flashinfer::math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
    #pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
        s[j] = flashinfer::math::ptx_exp2(s[j] - st.m);
        st.d += s[j];
    }
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        st.o[i] = st.o[i] * o_scale;
    }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void tq_update_local_state(
    const T* v_smem, const float* s,
    state_t<vec_size>& st, uint32_t tx
) {
    #pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
        vec_t<float, vec_size> v_vec;
        v_vec.cast_load(v_smem + (j * bdx + tx) * vec_size);
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            st.o[i] = st.o[i] + s[j] * v_vec[i];
        }
    }
}

// ── Load + dequant one KV tile into shared memory ────────────────────

template <uint32_t tile_size_per_bdx, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename IdType>
__device__ __forceinline__ void load_dequant_kv_tile(
    const paged_kv_turbo_t<IdType>& paged_kv,
    const uint8_t* kv_quant_data,  // k_quant or v_quant
    const __half* kv_norms,        // k_norms or v_norms
    __half* smem,                  // shared memory destination (fp16)
    uint32_t head_dim,             // actual head_dim (e.g. 128)
    uint32_t kv_head_idx,
    uint32_t chunk_idx,            // which 64-dim chunk
    uint32_t packed_page_iter_base,
    uint32_t chunk_start_token,    // start token within this KV chunk
    uint32_t chunk_size,           // total tokens in this chunk
    float codebook_scale,
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t kQuantBytes = paged_kv_turbo_t<IdType>::kQuantBytesPerChunk;
    constexpr uint32_t kTileDims = paged_kv_turbo_t<IdType>::kTileDims;

    // Each thread handles a subset of the tile_size_per_bdx tokens
    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t token_offset = ((j * bdz + tz) * bdy + ty) * tile_size_per_bdx;
        // Wait — this addressing needs to match FlashInfer's tile layout.
        // In FlashInfer: each (tz, ty, j) combination maps to one KV row.
        // The row index within the tile is: (tz * bdy + ty) * tile_size_per_bdx + j
        uint32_t row_in_tile = (tz * bdy + ty) * tile_size_per_bdx + j;
        uint32_t token_idx = chunk_start_token + row_in_tile;

        if (token_idx < chunk_size) {
            // Compute page and entry from packed iterator
            uint32_t packed = packed_page_iter_base + row_in_tile;
            uint32_t page_iter, entry_idx;
            paged_kv.page_size.divmod(packed, page_iter, entry_idx);

            // Load quantized bytes (28 bytes per token per chunk)
            size_t quant_off = paged_kv.protective_get_quant_offset(
                page_iter, kv_head_idx, entry_idx, chunk_idx, last_indptr);
            const uint8_t* qdata = kv_quant_data + quant_off;

            // Load norm
            size_t norm_off = paged_kv.protective_get_norm_offset(
                page_iter, kv_head_idx, entry_idx, chunk_idx, last_indptr);
            float norm = __half2float(kv_norms[norm_off]);

            // Parallel dequant: each of bdx threads writes its 8-dim slice
            paged_kv_turbo_t<IdType>::dequant_chunk_parallel(
                qdata, norm, codebook_scale,
                smem + row_in_tile * kTileDims, tx
            );
        } else {
            // Zero fill for out-of-bounds (each thread zeros its own slice)
            __half* dst = smem + row_in_tile * kTileDims + tx * 8;
            #pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                dst[i] = __float2half(0.0f);
            }
        }
    }
}

// ── Main decode kernel with TurboQuant ───────────────────────────────

// Simplified single-chunk decode (no partition-kv).
// One block handles one (batch, kv_head) pair.
// Template params match FlashInfer conventions.

template <uint32_t HEAD_DIM, uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          uint32_t tile_size_per_bdx, typename IdType>
__global__ void BatchDecodeWithTurboQuantKVKernel(
    const paged_kv_turbo_t<IdType> paged_kv,  // passed by value from host
    const __half* __restrict__ q,              // [batch_size, num_qo_heads, head_dim] (UNROTATED)
    __half* __restrict__ o,                    // [batch_size, num_qo_heads, head_dim] (UNROTATED output)
    float* __restrict__ lse,                   // [batch_size, num_qo_heads] or nullptr
    const float* __restrict__ signs,           // [padded_dim] Hadamard rotation signs (nullptr = no rotation)
    uint32_t num_qo_heads,
    float sm_scale
) {
    auto block = cg::this_thread_block();
    constexpr uint32_t kTileDims = 64;

    uint32_t batch_idx = blockIdx.x;
    uint32_t kv_head_idx = blockIdx.y;
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    uint32_t qo_head_idx = kv_head_idx * bdy + ty;

    if (batch_idx >= paged_kv.batch_size) return;

    uint32_t kv_len = paged_kv.get_length(batch_idx);
    if (kv_len == 0) return;

    uint32_t dim_chunks = paged_kv.dim_chunks;
    uint32_t padded_dim = paged_kv.padded_dim;

    float sm_scale_log2 = sm_scale * flashinfer::math::log2e;
    float codebook_scale = rsqrtf(static_cast<float>(padded_dim));

    // Shared memory: K smem + V smem (each holds one tile of fp16 KV)
    // Tile = tile_size_per_bdx * bdy * bdz tokens × kTileDims dims
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    extern __shared__ uint8_t smem_raw[];
    __half* k_smem = reinterpret_cast<__half*>(smem_raw);
    __half* v_smem = k_smem + tile_tokens * kTileDims;
    float* smem_md = reinterpret_cast<float*>(v_smem + tile_tokens * kTileDims);

    // Pre-load and rotate Q (once per decode, stored in registers per chunk)
    constexpr uint32_t MAX_CHUNKS = HEAD_DIM / kTileDims;
    float q_rotated[MAX_CHUNKS][vec_size];
    for (uint32_t chunk = 0; chunk < MAX_CHUNKS; chunk++) {
        uint32_t q_dim = chunk * kTileDims + tx * vec_size;
        if (q_dim < HEAD_DIM) {
            // Load raw Q
            vec_t<float, vec_size> q_raw;
            q_raw.cast_load(q + batch_idx * num_qo_heads * HEAD_DIM
                              + qo_head_idx * HEAD_DIM + q_dim);
            for (uint32_t i = 0; i < vec_size; i++) q_rotated[chunk][i] = q_raw[i];
        } else {
            for (uint32_t i = 0; i < vec_size; i++) q_rotated[chunk][i] = 0.f;
        }
        // TODO: In-kernel FWHT rotation (needs debugging).
        // For now, caller must pre-rotate Q and post-un-rotate output.
        (void)signs;
    }

    // Output accumulator per chunk
    float o_acc[MAX_CHUNKS][vec_size];
    for (uint32_t c = 0; c < MAX_CHUNKS; c++)
        for (uint32_t i = 0; i < vec_size; i++)
            o_acc[c][i] = 0.f;

    float m_global = -flashinfer::math::inf;
    float d_global = 0.f;

    IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];
    uint32_t num_tile_iters = (kv_len + tile_tokens - 1) / tile_tokens;
    uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size;

    for (uint32_t tile_iter = 0; tile_iter < num_tile_iters; tile_iter++) {
        uint32_t tile_start = tile_iter * tile_tokens;

        // === QK scores: accumulate dot products across all dim chunks ===
        float s[bdy * tile_size_per_bdx];
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; j++) s[j] = 0.f;

        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            load_dequant_kv_tile<tile_size_per_bdx, bdx, bdy, bdz>(
                paged_kv, paged_kv.k_quant, paged_kv.k_norms, k_smem,
                HEAD_DIM, kv_head_idx, chunk,
                packed_page_iter_base + tile_start,
                tile_start, kv_len,
                codebook_scale, last_indptr,
                tx, ty, tz
            );
            block.sync();

            // Use pre-loaded, pre-rotated Q for this chunk
            float* q_chunk_ptr = q_rotated[chunk];

            #pragma unroll
            for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
                vec_t<float, vec_size> k_vec;
                k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
                float partial = 0.f;
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    partial += q_chunk_ptr[i] * k_vec[i];
                }
                #pragma unroll
                for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
                    partial += flashinfer::math::shfl_xor_sync(partial, offset);
                }
                s[j] += partial;
            }
            block.sync();
        }

        // === Online softmax update ===
        float m_prev = m_global;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] *= sm_scale_log2;
            s[j] = (tile_start + j < kv_len)
                    ? s[j] : -flashinfer::math::inf;
            m_global = max(m_global, s[j]);
        }

        float o_scale = flashinfer::math::ptx_exp2(m_prev - m_global);
        d_global *= o_scale;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] = flashinfer::math::ptx_exp2(s[j] - m_global);
            d_global += s[j];
        }
        // Scale existing output accumulators
        for (uint32_t c = 0; c < MAX_CHUNKS; c++)
            for (uint32_t i = 0; i < vec_size; i++)
                o_acc[c][i] *= o_scale;

        // === V accumulation: each chunk writes to its own output slice ===
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t v_dim_start = chunk * kTileDims;
            if (v_dim_start + tx * vec_size >= HEAD_DIM) continue;

            load_dequant_kv_tile<tile_size_per_bdx, bdx, bdy, bdz>(
                paged_kv, paged_kv.v_quant, paged_kv.v_norms, v_smem,
                HEAD_DIM, kv_head_idx, chunk,
                packed_page_iter_base + tile_start,
                tile_start, kv_len,
                codebook_scale, last_indptr,
                tx, ty, tz
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

    // Final normalization
    float d_rcp = (m_global != -flashinfer::math::inf) ? flashinfer::math::ptx_rcp(d_global) : 0.f;
    for (uint32_t c = 0; c < MAX_CHUNKS; c++)
        for (uint32_t i = 0; i < vec_size; i++)
            o_acc[c][i] *= d_rcp;

    // TODO: In-kernel inverse Hadamard rotation (needs debugging).
    // For now, caller handles un-rotation in Python.

    // Write output (now in original, un-rotated space)
    if (tz == 0) {
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t o_dim_start = chunk * kTileDims;
            if (o_dim_start + tx * vec_size >= HEAD_DIM) continue;

            __half* o_ptr = o + (batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM
                              + o_dim_start + tx * vec_size;
            #pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) {
                o_ptr[i] = __float2half(o_acc[chunk][i]);
            }
        }
        if (lse != nullptr) {
            lse[batch_idx * num_qo_heads + qo_head_idx] =
                m_global + __logf(d_global);
        }
    }
}

}  // namespace turboquant
