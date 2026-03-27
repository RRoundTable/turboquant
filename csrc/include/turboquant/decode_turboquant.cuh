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

            // Dequant into shared memory
            // smem layout: [tile_rows, head_dim] in fp16
            // This thread writes the full 64-dim chunk for this row
            // Only thread tx==0 does the dequant (sequential per row, parallelized across rows)
            if (tx == 0) {
                paged_kv_turbo_t<IdType>::dequant_chunk_to_smem(
                    qdata, norm, codebook_scale,
                    smem + row_in_tile * kTileDims
                );
            }
        } else {
            // Zero fill for out-of-bounds
            if (tx == 0) {
                for (uint32_t d = 0; d < kTileDims; d++) {
                    smem[row_in_tile * kTileDims + d] = __float2half(0.0f);
                }
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
    const __half* __restrict__ q,              // [batch_size, num_qo_heads, head_dim]
    __half* __restrict__ o,                    // [batch_size, num_qo_heads, head_dim]
    float* __restrict__ lse,                   // [batch_size, num_qo_heads] or nullptr
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

    // Load query into registers
    vec_t<float, vec_size> q_vec;
    q_vec.cast_load(q + batch_idx * num_qo_heads * HEAD_DIM + qo_head_idx * HEAD_DIM + tx * vec_size);

    // Iterate over KV chunks
    // For head_dim > 64, we process each dim_chunk and accumulate QK across chunks
    // Actually — for the QK dot product, we need the FULL head_dim, not chunks.
    // But KV is stored in 64-dim chunks. So we need to accumulate across chunks.
    //
    // Approach: for each KV token tile, iterate over dim_chunks:
    //   partial_s[j] += dot(q_chunk, k_chunk)  for each chunk
    // Then do softmax on the accumulated s[j], then accumulate V similarly.

    state_t<vec_size> st;
    IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t num_tile_iters = (kv_len + tile_tokens - 1) / tile_tokens;
    uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size;

    for (uint32_t tile_iter = 0; tile_iter < num_tile_iters; tile_iter++) {
        uint32_t tile_start = tile_iter * tile_tokens;
        uint32_t tile_end = min(tile_start + tile_tokens, kv_len);

        // Accumulate QK scores across dim chunks
        float s[bdy * tile_size_per_bdx];
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; j++) s[j] = 0.f;

        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            // Load + dequant K tile for this chunk
            load_dequant_kv_tile<tile_size_per_bdx, bdx, bdy, bdz>(
                paged_kv, paged_kv.k_quant, paged_kv.k_norms, k_smem,
                HEAD_DIM, kv_head_idx, chunk,
                packed_page_iter_base + tile_start,
                tile_start, kv_len,
                codebook_scale, last_indptr,
                tx, ty, tz
            );
            block.sync();

            // Partial QK: accumulate dot products for this chunk's dims
            uint32_t q_dim_start = chunk * kTileDims;
            vec_t<float, vec_size> q_chunk;
            // Load the appropriate q slice
            if (q_dim_start + tx * vec_size < HEAD_DIM) {
                q_chunk.cast_load(q + batch_idx * num_qo_heads * HEAD_DIM
                                    + qo_head_idx * HEAD_DIM
                                    + q_dim_start + tx * vec_size);
            } else {
                for (uint32_t i = 0; i < vec_size; i++) q_chunk[i] = 0.f;
            }

            // Dot product with K from smem
            #pragma unroll
            for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
                vec_t<float, vec_size> k_vec;
                k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
                float partial = 0.f;
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    partial += q_chunk[i] * k_vec[i];
                }
                // Reduce across bdx threads
                #pragma unroll
                for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
                    partial += flashinfer::math::shfl_xor_sync(partial, offset);
                }
                s[j] += partial;
            }
            block.sync();
        }

        // Apply scale and softmax update
        float m_prev = st.m;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] *= sm_scale_log2;
            s[j] = (tile_start + (tz * bdy + ty) * tile_size_per_bdx + j < kv_len)
                    ? s[j] : -flashinfer::math::inf;
            st.m = max(st.m, s[j]);
        }

        float o_scale = flashinfer::math::ptx_exp2(m_prev - st.m);
        st.d *= o_scale;
        #pragma unroll
        for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
            s[j] = flashinfer::math::ptx_exp2(s[j] - st.m);
            st.d += s[j];
        }
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            st.o[i] *= o_scale;
        }

        // Accumulate V across dim chunks (only the dims corresponding to output)
        // Output dims = the original head_dim, covered by tx * vec_size
        for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
            uint32_t v_dim_start = chunk * kTileDims;
            if (v_dim_start + tx * vec_size >= HEAD_DIM) continue;

            // Load + dequant V tile for this chunk
            load_dequant_kv_tile<tile_size_per_bdx, bdx, bdy, bdz>(
                paged_kv, paged_kv.v_quant, paged_kv.v_norms, v_smem,
                HEAD_DIM, kv_head_idx, chunk,
                packed_page_iter_base + tile_start,
                tile_start, kv_len,
                codebook_scale, last_indptr,
                tx, ty, tz
            );
            block.sync();

            // Accumulate: o[i] += s[j] * v_vec[i]
            #pragma unroll
            for (uint32_t j = 0; j < bdy * tile_size_per_bdx; ++j) {
                vec_t<float, vec_size> v_vec;
                v_vec.cast_load(v_smem + (j * bdx + tx) * vec_size);
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    st.o[i] += s[j] * v_vec[i];
                }
            }
            block.sync();
        }
    }

    // Sync across warps (bdz > 1)
    if constexpr (bdz > 1) {
        constexpr uint32_t head_dim_val = bdx * vec_size;
        st.o.store(reinterpret_cast<float*>(smem_raw) + (tz * bdy + ty) * head_dim_val + tx * vec_size);
        smem_md[(tz * bdy + ty) * 2] = st.m;
        smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
        block.sync();
        st.init();
        #pragma unroll
        for (uint32_t j = 0; j < bdz; ++j) {
            float mz = smem_md[(j * bdy + ty) * 2];
            float dz = smem_md[(j * bdy + ty) * 2 + 1];
            vec_t<float, vec_size> oz;
            oz.load(reinterpret_cast<float*>(smem_raw) + (j * bdy + ty) * head_dim_val + tx * vec_size);
            st.merge(oz, mz, dz);
        }
    }

    // Final output normalization
    float d_rcp = (st.m != -flashinfer::math::inf) ? flashinfer::math::ptx_rcp(st.d) : 0.f;
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        st.o[i] *= d_rcp;
    }

    // Write output
    if (tz == 0) {
        st.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM + tx * vec_size);
        if (lse != nullptr) {
            lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
        }
    }
}

}  // namespace turboquant
