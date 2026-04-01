#pragma once
// TurboQuant decode v2: inject dequant-load into FlashInfer's decode pipeline.
//
// Uses FlashInfer's compute_qk, update_local_state, sync_state directly.
// Only the KV loading path changes: dequant-load instead of cp_async.
//
// Key design decisions:
//   1. Device function lives in namespace flashinfer to access anonymous-namespace
//      functions (compute_qk, update_local_state, sync_state).
//   2. Supports head_dim=64 (bdx=8) and head_dim=128 (bdx=16) via
//      dequant_load_kv_tile_full which maps tx/8 → chunk_idx.
//   3. TurboQuantBatchDecodeParams carries paged_kv_turbo_t instead of paged_kv_t.
//   4. num_stages_smem=1 (no cp_async pipelining — dequant is synchronous).

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "turboquant/page_turbo.cuh"
#include "turboquant/flashinfer_dequant_load.cuh"

namespace flashinfer {

// Params struct mirroring BatchDecodeParams but with TurboQuant's paged KV.
template <typename DTypeQ_, typename DTypeO_, typename IdType_>
struct TurboQuantBatchDecodeParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = __half;   // smem holds fp16 after dequant
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    DTypeQ* q;
    turboquant::paged_kv_turbo_t<IdType> tq_kv;
    DTypeO* o;
    float* lse;
    float* maybe_alibi_slopes;
    uint32_t padded_batch_size;
    uint32_t num_qo_heads;
    IdType q_stride_n;
    IdType q_stride_h;
    int32_t window_left;
    float logits_soft_cap;
    float sm_scale;
    float rope_rcp_scale;
    float rope_rcp_theta;

    IdType* request_indices;
    IdType* kv_tile_indices;
    IdType* o_indptr;
    IdType* kv_chunk_size_ptr;
    bool* block_valid_mask;
    bool partition_kv;

    __device__ __host__ TurboQuantBatchDecodeParams() {}

    __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }
    __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
        return tq_kv.get_length(batch_idx);
    }
};

// Device function: TurboQuant paged decode using FlashInfer's compute functions.
// Must be in namespace flashinfer to access anonymous-namespace functions.
template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__device__ __inline__ void TurboQuantPagedDecodeDevice(
    const Params& params,
    uint8_t smem[],
    const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y,
    const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y,
    const uint32_t tz = threadIdx.z
) {
    auto block = cg::this_thread_block();
    using DTypeQ = typename Params::DTypeQ;
    using DTypeO = typename Params::DTypeO;
    using DTypeKV = __half;
    using IdType = typename Params::IdType;

    constexpr uint32_t head_dim = bdx * vec_size;
    constexpr uint32_t num_stages_smem = 1;  // no pipelining

    const uint32_t num_qo_heads = params.num_qo_heads;
    const auto& tq_kv = params.tq_kv;

    const uint32_t batch_idx = params.request_indices[bx];
    const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
    const uint32_t kv_head_idx = by;
    const uint32_t qo_head_idx = kv_head_idx * bdy + ty;

    if (params.block_valid_mask && !params.block_valid_mask[bx]) return;

    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
    const uint32_t kv_len = tq_kv.get_length(batch_idx);
    const uint32_t max_chunk_size = params.partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = params.partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end = params.partition_kv ?
        min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    AttentionVariant variant(params, batch_idx, smem);

    // Shared memory layout (same as FlashInfer with num_stages=1)
    DTypeKV* k_smem = (DTypeKV*)smem;
    DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                        head_dim * sizeof(DTypeKV));
    float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                     head_dim * sizeof(DTypeKV));

    // Load Q
    vec_t<float, vec_size> q_vec;
    vec_t<float, vec_size> freq;
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kNone) {
        q_vec.cast_load(params.q + batch_idx * params.q_stride_n +
                        qo_head_idx * params.q_stride_h + tx * vec_size);
    }
    block.sync();

    // Main loop
    state_t<vec_size> st;
    float s[bdy * tile_size_per_bdx];

    uint32_t packed_page_iter_base = tq_kv.indptr[batch_idx] * tq_kv.page_size + chunk_start;
    IdType last_indptr = tq_kv.indptr[tq_kv.batch_size];
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;

    for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_tokens); ++iter) {
        uint32_t remaining = chunk_size - iter * tile_tokens;

        // === Dequant-load K tile (all dim chunks in parallel) ===
        turboquant::dequant_load_kv_tile_full<head_dim, tile_size_per_bdx, bdy, bdz>(
            k_smem, tq_kv, tq_kv.k_quant, tq_kv.k_norms,
            kv_head_idx,
            packed_page_iter_base + iter * tile_tokens,
            remaining, last_indptr, tx, ty, tz
        );
        block.sync();

        // === QK (FlashInfer's function) ===
        compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
            params, variant, batch_idx,
            k_smem + tz * bdy * tile_size_per_bdx * head_dim,
            q_vec, freq,
            chunk_start + iter * tile_tokens,
            iter * tile_tokens, chunk_size,
            qo_head_idx, kv_head_idx, s, st, tx, ty, tz);
        block.sync();

        // === Dequant-load V tile (all dim chunks in parallel) ===
        turboquant::dequant_load_kv_tile_full<head_dim, tile_size_per_bdx, bdy, bdz>(
            v_smem, tq_kv, tq_kv.v_quant, tq_kv.v_norms,
            kv_head_idx,
            packed_page_iter_base + iter * tile_tokens,
            remaining, last_indptr, tx, ty, tz
        );
        block.sync();

        // === V accumulate (FlashInfer's function) ===
        update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
            v_smem + tz * bdy * tile_size_per_bdx * head_dim,
            s, 0, st, tx);
        block.sync();
    }

    // === Cross-warp merge (FlashInfer's function) ===
    sync_state<vec_size, bdx, bdy, bdz>(
        variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);

    // === Output ===
    #pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
        st.o[i] = variant.OutputTransform(params, st.o[i], bx, 0,
                                           qo_head_idx, st.m, st.d, 1.0f);
    }

    if (tz == 0) {
        st.o.cast_store(params.o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
        if (params.lse != nullptr) {
            params.lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
        }
    }
}

// Kernel wrapper
template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__global__ void TurboQuantPagedDecodeKernel(const __grid_constant__ Params params) {
    extern __shared__ uint8_t smem[];
    TurboQuantPagedDecodeDevice<POS_ENCODING_MODE, tile_size_per_bdx,
                                vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

}  // namespace flashinfer
