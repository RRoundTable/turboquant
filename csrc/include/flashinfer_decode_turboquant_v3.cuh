#pragma once
// TurboQuant decode v3: cp_async staged pipeline.
//
// Two-phase load: cp_async packed bytes to smem staging (HW DMA, overlaps compute),
// then short ALU dequant from staging to fp16 smem.
//
// Pipeline: overlap VRAM→staging transfer with previous tile's compute.
// Only the ~25-cycle dequant remains serial per tile.
//
// Reuses FlashInfer's compute_qk, update_local_state, sync_state.

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/cp_async.cuh"
#include "turboquant/page_turbo.cuh"
#include "turboquant/flashinfer_dequant_load.cuh"

namespace flashinfer {

// ─── cp_async helpers for packed byte staging ───────────────────────

// Issue cp_async to load one tile's packed bytes into smem staging.
// Each thread loads one 16-byte segment. Total threads = tile_tokens * segments_per_row.
// Uses flat thread mapping: flat_tid → (row, segment).
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void cp_async_packed_tile(
    uint8_t* staging,                     // [tile_tokens, dim_chunks * 32] in smem
    const turboquant::paged_kv_turbo_t<IdType>& paged_kv,
    const uint8_t* kv_data,               // k_quant or v_quant
    uint32_t kv_head_idx,
    uint32_t packed_page_iter_base,
    uint32_t chunk_size,                  // remaining tokens in this chunk
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;  // packed bytes per token
    constexpr uint32_t segments_per_row = bytes_per_row / 16;  // 16-byte segments
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;

    uint32_t flat_tid = (tz * bdy + ty) * bdx + tx;
    uint32_t row = flat_tid / segments_per_row;
    uint32_t segment = flat_tid % segments_per_row;
    uint32_t chunk_idx = segment / 2;  // which 64-dim chunk
    uint32_t half = segment % 2;       // which 16-byte half of chunk

    bool valid = (row < tile_tokens) && (row < chunk_size);

    const uint8_t* src = kv_data;  // default, overwritten if valid
    if (valid) {
        uint32_t packed = packed_page_iter_base + row;
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed, page_iter, entry_idx);
        if (page_iter >= last_indptr) {
            valid = false;
        } else {
            size_t q_off = paged_kv.get_quant_offset(
                __ldg(paged_kv.indices + page_iter), kv_head_idx, entry_idx, chunk_idx);
            src = kv_data + q_off + half * 16;
        }
    }

    uint8_t* dst = staging + row * bytes_per_row + segment * 16;

    cp_async::pred_load<128, cp_async::PrefetchMode::kPrefetch,
                        cp_async::SharedMemFillMode::kFillZero>(
        dst, src, valid);
}

// Dequant from staging (packed bytes in smem) to fp16 smem.
// Norms are loaded directly from VRAM (2 bytes each, L2 cached).
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void dequant_staging_to_fp16(
    __half* fp16_smem,                    // [tile_tokens, head_dim] output
    const uint8_t* staging,               // [tile_tokens, dim_chunks * 32] input
    const turboquant::paged_kv_turbo_t<IdType>& paged_kv,
    const __half* kv_norms,
    uint32_t kv_head_idx,
    uint32_t packed_page_iter_base,
    uint32_t chunk_size,
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t CHUNK_DIMS = 64;
    constexpr uint32_t dim_chunks = head_dim / CHUNK_DIMS;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;
    float codebook_scale = rsqrtf(static_cast<float>(paged_kv.padded_dim));

    uint32_t chunk_idx = tx / 8;
    uint32_t inner_tx = tx % 8;

    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t row = (tz * bdy + ty) * tile_size_per_bdx + j;
        bool valid = (row < chunk_size) && (chunk_idx < dim_chunks);

        float ns = 0.0f;
        if (valid) {
            // Load norm from VRAM (2 bytes, should be L2 cached)
            uint32_t packed = packed_page_iter_base + row;
            uint32_t page_iter, entry_idx;
            paged_kv.page_size.divmod(packed, page_iter, entry_idx);
            if (page_iter < last_indptr) {
                size_t n_off = paged_kv.get_norm_offset(
                    __ldg(paged_kv.indices + page_iter), kv_head_idx, entry_idx, chunk_idx);
                ns = codebook_scale * __half2float(kv_norms[n_off]);
            } else {
                valid = false;
            }
        }

        // Read packed bytes from staging (smem → registers, fast)
        const uint8_t* src = staging + row * bytes_per_row + chunk_idx * 32 + inner_tx * 4;
        __half* dst = fp16_smem + row * head_dim + chunk_idx * CHUNK_DIMS + inner_tx * 8;

        if (valid) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                uint8_t p = src[i];
                dst[i * 2]     = __float2half(turboquant::kCodebook4bit[(p >> 4) & 0x0F] * ns);
                dst[i * 2 + 1] = __float2half(turboquant::kCodebook4bit[p & 0x0F] * ns);
            }
        } else if (chunk_idx < dim_chunks) {
            #pragma unroll
            for (uint32_t i = 0; i < 8; i++) dst[i] = __float2half(0.0f);
        }
    }
}

// ─── v3 kernel: cp_async staged pipeline ────────────────────────────

template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__device__ __inline__ void TurboQuantPagedDecodeDeviceV3(
    const Params& params,
    uint8_t smem[],
    const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y,
    const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y,
    const uint32_t tz = threadIdx.z
) {
    auto block = cg::this_thread_block();
    using DTypeKV = __half;
    using IdType = typename Params::IdType;

    constexpr uint32_t head_dim = bdx * vec_size;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t staging_bytes = tile_tokens * dim_chunks * 32;

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

    // Shared memory layout:
    //   k_fp16:  [tile_tokens × head_dim] __half
    //   v_fp16:  [tile_tokens × head_dim] __half
    //   staging: [tile_tokens × dim_chunks × 32] uint8  (16-byte aligned)
    //   smem_md: [bdy × bdz × 2] float
    constexpr uint32_t fp16_tile_bytes = tile_tokens * head_dim * sizeof(DTypeKV);

    DTypeKV* k_smem = (DTypeKV*)smem;
    DTypeKV* v_smem = (DTypeKV*)(smem + fp16_tile_bytes);
    uint8_t* staging = (uint8_t*)(smem + 2 * fp16_tile_bytes);
    // Align staging to 16 bytes
    staging = (uint8_t*)(((uintptr_t)staging + 15) & ~15ULL);
    float* smem_md = (float*)(staging + staging_bytes);

    // Load Q
    vec_t<float, vec_size> q_vec;
    vec_t<float, vec_size> freq;
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kNone) {
        q_vec.cast_load(params.q + batch_idx * params.q_stride_n +
                        qo_head_idx * params.q_stride_h + tx * vec_size);
    }
    block.sync();

    state_t<vec_size> st;
    float s[bdy * tile_size_per_bdx];

    uint32_t packed_page_iter_base = tq_kv.indptr[batch_idx] * tq_kv.page_size + chunk_start;
    IdType last_indptr = tq_kv.indptr[tq_kv.batch_size];
    const uint32_t num_iters = ceil_div(chunk_size, tile_tokens);

    if (num_iters == 0) return;

    // ═══ Preload: cp_async K[0] packed bytes → staging ═══
    cp_async_packed_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
        staging, tq_kv, tq_kv.k_quant,
        kv_head_idx, packed_page_iter_base,
        chunk_size, last_indptr, tx, ty, tz);
    cp_async::commit_group();

    // ═══ Main loop ═══
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t iter_base = packed_page_iter_base + iter * tile_tokens;
        uint32_t remaining = chunk_size - iter * tile_tokens;

        // --- Wait for K packed data in staging ---
        cp_async::wait_group<0>();
        block.sync();

        // --- Dequant K: staging → k_fp16 (norms from VRAM/L2) ---
        dequant_staging_to_fp16<head_dim, tile_size_per_bdx, bdy, bdz>(
            k_smem, staging, tq_kv, tq_kv.k_norms,
            kv_head_idx, iter_base, remaining, last_indptr, tx, ty, tz);
        block.sync();

        // --- QK compute + cp_async V[iter] packed bytes (OVERLAP!) ---
        compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
            params, variant, batch_idx,
            k_smem + tz * bdy * tile_size_per_bdx * head_dim,
            q_vec, freq,
            chunk_start + iter * tile_tokens,
            iter * tile_tokens, chunk_size,
            qo_head_idx, kv_head_idx, s, st, tx, ty, tz);

        // Issue V packed load (overlaps with QK compute above)
        cp_async_packed_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
            staging, tq_kv, tq_kv.v_quant,
            kv_head_idx, iter_base,
            remaining, last_indptr, tx, ty, tz);
        cp_async::commit_group();
        block.sync();

        // --- Wait for V packed data ---
        cp_async::wait_group<0>();
        block.sync();

        // --- Dequant V: staging → v_fp16 ---
        dequant_staging_to_fp16<head_dim, tile_size_per_bdx, bdy, bdz>(
            v_smem, staging, tq_kv, tq_kv.v_norms,
            kv_head_idx, iter_base, remaining, last_indptr, tx, ty, tz);
        block.sync();

        // --- V accumulate + cp_async K[iter+1] packed bytes (OVERLAP!) ---
        update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
            v_smem + tz * bdy * tile_size_per_bdx * head_dim,
            s, 0, st, tx);

        // Issue next K packed load (overlaps with V accumulate above)
        if (iter + 1 < num_iters) {
            cp_async_packed_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
                staging, tq_kv, tq_kv.k_quant,
                kv_head_idx, packed_page_iter_base + (iter + 1) * tile_tokens,
                chunk_size - (iter + 1) * tile_tokens, last_indptr, tx, ty, tz);
            cp_async::commit_group();
        }
        block.sync();
    }

    // ═══ Drain pipeline ═══
    cp_async::wait_group<0>();
    block.sync();

    // ═══ Cross-warp merge ═══
    sync_state<vec_size, bdx, bdy, bdz>(
        variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);

    // ═══ Output ═══
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
__global__ void TurboQuantPagedDecodeKernelV3(const __grid_constant__ Params params) {
    extern __shared__ uint8_t smem[];
    TurboQuantPagedDecodeDeviceV3<POS_ENCODING_MODE, tile_size_per_bdx,
                                  vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

}  // namespace flashinfer
