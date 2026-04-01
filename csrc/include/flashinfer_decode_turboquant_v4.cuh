#pragma once
// TurboQuant decode v4: fused inline dequant — no fp16 smem buffer.
//
// Eliminates the fp16 intermediate: packed bytes stay in staging smem,
// dequant happens inline during QK/V compute (directly to float registers).
//
// Benefits:
//   - No fp16 smem write/read (saves ~8KB per tile)
//   - No float↔half conversion (dequant produces float directly)
//   - ~7× less smem → higher occupancy → better latency hiding
//   - cp_async overlaps VRAM load of packed bytes with norm precompute
//
// Trade-off: can't reuse FlashInfer's compute_qk/update_local_state.
// QK/V code is reimplemented with inline dequant.

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/cp_async.cuh"
#include "turboquant/page_turbo.cuh"
#include "turboquant/flashinfer_dequant_load.cuh"
#include "flashinfer_decode_turboquant_v3.cuh"  // for cp_async_packed_tile

namespace flashinfer {

// ─── Inline dequant: read packed bytes from smem, return float ──────

// Read 8 float values from packed staging smem for one thread's slice.
// No fp16 conversion — dequant goes directly to float.
__device__ __forceinline__ void inline_dequant_8(
    float* out,                   // [8] output floats
    const uint8_t* staging_row,   // packed bytes for this row+chunk (32 bytes)
    uint32_t inner_tx,            // 0..7
    float norm_s                  // precomputed codebook_scale * norm
) {
    const uint8_t* src = staging_row + inner_tx * 4;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint8_t p = src[i];
        out[i * 2]     = turboquant::kCodebook4bit[(p >> 4) & 0x0F] * norm_s;
        out[i * 2 + 1] = turboquant::kCodebook4bit[p & 0x0F] * norm_s;
    }
}

// ─── Precompute page info + norms into smem ─────────────────────────

// Cooperatively precompute norm*scale for each (row, chunk) in the tile.
// Also prefetches the quant base offset.
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void precompute_norms(
    float* smem_norms,            // [tile_tokens, dim_chunks] output
    size_t* smem_quant_offsets,   // [tile_tokens, dim_chunks] output
    const turboquant::paged_kv_turbo_t<IdType>& paged_kv,
    const uint8_t* kv_data,
    const __half* kv_norms,
    uint32_t kv_head_idx,
    uint32_t packed_page_iter_base,
    uint32_t chunk_size,
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    constexpr uint32_t total_pairs = tile_tokens * dim_chunks;
    float codebook_scale = rsqrtf(static_cast<float>(paged_kv.padded_dim));

    uint32_t flat_tid = (tz * bdy + ty) * bdx + tx;

    // Each thread handles one or more (row, chunk) pairs
    for (uint32_t pair = flat_tid; pair < total_pairs; pair += bdx * bdy * bdz) {
        uint32_t row = pair / dim_chunks;
        uint32_t chunk = pair % dim_chunks;

        bool valid = row < chunk_size;
        float ns = 0.0f;
        size_t qoff = 0;

        if (valid) {
            uint32_t packed = packed_page_iter_base + row;
            uint32_t page_iter, entry_idx;
            paged_kv.page_size.divmod(packed, page_iter, entry_idx);
            if (page_iter < last_indptr) {
                IdType page_idx = __ldg(paged_kv.indices + page_iter);
                size_t n_off = paged_kv.get_norm_offset(page_idx, kv_head_idx, entry_idx, chunk);
                ns = codebook_scale * __half2float(kv_norms[n_off]);
                qoff = paged_kv.get_quant_offset(page_idx, kv_head_idx, entry_idx, chunk);
            }
        }

        smem_norms[row * dim_chunks + chunk] = ns;
        smem_quant_offsets[row * dim_chunks + chunk] = qoff;
    }
}

// ─── v4 kernel: fused inline dequant ────────────────────────────────

template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__device__ __inline__ void TurboQuantPagedDecodeDeviceV4(
    const Params& params,
    uint8_t smem[],
    const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y,
    const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y,
    const uint32_t tz = threadIdx.z
) {
    auto block = cg::this_thread_block();
    using IdType = typename Params::IdType;

    constexpr uint32_t head_dim = bdx * vec_size;
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;
    constexpr uint32_t tile_size = bdy * tile_size_per_bdx;  // rows per tz group

    const uint32_t num_qo_heads = params.num_qo_heads;
    const auto& tq_kv = params.tq_kv;

    const uint32_t batch_idx = params.request_indices[bx];
    const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
    const uint32_t kv_head_idx = by;
    const uint32_t qo_head_idx = kv_head_idx * bdy + ty;

    if (params.block_valid_mask && !params.block_valid_mask[bx]) return;

    const uint32_t kv_chunk_size_val = *(params.kv_chunk_size_ptr);
    const uint32_t kv_len = tq_kv.get_length(batch_idx);
    const uint32_t max_chunk_size = params.partition_kv ? kv_chunk_size_val : kv_len;
    const uint32_t chunk_start = params.partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end = params.partition_kv ?
        min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    AttentionVariant variant(params, batch_idx, smem);

    // ═══ Shared memory layout ═══
    // sync_state needs [bdz * bdy * head_dim] floats + [bdz * bdy * 2] floats at smem base.
    // During main loop, we use: staging + norms + offsets (much smaller).
    // Allocate max(sync_state_need, main_loop_need).
    constexpr uint32_t staging_bytes = ((tile_tokens * bytes_per_row + 15) / 16) * 16;
    constexpr uint32_t norms_bytes = tile_tokens * dim_chunks * sizeof(float);
    constexpr uint32_t offsets_bytes = tile_tokens * dim_chunks * sizeof(size_t);
    constexpr uint32_t main_loop_bytes = staging_bytes + norms_bytes + offsets_bytes;
    constexpr uint32_t sync_state_bytes = bdz * bdy * (head_dim + 2) * sizeof(float);
    // smem_md goes after the larger of main_loop or sync_state usage
    constexpr uint32_t md_offset = (main_loop_bytes > sync_state_bytes) ? main_loop_bytes : sync_state_bytes;

    uint8_t* staging = smem;
    float* smem_norms = (float*)(smem + staging_bytes);
    size_t* smem_qoffsets = (size_t*)((uint8_t*)smem_norms + norms_bytes);
    float* smem_md = (float*)(smem + md_offset);

    // ═══ Load Q ═══
    vec_t<float, vec_size> q_vec;
    vec_t<float, vec_size> freq;
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kNone) {
        q_vec.cast_load(params.q + batch_idx * params.q_stride_n +
                        qo_head_idx * params.q_stride_h + tx * vec_size);
    }
    block.sync();

    // ═══ State ═══
    state_t<vec_size> st;
    float s[tile_size];

    uint32_t packed_page_iter_base = tq_kv.indptr[batch_idx] * tq_kv.page_size + chunk_start;
    IdType last_indptr = tq_kv.indptr[tq_kv.batch_size];
    const uint32_t num_iters = ceil_div(chunk_size, tile_tokens);

    uint32_t chunk_idx = tx / 8;
    uint32_t inner_tx = tx % 8;

    // ═══ Main loop ═══
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t iter_base_abs = packed_page_iter_base + iter * tile_tokens;
        uint32_t remaining = chunk_size - iter * tile_tokens;

        // --- Phase 1: cp_async packed K bytes + precompute K norms ---
        // cp_async: each thread loads one 16-byte segment
        cp_async_packed_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
            staging, tq_kv, tq_kv.k_quant,
            kv_head_idx, iter_base_abs,
            remaining, last_indptr, tx, ty, tz);
        cp_async::commit_group();

        // Precompute norms (overlaps with cp_async in-flight)
        precompute_norms<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
            smem_norms, smem_qoffsets, tq_kv, tq_kv.k_quant, tq_kv.k_norms,
            kv_head_idx, iter_base_abs, remaining, last_indptr, tx, ty, tz);

        cp_async::wait_group<0>();
        block.sync();

        // --- Phase 2: QK with inline dequant from staging ---
        {
            float m_prev = st.m;
            #pragma unroll
            for (uint32_t j = 0; j < tile_size; ++j) {
                uint32_t row = tz * tile_size + j;

                // Inline dequant: packed bytes from staging → float k_vec
                float k_vals[vec_size];
                if (chunk_idx < dim_chunks) {
                    float ns = smem_norms[row * dim_chunks + chunk_idx];
                    inline_dequant_8(k_vals, staging + row * bytes_per_row + chunk_idx * 32,
                                     inner_tx, ns);
                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < vec_size; i++) k_vals[i] = 0.f;
                }

                // QK dot product
                s[j] = 0.f;
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    s[j] += q_vec[i] * k_vals[i];
                }

                // Warp reduce across bdx threads
                #pragma unroll
                for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
                    s[j] += math::shfl_xor_sync(s[j], offset);
                }

                // LogitsTransform + softmax scaling
                const uint32_t pos = chunk_start + iter * tile_tokens + tz * tile_size + j;
                s[j] = variant.LogitsTransform(params, s[j], batch_idx, 0, pos,
                                                qo_head_idx, kv_head_idx);
                if constexpr (AttentionVariant::use_softmax) {
                    s[j] *= variant.sm_scale_log2;
                }

                bool mask = variant.LogitsMask(params, batch_idx, 0, pos,
                                                qo_head_idx, kv_head_idx);
                s[j] = (iter * tile_tokens + tz * tile_size + j < chunk_size && mask)
                       ? s[j] : -math::inf;
                st.m = max(st.m, s[j]);
            }

            // Online softmax update
            if constexpr (AttentionVariant::use_softmax) {
                float o_scale = math::ptx_exp2(m_prev - st.m);
                st.d *= o_scale;
                #pragma unroll
                for (uint32_t j = 0; j < tile_size; ++j) {
                    s[j] = math::ptx_exp2(s[j] - st.m);
                    st.d += s[j];
                }
                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    st.o[i] *= o_scale;
                }
            }
        }
        block.sync();

        // --- Phase 3: cp_async packed V bytes + precompute V norms ---
        cp_async_packed_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
            staging, tq_kv, tq_kv.v_quant,
            kv_head_idx, iter_base_abs,
            remaining, last_indptr, tx, ty, tz);
        cp_async::commit_group();

        precompute_norms<head_dim, tile_size_per_bdx, bdx, bdy, bdz>(
            smem_norms, smem_qoffsets, tq_kv, tq_kv.v_quant, tq_kv.v_norms,
            kv_head_idx, iter_base_abs, remaining, last_indptr, tx, ty, tz);

        cp_async::wait_group<0>();
        block.sync();

        // --- Phase 4: V accumulate with inline dequant ---
        {
            #pragma unroll
            for (uint32_t j = 0; j < tile_size; ++j) {
                uint32_t row = tz * tile_size + j;

                float v_vals[vec_size];
                if (chunk_idx < dim_chunks) {
                    float ns = smem_norms[row * dim_chunks + chunk_idx];
                    inline_dequant_8(v_vals, staging + row * bytes_per_row + chunk_idx * 32,
                                     inner_tx, ns);
                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < vec_size; i++) v_vals[i] = 0.f;
                }

                #pragma unroll
                for (uint32_t i = 0; i < vec_size; ++i) {
                    st.o[i] += s[j] * v_vals[i];
                }
            }
        }
        block.sync();
    }

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
__global__ void TurboQuantPagedDecodeKernelV4(const __grid_constant__ Params params) {
    extern __shared__ uint8_t smem[];
    TurboQuantPagedDecodeDeviceV4<POS_ENCODING_MODE, tile_size_per_bdx,
                                  vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

}  // namespace flashinfer
