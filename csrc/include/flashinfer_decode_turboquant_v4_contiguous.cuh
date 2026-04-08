#pragma once
// TurboQuant decode v4 contiguous: fused inline dequant, no paging overhead.
//
// Identical to v4 paged but replaces paged addressing (divmod + page table lookup)
// with direct contiguous addressing. HYP-008 showed 32us paging overhead at
// seq=1024 (89us paged vs 57us contiguous). This kernel eliminates that overhead
// for non-paged KV cache layouts.
//
// Data layout (contiguous, HND):
//   k_quant: [batch, num_kv_heads, seq_len, quant_bytes_per_head] uint8
//   k_norms: [batch, num_kv_heads, seq_len, dim_chunks] fp16
//
// Same inline dequant logic, same bdz=16 thread config, same FlashInfer
// sync_state / state_t / math utilities as v4 paged.

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/cp_async.cuh"
#include "turboquant/page_turbo.cuh"
#include "turboquant/flashinfer_dequant_load.cuh"

namespace flashinfer {

// ---- Contiguous TurboQuant KV cache params --------------------------------
//
// Replaces paged_kv_turbo_t with flat pointer + stride arithmetic.
// No page_size, no divmod, no indices table.
// Includes all fields needed by DefaultAttention constructor so we can
// use the standard AttentionVariant system.
template <typename DTypeQ_, typename DTypeO_, typename IdType_>
struct ContiguousTurboQuantDecodeParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = __half;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    // Quantized KV data (flat, contiguous)
    uint8_t* k_quant;    // [batch, num_kv_heads, seq_len, quant_bytes_per_head]
    uint8_t* v_quant;
    __half*  k_norms;    // [batch, num_kv_heads, seq_len, dim_chunks]
    __half*  v_norms;

    // Geometry
    uint32_t seq_len;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t padded_dim;
    uint32_t dim_chunks;     // padded_dim / 64

    // Quant strides (in bytes): k_quant[b, h, t, :] = k_quant + b*qsb + h*qsh + t*qst
    size_t quant_stride_batch;   // num_kv_heads * seq_len * dim_chunks * 32
    size_t quant_stride_head;    // seq_len * dim_chunks * 32
    size_t quant_stride_token;   // dim_chunks * 32

    // Norm strides (in elements): k_norms[b, h, t, c] = k_norms + b*nsb + h*nsh + t*nst + c
    size_t norm_stride_batch;    // num_kv_heads * seq_len * dim_chunks
    size_t norm_stride_head;     // seq_len * dim_chunks
    size_t norm_stride_token;    // dim_chunks

    // Attention I/O
    DTypeQ*  q;              // [batch, num_qo_heads, padded_dim]
    DTypeO*  o;              // [batch, num_qo_heads, padded_dim]
    float*   lse;            // [batch, num_qo_heads] or nullptr
    float    sm_scale;
    uint32_t num_qo_heads;
    IdType   q_stride_n;     // num_qo_heads * padded_dim
    IdType   q_stride_h;     // padded_dim

    // Fields required by DefaultAttention constructor
    float*   maybe_alibi_slopes;
    float    logits_soft_cap;
    int32_t  window_left;

    // Batch dispatch
    uint32_t batch_size;

    __host__ __device__ ContiguousTurboQuantDecodeParams() {}

    __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }
    __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
        return seq_len;
    }

    __host__ ContiguousTurboQuantDecodeParams(
        uint8_t* k_quant, uint8_t* v_quant,
        __half* k_norms, __half* v_norms,
        uint32_t batch_size, uint32_t num_kv_heads,
        uint32_t seq_len, uint32_t head_dim, uint32_t padded_dim,
        DTypeQ* q, DTypeO* o, float* lse,
        float sm_scale, uint32_t num_qo_heads
    ) : k_quant(k_quant), v_quant(v_quant),
        k_norms(k_norms), v_norms(v_norms),
        batch_size(batch_size), num_kv_heads(num_kv_heads),
        seq_len(seq_len), head_dim(head_dim), padded_dim(padded_dim),
        q(q), o(o), lse(lse), sm_scale(sm_scale), num_qo_heads(num_qo_heads),
        maybe_alibi_slopes(nullptr), logits_soft_cap(0), window_left(-1)
    {
        dim_chunks = padded_dim / 64;
        uint32_t bytes_per_token = dim_chunks * 32;
        quant_stride_token = bytes_per_token;
        quant_stride_head  = (size_t)seq_len * bytes_per_token;
        quant_stride_batch = (size_t)num_kv_heads * quant_stride_head;

        norm_stride_token = dim_chunks;
        norm_stride_head  = (size_t)seq_len * dim_chunks;
        norm_stride_batch = (size_t)num_kv_heads * norm_stride_head;

        q_stride_h = padded_dim;
        q_stride_n = num_qo_heads * padded_dim;
    }
};

// ---- Inline dequant (identical logic to v4 paged) --------------------------

__device__ __forceinline__ void inline_dequant_8_contiguous(
    float* out,                   // [8] output floats
    const uint8_t* staging_row,   // packed bytes for this row+chunk (32 bytes)
    uint32_t inner_tx,            // 0..7
    float norm_s                  // codebook_scale * norm
) {
    const uint8_t* src = staging_row + inner_tx * 4;
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        uint8_t p = src[i];
        out[i * 2]     = turboquant::kCodebook4bit[(p >> 4) & 0x0F] * norm_s;
        out[i * 2 + 1] = turboquant::kCodebook4bit[p & 0x0F] * norm_s;
    }
}

// ---- cp_async contiguous tile: no divmod, no page table --------------------
//
// Each thread loads one 16-byte segment from contiguous KV memory via cp_async.
// Direct address: kv_data + base_offset + token_idx * token_stride + segment * 16
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void cp_async_contiguous_tile(
    uint8_t* staging,                                  // smem staging buffer
    const uint8_t* kv_data,                            // k_quant or v_quant (flat)
    size_t base_offset,                                // batch*batch_stride + head*head_stride
    size_t token_stride,                               // quant_stride_token
    uint32_t token_start,                              // first token index in this tile
    uint32_t chunk_size,                               // remaining tokens
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t bytes_per_row = dim_chunks * 32;
    constexpr uint32_t segments_per_row = bytes_per_row / 16;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;

    uint32_t flat_tid = (tz * bdy + ty) * bdx + tx;
    uint32_t row = flat_tid / segments_per_row;
    uint32_t segment = flat_tid % segments_per_row;

    bool valid = (row < tile_tokens) && (row < chunk_size);

    const uint8_t* src = kv_data;  // default, overwritten if valid
    if (valid) {
        uint32_t token_idx = token_start + row;
        src = kv_data + base_offset + (size_t)token_idx * token_stride + segment * 16;
    }

    uint8_t* dst = staging + row * bytes_per_row + segment * 16;

    cp_async::pred_load<128, cp_async::PrefetchMode::kPrefetch,
                        cp_async::SharedMemFillMode::kFillZero>(
        dst, src, valid);
}

// ---- Precompute norms contiguous: direct address, no divmod ----------------
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdx, uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void precompute_norms_contiguous(
    float* smem_norms,            // [tile_tokens, dim_chunks] output
    const __half* kv_norms,       // flat norm array
    size_t norm_base_offset,      // batch*norm_batch_stride + head*norm_head_stride
    size_t norm_token_stride,     // dim_chunks
    uint32_t token_start,         // first token in this tile
    uint32_t chunk_size,          // remaining tokens
    float codebook_scale,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t dim_chunks = head_dim / 64;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    constexpr uint32_t total_pairs = tile_tokens * dim_chunks;

    uint32_t flat_tid = (tz * bdy + ty) * bdx + tx;

    for (uint32_t pair = flat_tid; pair < total_pairs; pair += bdx * bdy * bdz) {
        uint32_t row = pair / dim_chunks;
        uint32_t chunk = pair % dim_chunks;

        bool valid = (row < chunk_size);
        float ns = 0.0f;

        if (valid) {
            uint32_t token_idx = token_start + row;
            size_t n_off = norm_base_offset + (size_t)token_idx * norm_token_stride + chunk;
            ns = codebook_scale * __half2float(kv_norms[n_off]);
        }

        smem_norms[row * dim_chunks + chunk] = ns;
    }
}

// ---- v4 contiguous kernel: fused inline dequant, no paging -----------------

template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__device__ __inline__ void TurboQuantContiguousDecodeDeviceV4(
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
    constexpr uint32_t tile_size = bdy * tile_size_per_bdx;

    const uint32_t num_qo_heads = params.num_qo_heads;
    const uint32_t batch_idx = bx;
    const uint32_t kv_head_idx = by;
    const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
    const uint32_t seq_len = params.seq_len;

    float codebook_scale = rsqrtf(static_cast<float>(params.padded_dim));

    // Precompute base offsets for this (batch, head)
    size_t quant_base = (size_t)batch_idx * params.quant_stride_batch
                      + (size_t)kv_head_idx * params.quant_stride_head;
    size_t norm_base  = (size_t)batch_idx * params.norm_stride_batch
                      + (size_t)kv_head_idx * params.norm_stride_head;

    // Shared memory layout (similar to v4 paged, but no quant_offsets needed)
    constexpr uint32_t staging_bytes = ((tile_tokens * bytes_per_row + 15) / 16) * 16;
    constexpr uint32_t norms_bytes = tile_tokens * dim_chunks * sizeof(float);
    constexpr uint32_t main_loop_bytes = staging_bytes + norms_bytes;
    constexpr uint32_t sync_state_bytes = bdz * bdy * (head_dim + 2) * sizeof(float);
    constexpr uint32_t md_offset = (main_loop_bytes > sync_state_bytes) ? main_loop_bytes : sync_state_bytes;

    uint8_t* staging = smem;
    float* smem_norms = (float*)(smem + staging_bytes);
    float* smem_md = (float*)(smem + md_offset);

    AttentionVariant variant(params, batch_idx, smem);

    // Load Q
    vec_t<float, vec_size> q_vec;
    vec_t<float, vec_size> freq;
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kNone) {
        q_vec.cast_load(params.q + batch_idx * params.q_stride_n +
                        qo_head_idx * params.q_stride_h + tx * vec_size);
    }
    block.sync();

    // State
    state_t<vec_size> st;
    float s[tile_size];

    const uint32_t num_iters = ceil_div(seq_len, tile_tokens);

    uint32_t chunk_idx = tx / 8;
    uint32_t inner_tx = tx % 8;

    // Main loop
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t token_start = iter * tile_tokens;
        uint32_t remaining = seq_len - token_start;

        // --- Phase 1: cp_async packed K bytes + precompute K norms ---
        cp_async_contiguous_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz, IdType>(
            staging, params.k_quant, quant_base, params.quant_stride_token,
            token_start, remaining, tx, ty, tz);
        cp_async::commit_group();

        precompute_norms_contiguous<head_dim, tile_size_per_bdx, bdx, bdy, bdz, IdType>(
            smem_norms, params.k_norms, norm_base, params.norm_stride_token,
            token_start, remaining, codebook_scale, tx, ty, tz);

        cp_async::wait_group<0>();
        block.sync();

        // --- Phase 2: QK with inline dequant from staging ---
        {
            float m_prev = st.m;
            #pragma unroll
            for (uint32_t j = 0; j < tile_size; ++j) {
                uint32_t row = tz * tile_size + j;

                float k_vals[vec_size];
                if (chunk_idx < dim_chunks) {
                    float ns = smem_norms[row * dim_chunks + chunk_idx];
                    inline_dequant_8_contiguous(k_vals, staging + row * bytes_per_row + chunk_idx * 32,
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
                const uint32_t pos = iter * tile_tokens + tz * tile_size + j;
                s[j] = variant.LogitsTransform(params, s[j], batch_idx, 0, pos,
                                                qo_head_idx, kv_head_idx);
                if constexpr (AttentionVariant::use_softmax) {
                    s[j] *= variant.sm_scale_log2;
                }

                bool mask = variant.LogitsMask(params, batch_idx, 0, pos,
                                                qo_head_idx, kv_head_idx);
                s[j] = (iter * tile_tokens + tz * tile_size + j < seq_len && mask)
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
        cp_async_contiguous_tile<head_dim, tile_size_per_bdx, bdx, bdy, bdz, IdType>(
            staging, params.v_quant, quant_base, params.quant_stride_token,
            token_start, remaining, tx, ty, tz);
        cp_async::commit_group();

        precompute_norms_contiguous<head_dim, tile_size_per_bdx, bdx, bdy, bdz, IdType>(
            smem_norms, params.v_norms, norm_base, params.norm_stride_token,
            token_start, remaining, codebook_scale, tx, ty, tz);

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
                    inline_dequant_8_contiguous(v_vals, staging + row * bytes_per_row + chunk_idx * 32,
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

    // Cross-warp merge (identical to v4 paged)
    sync_state<vec_size, bdx, bdy, bdz>(
        variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);

    // Output
    #pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
        st.o[i] = variant.OutputTransform(params, st.o[i], bx, 0,
                                           qo_head_idx, st.m, st.d, 1.0f);
    }

    if (tz == 0) {
        st.o.cast_store(params.o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
        if (params.lse != nullptr) {
            params.lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
        }
    }
}

// Kernel wrapper
template <PosEncodingMode POS_ENCODING_MODE,
          uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          typename AttentionVariant, typename Params>
__global__ void TurboQuantContiguousDecodeKernelV4(const __grid_constant__ Params params) {
    extern __shared__ uint8_t smem[];
    TurboQuantContiguousDecodeDeviceV4<POS_ENCODING_MODE, tile_size_per_bdx,
                                       vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

}  // namespace flashinfer
