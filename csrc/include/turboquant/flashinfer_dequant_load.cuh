#pragma once
// TurboQuant dequant-load for FlashInfer integration.
// Replaces cp_async KV load with: load 4-bit packed → codebook lookup → fp16 smem.
//
// Call this instead of cp_async::pred_load for each KV element.
// The output is fp16 in shared memory — FlashInfer's existing
// cast_load/QK/softmax/V code runs unchanged.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "turboquant/page_turbo.cuh"

namespace turboquant {

// Dequant-load one thread's vec_size values from 4-bit packed source to fp16 smem.
// Each thread reads vec_size/2 packed bytes → vec_size fp16 values.
template <uint32_t vec_size>
__device__ __forceinline__ void dequant_load_to_smem(
    __half* smem_dst,         // write position in shared memory
    const uint8_t* quant_src, // start of this token's chunk quantized bytes (32 bytes)
    float norm_times_scale,   // codebook_scale * norm (precomputed)
    uint32_t tx,              // thread x (0..bdx-1)
    bool predicate
) {
    if (!predicate) {
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
            smem_dst[i] = __float2half(0.0f);
        }
        return;
    }

    const uint8_t* src = quant_src + tx * (vec_size / 2);
    float s = norm_times_scale;

    #pragma unroll
    for (uint32_t i = 0; i < vec_size / 2; i++) {
        uint8_t packed = src[i];
        smem_dst[i * 2]     = __float2half(kCodebook4bit[(packed >> 4) & 0x0F] * s);
        smem_dst[i * 2 + 1] = __float2half(kCodebook4bit[packed & 0x0F] * s);
    }
}

// Load one tile of KV data from quantized paged cache into fp16 shared memory.
// This replaces the cp_async preload/loop in FlashInfer's BatchDecodeWithPagedKVCacheDevice.
//
// smem: [tile_tokens, head_dim] fp16 shared memory
// paged_kv: quantized paged KV cache struct
// kv_data: k_quant or v_quant pointer
// kv_norms: k_norms or v_norms pointer
template <uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void dequant_load_kv_tile(
    __half* smem,                       // [tile_tokens, CHUNK_DIMS] fp16 — one 64-dim chunk
    const paged_kv_turbo_t<IdType>& paged_kv,
    const uint8_t* kv_data,
    const __half* kv_norms,
    uint32_t kv_head_idx,
    uint32_t chunk_idx,                 // which 64-dim chunk to load
    uint32_t packed_page_iter_base,
    uint32_t chunk_size,
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    float codebook_scale = rsqrtf(static_cast<float>(paged_kv.padded_dim));
    constexpr uint32_t CHUNK_DIMS = 64;

    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t row_in_tile = (tz * bdy + ty) * tile_size_per_bdx + j;
        bool valid = row_in_tile < chunk_size;

        uint32_t packed = packed_page_iter_base + row_in_tile;
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed, page_iter, entry_idx);

        if (valid && page_iter >= last_indptr) valid = false;

        size_t q_off = valid ?
            paged_kv.get_quant_offset(__ldg(paged_kv.indices + page_iter),
                                       kv_head_idx, entry_idx, chunk_idx) : 0;
        size_t n_off = valid ?
            paged_kv.get_norm_offset(__ldg(paged_kv.indices + page_iter),
                                      kv_head_idx, entry_idx, chunk_idx) : 0;

        float norm = valid ? __half2float(kv_norms[n_off]) : 0.0f;
        float ns = codebook_scale * norm;

        // Write to smem row: [row_in_tile, tx * vec_size .. +vec_size]
        __half* dst = smem + row_in_tile * CHUNK_DIMS + tx * vec_size;
        dequant_load_to_smem<vec_size>(dst, kv_data + q_off, ns, tx, valid);
    }
}

// Full head_dim dequant-load for FlashInfer integration.
// Fills [tile_tokens, head_dim] smem, loading all 64-dim chunks in parallel.
//
// Thread mapping: tx / 8 = chunk_idx, tx % 8 = inner thread within chunk.
// For head_dim=64 (bdx=8): all threads handle chunk 0.
// For head_dim=128 (bdx=16): threads 0-7 → chunk 0, threads 8-15 → chunk 1.
template <uint32_t head_dim, uint32_t tile_size_per_bdx,
          uint32_t bdy, uint32_t bdz, typename IdType>
__device__ __forceinline__ void dequant_load_kv_tile_full(
    __half* smem,                       // [tile_tokens, head_dim] fp16
    const paged_kv_turbo_t<IdType>& paged_kv,
    const uint8_t* kv_data,
    const __half* kv_norms,
    uint32_t kv_head_idx,
    uint32_t packed_page_iter_base,
    uint32_t chunk_size,
    IdType last_indptr,
    uint32_t tx, uint32_t ty, uint32_t tz
) {
    constexpr uint32_t CHUNK_DIMS = 64;
    constexpr uint32_t dim_chunks = head_dim / CHUNK_DIMS;
    float codebook_scale = rsqrtf(static_cast<float>(paged_kv.padded_dim));

    uint32_t chunk_idx = tx / 8;
    uint32_t inner_tx = tx % 8;

    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t row_in_tile = (tz * bdy + ty) * tile_size_per_bdx + j;
        bool valid = row_in_tile < chunk_size;

        uint32_t packed = packed_page_iter_base + row_in_tile;
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed, page_iter, entry_idx);

        if (valid && page_iter >= last_indptr) valid = false;

        if (chunk_idx < dim_chunks) {
            size_t q_off = valid ?
                paged_kv.get_quant_offset(__ldg(paged_kv.indices + page_iter),
                                           kv_head_idx, entry_idx, chunk_idx) : 0;
            size_t n_off = valid ?
                paged_kv.get_norm_offset(__ldg(paged_kv.indices + page_iter),
                                          kv_head_idx, entry_idx, chunk_idx) : 0;

            float norm = valid ? __half2float(kv_norms[n_off]) : 0.0f;
            float ns = codebook_scale * norm;

            __half* dst = smem + row_in_tile * head_dim + chunk_idx * CHUNK_DIMS + inner_tx * 8;
            if (valid) {
                const uint8_t* src = kv_data + q_off + inner_tx * 4;
                #pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    uint8_t packed_byte = src[i];
                    dst[i * 2]     = __float2half(kCodebook4bit[(packed_byte >> 4) & 0x0F] * ns);
                    dst[i * 2 + 1] = __float2half(kCodebook4bit[packed_byte & 0x0F] * ns);
                }
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < 8; i++) dst[i] = __float2half(0.0f);
            }
        }
    }
}

}  // namespace turboquant
