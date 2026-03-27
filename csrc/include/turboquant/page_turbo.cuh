#pragma once
// Paged KV cache for TurboQuant quantized data.
// Mirrors FlashInfer's paged_kv_t but stores quantized bytes + norms
// instead of raw DTypeKV elements.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Use FlashInfer's fastdiv
#include "flashinfer/fastdiv.cuh"

namespace turboquant {

// Lloyd-Max codebook centroids for N(0,1) — stored in constant memory.
// 4-bit: 16 levels, 3-bit: 8 levels.
// Scaled by 1/sqrt(dim) at runtime in the kernel.
__device__ __constant__ float kCodebook4bit[16] = {
    -2.7326368f, -2.0693470f, -1.6180345f, -1.2562491f,
    -0.9423684f, -0.6567591f, -0.3880823f, -0.1284154f,
     0.1284154f,  0.3880823f,  0.6567591f,  0.9423684f,
     1.2562491f,  1.6180345f,  2.0693470f,  2.7326368f,
};

__device__ __constant__ float kCodebook3bit[8] = {
    -2.1519775f, -1.3439709f, -0.7560052f, -0.2451210f,
     0.2451210f,  0.7560052f,  1.3439709f,  2.1519775f,
};

// TurboQuant paged KV cache.
//
// Quantized storage per token per head:
//   hi_dims (first 32 dims): 4-bit codebook indices, nibble-packed → 16 bytes
//   lo_dims (last 32 dims):  3-bit codebook indices, GGML-packed  → 12 bytes
//   norm: FP16 L2 norm → 2 bytes
//   Total: 30 bytes per token per head (vs 128 bytes for fp16 head_dim=64)
//
// Page layout (HND-like):
//   k_quant: [max_pages, num_heads, page_size, quant_bytes_per_token]
//   v_quant: same
//   k_norms: [max_pages, num_heads, page_size]  (FP16)
//   v_norms: same
//
// For head_dim > 64: multiple "dim chunks" of 64 each, stored sequentially
// within the quant_bytes_per_token dimension.

template <typename IdType>
struct paged_kv_turbo_t {
    static constexpr uint32_t kTileDims = 64;
    static constexpr uint32_t kHiDims = 32;
    static constexpr uint32_t kLoDims = 32;
    static constexpr uint32_t kHiBytesPerToken = 16;  // 32 dims × 4 bits / 8
    static constexpr uint32_t kLoBytesPerToken = 12;  // 32 dims × 3 bits / 8
    static constexpr uint32_t kQuantBytesPerChunk = kHiBytesPerToken + kLoBytesPerToken;  // 28

    flashinfer::uint_fastdiv page_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t padded_dim;       // next_power_of_2(head_dim)
    uint32_t dim_chunks;       // padded_dim / kTileDims
    uint32_t batch_size;
    uint32_t quant_stride_page;  // num_heads * page_size * dim_chunks * kQuantBytesPerChunk
    uint32_t quant_stride_n;     // dim_chunks * kQuantBytesPerChunk (bytes per token per head)
    uint32_t quant_stride_h;     // page_size * dim_chunks * kQuantBytesPerChunk
    uint32_t norm_stride_page;   // num_heads * page_size * dim_chunks
    uint32_t norm_stride_n;      // dim_chunks
    uint32_t norm_stride_h;      // page_size * dim_chunks

    uint8_t* k_quant;   // quantized K data (packed codebook indices)
    uint8_t* v_quant;   // quantized V data
    __half* k_norms;     // FP16 L2 norms for K
    __half* v_norms;     // FP16 L2 norms for V

    IdType* indices;     // page table (same as paged_kv_t)
    IdType* indptr;      // CSR offsets (same as paged_kv_t)
    IdType* last_page_len;
    IdType* rope_pos_offset;

    __host__ __device__ __forceinline__ paged_kv_turbo_t() {}

    __host__ __forceinline__ paged_kv_turbo_t(
        uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
        uint32_t padded_dim, uint32_t batch_size,
        uint8_t* k_quant, uint8_t* v_quant,
        __half* k_norms, __half* v_norms,
        IdType* indices, IdType* indptr, IdType* last_page_len,
        IdType* rope_pos_offset = nullptr)
        : num_heads(num_heads), page_size(page_size), head_dim(head_dim),
          padded_dim(padded_dim), batch_size(batch_size),
          k_quant(k_quant), v_quant(v_quant),
          k_norms(k_norms), v_norms(v_norms),
          indices(indices), indptr(indptr), last_page_len(last_page_len),
          rope_pos_offset(rope_pos_offset)
    {
        dim_chunks = padded_dim / kTileDims;
        // HND layout for quantized data
        quant_stride_n = dim_chunks * kQuantBytesPerChunk;
        quant_stride_h = page_size * quant_stride_n;
        quant_stride_page = num_heads * quant_stride_h;
        norm_stride_n = dim_chunks;
        norm_stride_h = page_size * norm_stride_n;
        norm_stride_page = num_heads * norm_stride_h;
    }

    __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
        if (indptr[batch_idx + 1] == indptr[batch_idx]) return 0;
        return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
    }

    // Get byte offset into k_quant/v_quant for a specific token
    __device__ __forceinline__ size_t get_quant_offset(
        size_t page_idx, uint32_t head_idx, uint32_t entry_idx, uint32_t chunk_idx
    ) const {
        return page_idx * quant_stride_page
             + head_idx * quant_stride_h
             + entry_idx * quant_stride_n
             + chunk_idx * kQuantBytesPerChunk;
    }

    // Get offset into k_norms/v_norms
    __device__ __forceinline__ size_t get_norm_offset(
        size_t page_idx, uint32_t head_idx, uint32_t entry_idx, uint32_t chunk_idx
    ) const {
        return page_idx * norm_stride_page
             + head_idx * norm_stride_h
             + entry_idx * norm_stride_n
             + chunk_idx;
    }

    __device__ __forceinline__ size_t protective_get_quant_offset(
        IdType page_iter, uint32_t head_idx, uint32_t entry_idx,
        uint32_t chunk_idx, IdType last_indptr
    ) const {
        if (page_iter < last_indptr) {
            return get_quant_offset(__ldg(indices + page_iter), head_idx, entry_idx, chunk_idx);
        }
        return 0;
    }

    __device__ __forceinline__ size_t protective_get_norm_offset(
        IdType page_iter, uint32_t head_idx, uint32_t entry_idx,
        uint32_t chunk_idx, IdType last_indptr
    ) const {
        if (page_iter < last_indptr) {
            return get_norm_offset(__ldg(indices + page_iter), head_idx, entry_idx, chunk_idx);
        }
        return 0;
    }

    // ── Inline dequantization ────────────────────────────────────────

    // Serial dequant: one thread writes all 64 dims (used as fallback).
    __device__ __forceinline__ static void dequant_chunk_to_smem(
        const uint8_t* quant_bytes, float norm, float codebook_scale,
        __half* out
    ) {
        float s = codebook_scale * norm;
        for (uint32_t i = 0; i < kHiBytesPerToken; i++) {
            uint8_t packed = quant_bytes[i];
            out[i * 2]     = __float2half(kCodebook4bit[(packed >> 4) & 0x0F] * s);
            out[i * 2 + 1] = __float2half(kCodebook4bit[packed & 0x0F] * s);
        }
        const uint8_t* lo = quant_bytes + kHiBytesPerToken;
        __half* lo_out = out + kHiDims;
        for (uint32_t g = 0; g < kLoDims / 8; g++) {
            const uint8_t* p = lo + g * 3;
            lo_out[g*8+0] = __float2half(kCodebook3bit[p[0] & 7] * s);
            lo_out[g*8+1] = __float2half(kCodebook3bit[(p[0] >> 3) & 7] * s);
            lo_out[g*8+2] = __float2half(kCodebook3bit[((p[0] >> 6) & 3) | ((p[1] & 1) << 2)] * s);
            lo_out[g*8+3] = __float2half(kCodebook3bit[(p[1] >> 1) & 7] * s);
            lo_out[g*8+4] = __float2half(kCodebook3bit[(p[1] >> 4) & 7] * s);
            lo_out[g*8+5] = __float2half(kCodebook3bit[((p[1] >> 7) & 1) | ((p[2] & 3) << 1)] * s);
            lo_out[g*8+6] = __float2half(kCodebook3bit[(p[2] >> 2) & 7] * s);
            lo_out[g*8+7] = __float2half(kCodebook3bit[(p[2] >> 5) & 7] * s);
        }
    }

    // Parallel dequant: thread `tx` (0..7) writes its 8-dim slice.
    // bdx=8, vec_size=8: thread tx owns dims [tx*8 .. tx*8+7].
    // tx 0-3: hi dims (4-bit nibble packed, 4 bytes per thread → 8 dims)
    // tx 4-7: lo dims (3-bit GGML packed, 3 bytes per 8-dim group)
    __device__ __forceinline__ static void dequant_chunk_parallel(
        const uint8_t* quant_bytes, float norm, float codebook_scale,
        __half* out, uint32_t tx
    ) {
        float s = codebook_scale * norm;
        uint32_t dim_start = tx * 8;

        if (tx < 4) {
            // Hi dims: tx*8 .. tx*8+7, 4-bit nibble packed
            // 8 dims = 4 packed bytes starting at quant_bytes[tx * 4]
            const uint8_t* src = quant_bytes + tx * 4;
            __half* dst = out + dim_start;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
                uint8_t packed = src[i];
                dst[i * 2]     = __float2half(kCodebook4bit[(packed >> 4) & 0x0F] * s);
                dst[i * 2 + 1] = __float2half(kCodebook4bit[packed & 0x0F] * s);
            }
        } else {
            // Lo dims: (tx-4)*8 + 32 .. (tx-4)*8 + 39
            // Each 8-dim group is 3 bytes in GGML layout
            uint32_t group = tx - 4;
            const uint8_t* p = quant_bytes + kHiBytesPerToken + group * 3;
            __half* dst = out + kHiDims + group * 8;

            dst[0] = __float2half(kCodebook3bit[p[0] & 7] * s);
            dst[1] = __float2half(kCodebook3bit[(p[0] >> 3) & 7] * s);
            dst[2] = __float2half(kCodebook3bit[((p[0] >> 6) & 3) | ((p[1] & 1) << 2)] * s);
            dst[3] = __float2half(kCodebook3bit[(p[1] >> 1) & 7] * s);
            dst[4] = __float2half(kCodebook3bit[(p[1] >> 4) & 7] * s);
            dst[5] = __float2half(kCodebook3bit[((p[1] >> 7) & 1) | ((p[2] & 3) << 1)] * s);
            dst[6] = __float2half(kCodebook3bit[(p[2] >> 2) & 7] * s);
            dst[7] = __float2half(kCodebook3bit[(p[2] >> 5) & 7] * s);
        }
    }
};

}  // namespace turboquant
