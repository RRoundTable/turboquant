#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "turboquant/hadamard.cuh"
#include "turboquant/tile.h"

namespace turboquant {

// Lloyd-Max boundaries in device constant memory
__device__ __constant__ float d_boundaries_4bit[15] = {
    -2.4009919259139040f, -1.8436907609368955f, -1.4371417912897455f,
    -1.0993087223329937f, -0.7995637494657970f, -0.5224207238480534f,
    -0.2582488430577470f,  0.0000000000000000f,  0.2582488430577470f,
     0.5224207238480534f,  0.7995637494657970f,  1.0993087223329937f,
     1.4371417912897455f,  1.8436907609368955f,  2.4009919259139040f,
};

__device__ __constant__ float d_boundaries_3bit[7] = {
    -1.7479742217388095f, -1.0499880858322919f, -0.5005631045163847f,
     0.0000000000000000f,  0.5005631045163847f,  1.0499880858322919f,
     1.7479742217388095f,
};

// ── Device helpers ───────────────────────────────────────────────────

// Bucketize a single value against sorted boundaries.
__device__ __forceinline__ uint8_t d_bucketize_4bit(float val, const float* boundaries) {
    // Binary search over 15 boundaries → index 0..15
    uint8_t lo = 0, hi = 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) {  // ceil(log2(16)) = 4 iterations
        uint8_t mid = (lo + hi) >> 1;
        if (mid < 15 && val >= boundaries[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ uint8_t d_bucketize_3bit(float val, const float* boundaries) {
    // Binary search over 7 boundaries → index 0..7
    uint8_t lo = 0, hi = 7;
    #pragma unroll
    for (int i = 0; i < 3; i++) {  // ceil(log2(8)) = 3 iterations
        uint8_t mid = (lo + hi) >> 1;
        if (mid < 7 && val >= boundaries[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Pack two 4-bit values into one byte (high nibble = even, low nibble = odd)
__device__ __forceinline__ uint8_t d_pack_4bit_pair(uint8_t even, uint8_t odd) {
    return (even << 4) | (odd & 0x0F);
}

// Pack 8 3-bit values into 3 bytes (GGML little-endian)
__device__ __forceinline__ void d_pack_3bit_group(
    const uint8_t v[8], uint8_t out[3]
) {
    out[0] = (v[0] & 7) | ((v[1] & 7) << 3) | ((v[2] & 3) << 6);
    out[1] = ((v[2] >> 2) & 1) | ((v[3] & 7) << 1) | ((v[4] & 7) << 4) | ((v[5] & 1) << 7);
    out[2] = ((v[5] >> 1) & 3) | ((v[6] & 7) << 2) | ((v[7] & 7) << 5);
}

// ── Write Kernel ─────────────────────────────────────────────────────

// One block processes one tile (16 tokens × 64 dims of the rotated vector).
// Grid: (num_token_tiles * tiles_per_row) blocks.
// Block: 64 threads (one per dim within the tile chunk).
//
// Pipeline per block:
//   1. Load 16×64 floats from input (already in [N, padded_dim] layout)
//   2. Each row: compute L2 norm, normalize (norm stored once per first chunk)
//   3. Apply random Hadamard rotation in shared memory (FWHT + sign flip)
//   4. Bucketize rotated values → codebook indices
//   5. Pack 4-bit (first 32) and 3-bit (last 32) into tile bytes
//   6. Write tile to output

__global__ void quantize_kv_kernel(
    const float* __restrict__ input,     // [num_tokens, head_dim]
    uint8_t* __restrict__ output,        // [num_token_tiles * tiles_per_row * TILE_BYTES]
    const float* __restrict__ signs,     // [padded_dim] random ±1 signs for Hadamard
    int num_tokens,
    int head_dim,
    int padded_dim,
    int tiles_per_row                    // padded_dim / TILE_DIMS
) {
    // Block ID encodes (token_tile_idx, chunk_idx)
    int block_id = blockIdx.x;
    int token_tile_idx = block_id / tiles_per_row;
    int chunk_idx = block_id % tiles_per_row;
    int tid = threadIdx.x;  // 0..63 = dim within this chunk

    int global_dim = chunk_idx * TurboQuantTile::kDims + tid;

    // Shared memory for FWHT: 16 rows × padded_dim floats (only for this chunk)
    // We actually need the full padded_dim for FWHT, but that's too large.
    // Instead, we do FWHT on the full vector in a different pass.
    //
    // For Phase 2 MVP: skip in-kernel FWHT. Expect pre-rotated input.
    // The PyTorch reference handles rotation. The CUDA FWHT fusion is Phase 2b.

    // ── Constant memory: codebook boundaries (pre-scaled by 1/sqrt(padded_dim)) ──
    // Loaded from global memory (could be __constant__ in production)
    float scale = rsqrtf(static_cast<float>(padded_dim));

    // Pre-scale boundaries in registers
    float b4[15], b3[7];
    #pragma unroll
    for (int i = 0; i < 15; i++) b4[i] = d_boundaries_4bit[i] * scale;
    #pragma unroll
    for (int i = 0; i < 7; i++) b3[i] = d_boundaries_3bit[i] * scale;

    // Output tile pointer
    uint8_t* tile_ptr = output + (size_t)block_id * TurboQuantTile::kDims * 30;
    // Actually: output + block_id * sizeof(TurboQuantTile)
    tile_ptr = output + (size_t)block_id * 480;

    // Shared memory for norms and quantized indices
    __shared__ float s_norms[TurboQuantTile::kTokens];
    __shared__ uint8_t s_hi_indices[TurboQuantTile::kTokens * TurboQuantTile::kHiDims];
    __shared__ uint8_t s_lo_indices[TurboQuantTile::kTokens * TurboQuantTile::kLoDims];

    // Process each of the 16 rows (tokens)
    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        int token_idx = token_tile_idx * TurboQuantTile::kTokens + row;

        float val = 0.0f;
        if (token_idx < num_tokens && global_dim < padded_dim) {
            if (global_dim < head_dim) {
                val = input[(size_t)token_idx * head_dim + global_dim];
            }
            // else: zero padding for padded dims beyond head_dim
        }

        // Compute L2 norm via warp reduction (all 64 threads = 2 warps)
        float val_sq = val * val;

        // Intra-warp reduction (warp size = 32)
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_sq += __shfl_down_sync(0xFFFFFFFF, val_sq, offset);
        }

        // Lane 0 of each warp has its partial sum → combine via shared memory
        __shared__ float s_partial[2];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) s_partial[warp_id] = val_sq;
        __syncthreads();

        float l2_sq = s_partial[0] + s_partial[1];
        float l2_norm = sqrtf(l2_sq);
        float inv_norm = (l2_norm > 1e-8f) ? (1.0f / l2_norm) : 0.0f;

        // Store norm (only tid=0 writes, once per chunk_idx==0)
        if (tid == 0 && chunk_idx == 0) {
            s_norms[row] = l2_norm;
        }

        // Normalize
        float normalized = val * inv_norm;

        // Bucketize
        if (tid < TurboQuantTile::kHiDims) {
            // First 32 dims → 4-bit
            s_hi_indices[row * TurboQuantTile::kHiDims + tid] = d_bucketize_4bit(normalized, b4);
        } else {
            // Last 32 dims → 3-bit
            int lo_idx = tid - TurboQuantTile::kHiDims;
            s_lo_indices[row * TurboQuantTile::kLoDims + lo_idx] = d_bucketize_3bit(normalized, b3);
        }
        __syncthreads();
    }

    // ── Pack and write tile ──────────────────────────────────────────

    // Write norms (tid 0..15 each write one FP16 norm)
    if (tid < TurboQuantTile::kTokens) {
        __half norm_fp16 = __float2half(s_norms[tid]);
        uint16_t norm_bits;
        memcpy(&norm_bits, &norm_fp16, sizeof(uint16_t));
        tile_ptr[tid * 2]     = norm_bits & 0xFF;
        tile_ptr[tid * 2 + 1] = (norm_bits >> 8) & 0xFF;
    }

    __syncthreads();

    // Pack 4-bit: 16 threads pack 16 rows (each thread packs one row's 32 dims → 16 bytes)
    if (tid < TurboQuantTile::kTokens) {
        int row = tid;
        uint8_t* row_hi = s_hi_indices + row * TurboQuantTile::kHiDims;
        uint8_t* dst = tile_ptr + 32 + row * TurboQuantTile::kHiBytesPerRow;
        for (int i = 0; i < TurboQuantTile::kHiDims; i += 2) {
            dst[i / 2] = d_pack_4bit_pair(row_hi[i], row_hi[i + 1]);
        }
    }

    // Pack 3-bit: next 16 threads pack 16 rows (each thread packs one row's 32 dims → 12 bytes)
    if (tid >= 16 && tid < 32) {
        int row = tid - 16;
        uint8_t* row_lo = s_lo_indices + row * TurboQuantTile::kLoDims;
        uint8_t* dst = tile_ptr + 288 + row * TurboQuantTile::kLoBytesPerRow;
        for (int g = 0; g < TurboQuantTile::kLoDims; g += 8) {
            uint8_t group[8];
            for (int k = 0; k < 8; k++) group[k] = row_lo[g + k];
            uint8_t packed[3];
            d_pack_3bit_group(group, packed);
            dst[(g / 8) * 3]     = packed[0];
            dst[(g / 8) * 3 + 1] = packed[1];
            dst[(g / 8) * 3 + 2] = packed[2];
        }
    }
}

// ── Host launcher ────────────────────────────────────────────────────

// Quantize pre-rotated KV vectors into TurboQuantTile format.
// Input: [num_tokens, head_dim] float (already Hadamard-rotated, or raw for Phase 2 MVP)
// Output: [num_tiles_total * 480] uint8
// Signs: [padded_dim] float ±1 (for future FWHT fusion; unused in MVP)
cudaError_t launch_quantize_kv(
    const float* input,
    uint8_t* output,
    const float* signs,
    int num_tokens,
    int head_dim,
    int padded_dim,
    int tiles_per_row,
    cudaStream_t stream = nullptr
) {
    int pad_tokens = (TurboQuantTile::kTokens - num_tokens % TurboQuantTile::kTokens)
                     % TurboQuantTile::kTokens;
    int num_token_tiles = (num_tokens + pad_tokens) / TurboQuantTile::kTokens;
    int num_blocks = num_token_tiles * tiles_per_row;
    int threads_per_block = TurboQuantTile::kDims;  // 64

    quantize_kv_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input, output, signs,
        num_tokens, head_dim, padded_dim, tiles_per_row
    );

    return cudaGetLastError();
}

}  // namespace turboquant
