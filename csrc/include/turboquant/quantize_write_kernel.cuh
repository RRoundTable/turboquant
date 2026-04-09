#pragma once
// Fused quantize-and-write kernel for TurboQuant prefill path.
//
// Pipeline: L2 normalize -> codebook quantize (4-bit) -> nibble pack -> store.
// Replaces the slow Python quantize loop (15-1500x slower than fp16 memcpy).
//
// Output layout matches the v4 contiguous decode kernel expectations:
//   quant_out: [num_tokens, num_heads, dim_chunks * 32] uint8
//   norms_out: [num_tokens, num_heads, dim_chunks] fp16
//
// No Hadamard rotation (FWHT) — can be fused in a later pass.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace turboquant {

// Lloyd-Max 4-bit boundaries for N(0,1) in constant memory.
// 15 boundaries separate 16 levels. Midpoints of adjacent centroids.
// Scaled by 1/sqrt(padded_dim) at runtime in the kernel.
__device__ __constant__ float kWriteBoundaries4bit[15] = {
    -2.4009919259139040f, -1.8436907609368955f, -1.4371417912897455f,
    -1.0993087223329937f, -0.7995637494657970f, -0.5224207238480534f,
    -0.2582488430577470f,  0.0000000000000000f,  0.2582488430577470f,
     0.5224207238480534f,  0.7995637494657970f,  1.0993087223329937f,
     1.4371417912897455f,  1.8436907609368955f,  2.4009919259139040f,
};

// ---- Device helpers ---------------------------------------------------------

// Binary search over 15 sorted boundaries -> index in [0..15].
// 4 iterations for ceil(log2(16)).
__device__ __forceinline__ uint8_t write_bucketize_4bit(float val, const float* b) {
    uint8_t lo = 0, hi = 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t mid = (lo + hi) >> 1;
        if (mid < 15 && val >= b[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// ---- Fused quantize-write kernel -------------------------------------------
//
// Grid:  (num_tokens, num_heads)
// Block: (32,) -- 32 threads per (token, head)
//
// Each thread handles 2 dimensions per chunk iteration (64 dims / 32 threads).
// For head_dim=128 (2 chunks of 64), the block loops over chunks sequentially.
//
// Algorithm per (token, head):
//   1. Compute L2 norm over full head_dim
//   2. For each chunk of 64 dims:
//      a. Normalize: val / norm
//      b. Scale boundaries by 1/sqrt(padded_dim)
//      c. Bucketize -> 4-bit index [0..15]
//      d. Pack 2 indices per byte (high nibble = even dim, low nibble = odd dim)
//      e. Store packed bytes to quant_out
//      f. Store norm (fp16) to norms_out

__global__ void quantize_write_kernel(
    const __half* __restrict__ kv_in,       // [num_tokens, num_heads, head_dim] fp16
    uint8_t*      __restrict__ quant_out,   // [num_tokens, num_heads, dim_chunks * 32] uint8
    __half*       __restrict__ norms_out,   // [num_tokens, num_heads, dim_chunks] fp16
    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_dim,
    const uint32_t padded_dim,
    const uint32_t dim_chunks,
    const uint32_t quant_bytes_per_head,    // dim_chunks * 32
    const float    boundary_scale           // 1/sqrt(padded_dim)
) {
    const uint32_t token_idx = blockIdx.x;
    const uint32_t head_idx  = blockIdx.y;
    const uint32_t tid       = threadIdx.x;  // 0..31

    if (token_idx >= num_tokens) return;

    // Pre-scale all 15 boundaries into registers
    float b_scaled[15];
    #pragma unroll
    for (int i = 0; i < 15; i++) {
        b_scaled[i] = kWriteBoundaries4bit[i] * boundary_scale;
    }

    // ---- Step 1: Compute L2 norm over full head_dim ----
    // Each of the 32 threads accumulates partial sum over its strided dims.
    const __half* head_ptr = kv_in + (size_t)token_idx * num_heads * head_dim
                                   + (size_t)head_idx * head_dim;

    float sum_sq = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += 32) {
        float v = __half2float(head_ptr[d]);
        sum_sq += v * v;
    }

    // Warp reduction (all 32 threads in one warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    // Broadcast norm from lane 0 to all threads
    float l2_norm = sqrtf(__shfl_sync(0xFFFFFFFF, sum_sq, 0));
    float inv_norm = (l2_norm > 1e-8f) ? (1.0f / l2_norm) : 0.0f;

    // ---- Step 2: Per-chunk quantize and pack ----
    // Output pointers for this (token, head)
    uint8_t* quant_ptr = quant_out + (size_t)token_idx * num_heads * quant_bytes_per_head
                                   + (size_t)head_idx * quant_bytes_per_head;
    __half* norm_ptr = norms_out + (size_t)token_idx * num_heads * dim_chunks
                                 + (size_t)head_idx * dim_chunks;

    __half norm_fp16 = __float2half(l2_norm);

    for (uint32_t chunk = 0; chunk < dim_chunks; chunk++) {
        uint32_t chunk_base = chunk * 64;

        // Each thread handles 2 consecutive dimensions: tid*2 and tid*2+1
        uint32_t dim0 = chunk_base + tid * 2;
        uint32_t dim1 = chunk_base + tid * 2 + 1;

        // Load and normalize
        float v0 = 0.0f, v1 = 0.0f;
        if (dim0 < head_dim) {
            v0 = __half2float(head_ptr[dim0]) * inv_norm;
        }
        if (dim1 < head_dim) {
            v1 = __half2float(head_ptr[dim1]) * inv_norm;
        }

        // Bucketize
        uint8_t idx0 = write_bucketize_4bit(v0, b_scaled);
        uint8_t idx1 = write_bucketize_4bit(v1, b_scaled);

        // Pack: high nibble = even dim index, low nibble = odd dim index
        uint8_t packed = (idx0 << 4) | (idx1 & 0x0F);

        // Store packed byte
        quant_ptr[chunk * 32 + tid] = packed;

        // Store norm (same value for each chunk, for read kernel compatibility)
        if (tid == 0) {
            norm_ptr[chunk] = norm_fp16;
        }
    }
}

// ---- Host launcher ----------------------------------------------------------

inline cudaError_t launch_quantize_write(
    const __half* kv_in,        // [num_tokens, num_heads, head_dim] fp16
    uint8_t*      quant_out,    // [num_tokens, num_heads, dim_chunks * 32] uint8
    __half*       norms_out,    // [num_tokens, num_heads, dim_chunks] fp16
    uint32_t      num_tokens,
    uint32_t      num_heads,
    uint32_t      head_dim,
    uint32_t      padded_dim,
    cudaStream_t  stream = nullptr
) {
    if (num_tokens == 0 || num_heads == 0) return cudaSuccess;

    uint32_t dim_chunks = padded_dim / 64;
    uint32_t quant_bytes_per_head = dim_chunks * 32;
    float boundary_scale = rsqrtf(static_cast<float>(padded_dim));

    dim3 grid(num_tokens, num_heads);
    dim3 block(32);

    quantize_write_kernel<<<grid, block, 0, stream>>>(
        kv_in, quant_out, norms_out,
        num_tokens, num_heads, head_dim, padded_dim,
        dim_chunks, quant_bytes_per_head, boundary_scale
    );

    return cudaGetLastError();
}

}  // namespace turboquant
