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

// ═══ Fused quantize-write WITH Hadamard rotation ═══════════════════════
//
// Full pipeline in one kernel: load fp16 → L2 norm → normalize → signs ×
// → FWHT (warp shuffles) → codebook quantize → nibble pack → store.
//
// 32 threads per (token, head). Each thread holds VEC_SIZE = padded_dim/32
// elements. FWHT stages: local (h < VEC_SIZE) + cross-thread (shfl_xor).
//
// Template on VEC_SIZE for compile-time unrolling.

template <uint32_t VEC_SIZE>
__global__ void quantize_write_hadamard_kernel(
    const float*  __restrict__ kv_in,       // [num_tokens, num_heads, head_dim] float32
    uint8_t*      __restrict__ quant_out,   // [num_tokens, num_heads, dim_chunks * 32] uint8
    __half*       __restrict__ norms_out,   // [num_tokens, num_heads, dim_chunks] fp16
    const float*  __restrict__ signs,       // [padded_dim] float32
    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_dim,
    const uint32_t padded_dim,
    const uint32_t dim_chunks,
    const uint32_t quant_bytes_per_head,
    const float    boundary_scale
) {
    const uint32_t token_idx = blockIdx.x;
    const uint32_t head_idx  = blockIdx.y;
    const uint32_t tid       = threadIdx.x;  // 0..31

    if (token_idx >= num_tokens) return;

    const float* head_ptr = kv_in + (size_t)token_idx * num_heads * head_dim
                                  + (size_t)head_idx * head_dim;

    // ── Step 1: Load float32 + compute L2 norm ──
    float v[VEC_SIZE];
    float sum_sq = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; i++) {
        uint32_t d = tid * VEC_SIZE + i;
        float val = (d < head_dim) ? head_ptr[d] : 0.0f;
        v[i] = val;
        sum_sq += val * val;
    }

    // Warp reduce for L2 norm
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    float l2_norm = sqrtf(__shfl_sync(0xFFFFFFFF, sum_sq, 0));
    float inv_norm = (l2_norm > 1e-8f) ? (1.0f / l2_norm) : 0.0f;

    // ── Step 2: Normalize ──
    #pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; i++) {
        v[i] *= inv_norm;
    }

    // ── Step 3: Signs multiply ──
    #pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; i++) {
        v[i] *= signs[tid * VEC_SIZE + i];
    }

    // ── Step 4: FWHT via warp shuffles ──
    // Local stages: h = 1, 2, ..., VEC_SIZE/2
    #pragma unroll
    for (uint32_t h = 1; h < VEC_SIZE; h <<= 1) {
        #pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; i += 2 * h) {
            #pragma unroll
            for (uint32_t k = 0; k < h; k++) {
                float a = v[i + k], b = v[i + k + h];
                v[i + k] = a + b;
                v[i + k + h] = a - b;
            }
        }
    }

    // Cross-thread stages via shfl_xor
    #pragma unroll
    for (uint32_t delta = 1; delta < 32; delta <<= 1) {
        #pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; i++) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], delta);
            if (tid & delta)
                v[i] = partner - v[i];
            else
                v[i] = v[i] + partner;
        }
    }

    // Normalize: scale by 1/sqrt(padded_dim)
    float fwht_scale = rsqrtf(static_cast<float>(padded_dim));
    #pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; i++) {
        v[i] *= fwht_scale;
    }

    // ── Step 5: Pre-scale boundaries ──
    float b_scaled[15];
    #pragma unroll
    for (int i = 0; i < 15; i++) {
        b_scaled[i] = kWriteBoundaries4bit[i] * boundary_scale;
    }

    // ── Step 6: Quantize + pack ──
    // Thread tid holds dims [tid*VEC_SIZE .. tid*VEC_SIZE+VEC_SIZE-1].
    // Pack pairs of adjacent dims into nibbles.
    uint8_t* quant_ptr = quant_out + (size_t)token_idx * num_heads * quant_bytes_per_head
                                   + (size_t)head_idx * quant_bytes_per_head;
    __half* norm_ptr = norms_out + (size_t)token_idx * num_heads * dim_chunks
                                 + (size_t)head_idx * dim_chunks;
    __half norm_fp16 = __float2half(l2_norm);

    #pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; i += 2) {
        uint8_t idx0 = write_bucketize_4bit(v[i], b_scaled);
        uint8_t idx1 = write_bucketize_4bit(v[i + 1], b_scaled);
        uint8_t packed = (idx0 << 4) | (idx1 & 0x0F);

        // Compute output byte position: dims are contiguous in the rotated vector,
        // so byte position = (tid * VEC_SIZE + i) / 2
        uint32_t byte_pos = (tid * VEC_SIZE + i) / 2;
        quant_ptr[byte_pos] = packed;
    }

    // Store norm for each chunk
    if (tid == 0) {
        #pragma unroll
        for (uint32_t c = 0; c < dim_chunks; c++) {
            norm_ptr[c] = norm_fp16;
        }
    }
}

// ---- Fused launcher (with Hadamard) ----------------------------------------

inline cudaError_t launch_quantize_write_hadamard(
    const float*  kv_in,       // [num_tokens, num_heads, head_dim] float32
    uint8_t*      quant_out,
    __half*       norms_out,
    const float*  signs,        // [padded_dim] Hadamard signs
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

    uint32_t vec_size = padded_dim / 32;

    switch (vec_size) {
        case 2:  // padded_dim=64
            quantize_write_hadamard_kernel<2><<<grid, block, 0, stream>>>(
                kv_in, quant_out, norms_out, signs,
                num_tokens, num_heads, head_dim, padded_dim,
                dim_chunks, quant_bytes_per_head, boundary_scale);
            break;
        case 4:  // padded_dim=128
            quantize_write_hadamard_kernel<4><<<grid, block, 0, stream>>>(
                kv_in, quant_out, norms_out, signs,
                num_tokens, num_heads, head_dim, padded_dim,
                dim_chunks, quant_bytes_per_head, boundary_scale);
            break;
        case 8:  // padded_dim=256
            quantize_write_hadamard_kernel<8><<<grid, block, 0, stream>>>(
                kv_in, quant_out, norms_out, signs,
                num_tokens, num_heads, head_dim, padded_dim,
                dim_chunks, quant_bytes_per_head, boundary_scale);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

}  // namespace turboquant
