#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turboquant {

// In-place Fast Walsh-Hadamard Transform on shared memory.
// `data` is a shared memory array of `d` floats (d must be power of 2).
// Each thread processes one element. Call with d threads.
__device__ __forceinline__ void fwht_shared(float* data, int d, int tid) {
    for (int h = 1; h < d; h *= 2) {
        __syncthreads();
        if ((tid & (2 * h - 1)) < h) {
            int i = tid;
            int j = tid + h;
            float a = data[i];
            float b = data[j];
            data[i] = a + b;
            data[j] = a - b;
        }
    }
    __syncthreads();

    // Normalize by 1/sqrt(d)
    float inv_sqrt_d = rsqrtf(static_cast<float>(d));
    data[tid] *= inv_sqrt_d;
    __syncthreads();
}

// Register-based FWHT for vec_size elements per thread, bdx threads total.
// Each thread holds vec_size consecutive elements. Total d = bdx * vec_size.
// Uses warp shuffles for cross-thread butterflies.
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ void fwht_register(float vals[vec_size], uint32_t tx) {
    constexpr uint32_t d = bdx * vec_size;  // total dims (e.g. 64)

    // Phase 1: intra-thread butterflies (within each thread's vec_size elements)
    for (uint32_t h = 1; h < vec_size; h *= 2) {
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
            uint32_t partner = i ^ h;
            if (partner > i) {
                float a = vals[i];
                float b = vals[partner];
                vals[i] = a + b;
                vals[partner] = a - b;
            }
        }
    }

    // Phase 2: inter-thread butterflies (across threads via warp shuffle)
    for (uint32_t h = 1; h < bdx; h *= 2) {
        uint32_t partner_tx = tx ^ h;
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; i++) {
            float partner_val = __shfl_xor_sync(0xFFFFFFFF, vals[i], h);
            if (partner_tx > tx) {
                vals[i] = vals[i] + partner_val;
            } else {
                vals[i] = partner_val - vals[i];
            }
        }
    }

    // Normalize
    float inv_sqrt_d = rsqrtf(static_cast<float>(d));
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; i++) {
        vals[i] *= inv_sqrt_d;
    }
}

// Apply random Hadamard rotation in registers.
// signs: [d] array of ±1 values (in global/constant memory).
// Forward: y = FWHT(signs * x)
// Inverse: x = signs * FWHT(y)
template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ void hadamard_rotate_register(
    float vals[vec_size], const float* signs, uint32_t tx
) {
    // Multiply by signs
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; i++) {
        vals[i] *= signs[tx * vec_size + i];
    }
    // FWHT
    fwht_register<vec_size, bdx>(vals, tx);
}

template <uint32_t vec_size, uint32_t bdx>
__device__ __forceinline__ void hadamard_inverse_register(
    float vals[vec_size], const float* signs, uint32_t tx
) {
    // FWHT
    fwht_register<vec_size, bdx>(vals, tx);
    // Multiply by signs
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; i++) {
        vals[i] *= signs[tx * vec_size + i];
    }
}

}  // namespace turboquant
