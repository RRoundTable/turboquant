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

}  // namespace turboquant
