#pragma once
// HYP-056: A_35_prime outlier-aware quantize-write kernel (skeleton).
//
// Pipeline per (token, head), one warp of 32 threads:
//   1. Load [head_dim=128] fp16 → smem.
//   2. Gather outlier 64 dims via outlier_idx[H, 64] into per-thread regs (2 dims/thread).
//   3. Gather regular 64 dims via regular_idx[H, 64] into per-thread regs (2 dims/thread).
//   4. Per tier: L2-norm over 64 → normalize → signs × → FWHT-64 → scale by 1/sqrt(64).
//   5. Per tier: bucketize with tier's codebook boundaries.
//   6. Pack: outlier → nibble (32 B), regular → 3-bit GGML (24 B), norms → 4 B.
//   7. Write 60 B tile to kv_cache (64 B aligned).
//
// Output tile layout [tile_bytes_aligned = 64]:
//   [0..32)     outlier nibbles  (2 idx/byte, tid=0..15 writes 2 bytes each)
//   [32..56)    regular GGML-3b  (8 idx in 3 bytes, see pack_3bit_ggml)
//   [56..58)    outlier fp16 norm
//   [58..60)    regular fp16 norm
//   [60..64)    padding (0)
//
// This skeleton uses ONE warp per slot. Reachable throughput target from
// HYP-056 Phase 2: ~20-30 μs per 1 k tokens × 8 KV heads on A100. The
// correctness test (HYP-056 Phase 2b) gates against the Python reference
// in turboquant/outlier_aware_a35.py.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace turboquant {

// Boundaries for bucketize. Lloyd-Max midpoints for N(0,1).
__device__ __constant__ float kA35_B_out4[15] = {
    -2.4009919259139040f, -1.8436907609368955f, -1.4371417912897455f,
    -1.0993087223329937f, -0.7995637494657970f, -0.5224207238480534f,
    -0.2582488430577470f,  0.0000000000000000f,  0.2582488430577470f,
     0.5224207238480534f,  0.7995637494657970f,  1.0993087223329937f,
     1.4371417912897455f,  1.8436907609368955f,  2.4009919259139040f,
};

__device__ __constant__ float kA35_B_reg3[7] = {
    -1.7479880913323015f, -1.0499880858322919f, -0.5005631045163846f,
     0.0000000000000000f,  0.5005631045163846f,  1.0499880858322919f,
     1.7479880913323015f,
};

__device__ __forceinline__ uint8_t a35_bucket4(float v, const float* b) {
    uint8_t lo = 0, hi = 15;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t mid = (lo + hi) >> 1;
        if (mid < 15 && v >= b[mid]) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ uint8_t a35_bucket3(float v, const float* b) {
    uint8_t lo = 0, hi = 7;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        uint8_t mid = (lo + hi) >> 1;
        if (mid < 7 && v >= b[mid]) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// FWHT on 64 dims spread across 32 threads (2 dims/thread).
// Local stage (h=1): butterfly within thread (v0, v1).
// Cross-thread stages (delta=1..16): shfl_xor butterflies.
// Final scale: × 1/sqrt(64).
__device__ __forceinline__ void a35_fwht_64(float& v0, float& v1) {
    // Local h=1
    float a = v0, b = v1;
    v0 = a + b;
    v1 = a - b;

    // Cross-thread delta=1,2,4,8,16
    const uint32_t tid = threadIdx.x & 31;
    #pragma unroll
    for (uint32_t delta = 1; delta < 32; delta <<= 1) {
        float p0 = __shfl_xor_sync(0xFFFFFFFF, v0, delta);
        float p1 = __shfl_xor_sync(0xFFFFFFFF, v1, delta);
        if (tid & delta) { v0 = p0 - v0; v1 = p1 - v1; }
        else             { v0 = v0 + p0; v1 = v1 + p1; }
    }

    // Normalize FWHT
    const float s = 0.125f;  // 1/sqrt(64)
    v0 *= s;
    v1 *= s;
}

// Cooperative 3-bit GGML packing: 8 indices per group → 3 bytes.
// For a 64-dim regular tier with 2 dims/thread, one warp has 8 groups of 8.
// Threads 0..3 write byte 0..23 (3 bytes per group × 8 groups / 32 lanes = 3/4 byte per lane).
// Simpler: 32 lanes × 64 indices → use shfl to collect 8 consecutive indices, one thread writes 3 bytes.
__device__ __forceinline__ void a35_pack_3bit_group(
    uint32_t idx_lo,        // my 1st 3-bit index (dim 2*tid)
    uint32_t idx_hi,        // my 2nd 3-bit index (dim 2*tid+1)
    uint8_t* out_ptr        // points to start of 24 B regular region
) {
    // A group = 8 consecutive dims = 4 threads (each owns 2 dims).
    // Collect via shfl: group g (g in 0..7) is threads 4*g..4*g+3.
    // Lane 0 of each group writes the 3 output bytes.
    const uint32_t tid = threadIdx.x & 31;
    const uint32_t group = tid >> 2;       // 0..7
    const uint32_t lane_in_grp = tid & 3;  // 0..3

    // Gather 8 indices of my group into lane 0 of the group.
    // Sources: lane_in_grp=0 provides idx_lo=d0, idx_hi=d1
    //          lane_in_grp=1 provides           d2,       d3
    //          lane_in_grp=2 provides           d4,       d5
    //          lane_in_grp=3 provides           d6,       d7
    uint32_t src = group * 4;
    uint32_t i0 = __shfl_sync(0xFFFFFFFF, idx_lo, src + 0) & 0x7;
    uint32_t i1 = __shfl_sync(0xFFFFFFFF, idx_hi, src + 0) & 0x7;
    uint32_t i2 = __shfl_sync(0xFFFFFFFF, idx_lo, src + 1) & 0x7;
    uint32_t i3 = __shfl_sync(0xFFFFFFFF, idx_hi, src + 1) & 0x7;
    uint32_t i4 = __shfl_sync(0xFFFFFFFF, idx_lo, src + 2) & 0x7;
    uint32_t i5 = __shfl_sync(0xFFFFFFFF, idx_hi, src + 2) & 0x7;
    uint32_t i6 = __shfl_sync(0xFFFFFFFF, idx_lo, src + 3) & 0x7;
    uint32_t i7 = __shfl_sync(0xFFFFFFFF, idx_hi, src + 3) & 0x7;

    if (lane_in_grp == 0) {
        uint32_t p24 = i0 | (i1 << 3) | (i2 << 6) | (i3 << 9)
                     | (i4 << 12) | (i5 << 15) | (i6 << 18) | (i7 << 21);
        out_ptr[group * 3 + 0] = static_cast<uint8_t>(p24 & 0xFF);
        out_ptr[group * 3 + 1] = static_cast<uint8_t>((p24 >> 8) & 0xFF);
        out_ptr[group * 3 + 2] = static_cast<uint8_t>((p24 >> 16) & 0xFF);
    }
}

// Cooperative 4-bit nibble packing: lane t owns dims (2t, 2t+1), writes 1 byte at byte t (0..15),
// plus dim-pair from lane t+16 goes to byte t+16. Since we have 64 dims with 2 dims/thread:
// lanes 0..31 each own 2 dims → 32 bytes total.
__device__ __forceinline__ void a35_pack_4bit_nibble(
    uint32_t idx_lo,
    uint32_t idx_hi,
    uint8_t* out_ptr
) {
    const uint32_t tid = threadIdx.x & 31;
    uint8_t packed = static_cast<uint8_t>((idx_lo & 0x0F) | ((idx_hi & 0x0F) << 4));
    out_ptr[tid] = packed;
}

// ── Write kernel ────────────────────────────────────────────────────────────
//
// Assumptions for the first version:
//   - head_dim = 128
//   - num_kv_heads arbitrary
//   - tile_bytes = 64, raw=60
//   - signs are two separate 64-element arrays: signs_out[64], signs_reg[64]
//
// Grid:  (num_tokens, num_kv_heads)
// Block: 32 threads (one warp)

__global__ void quantize_write_a35_kernel(
    const __half*  __restrict__ kv_in,        // [num_tokens, num_kv_heads, 128] fp16
    uint8_t*       __restrict__ tiles_out,    // [num_tokens, num_kv_heads, 64] uint8
    const int32_t* __restrict__ outlier_idx,  // [num_kv_heads, 64]
    const int32_t* __restrict__ regular_idx,  // [num_kv_heads, 64]
    const float*   __restrict__ signs_out,    // [64]
    const float*   __restrict__ signs_reg,    // [64]
    uint32_t num_tokens,
    uint32_t num_kv_heads
) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;  // 0..31

    if (t >= num_tokens) return;

    const __half* head_ptr = kv_in + ((size_t)t * num_kv_heads + h) * 128;

    // Load all 128 dims into smem (fp32 for safer reductions).
    __shared__ float smem_x[128];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        smem_x[tid * 4 + i] = __half2float(head_ptr[tid * 4 + i]);
    }
    __syncwarp();

    const int32_t* out_idx_h = outlier_idx + (size_t)h * 64;
    const int32_t* reg_idx_h = regular_idx + (size_t)h * 64;

    // Gather outlier (2 dims per thread). outlier_idx is [64]: slot i = tid*2+{0,1}? No — we want
    // consecutive packing, so assign tid 0..31 to outlier dims (2*tid, 2*tid+1) in the GATHERED order.
    // The gather resolves through smem_x[ outlier_idx_h[k] ].
    float vo0 = smem_x[out_idx_h[tid * 2 + 0]];
    float vo1 = smem_x[out_idx_h[tid * 2 + 1]];
    float vr0 = smem_x[reg_idx_h[tid * 2 + 0]];
    float vr1 = smem_x[reg_idx_h[tid * 2 + 1]];

    // L2 norm per tier via warp-reduction.
    float ss_o = vo0 * vo0 + vo1 * vo1;
    float ss_r = vr0 * vr0 + vr1 * vr1;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss_o += __shfl_down_sync(0xFFFFFFFF, ss_o, off);
        ss_r += __shfl_down_sync(0xFFFFFFFF, ss_r, off);
    }
    float l2_o = sqrtf(__shfl_sync(0xFFFFFFFF, ss_o, 0));
    float l2_r = sqrtf(__shfl_sync(0xFFFFFFFF, ss_r, 0));
    float inv_o = (l2_o > 1e-8f) ? (1.0f / l2_o) : 0.0f;
    float inv_r = (l2_r > 1e-8f) ? (1.0f / l2_r) : 0.0f;

    // Normalize, apply signs, FWHT (per-tier).
    vo0 *= inv_o; vo1 *= inv_o;
    vr0 *= inv_r; vr1 *= inv_r;

    vo0 *= signs_out[tid * 2 + 0];
    vo1 *= signs_out[tid * 2 + 1];
    vr0 *= signs_reg[tid * 2 + 0];
    vr1 *= signs_reg[tid * 2 + 1];

    a35_fwht_64(vo0, vo1);
    a35_fwht_64(vr0, vr1);

    // Bucketize with per-tier boundaries, scaled by 1/sqrt(64) = 0.125.
    const float scale64 = 0.125f;
    float b4s[15];
    float b3s[7];
    #pragma unroll
    for (int i = 0; i < 15; i++) b4s[i] = kA35_B_out4[i] * scale64;
    #pragma unroll
    for (int i = 0; i < 7; i++)  b3s[i] = kA35_B_reg3[i] * scale64;

    uint32_t oi0 = a35_bucket4(vo0, b4s);
    uint32_t oi1 = a35_bucket4(vo1, b4s);
    uint32_t ri0 = a35_bucket3(vr0, b3s);
    uint32_t ri1 = a35_bucket3(vr1, b3s);

    // Output tile pointer.
    uint8_t* tile = tiles_out + ((size_t)t * num_kv_heads + h) * 64;

    // Outlier tier: 32 bytes.
    a35_pack_4bit_nibble(oi0, oi1, tile);

    // Regular tier: 24 bytes starting at offset 32.
    a35_pack_3bit_group(ri0, ri1, tile + 32);

    // Norms (fp16): outlier at 56, regular at 58.
    if (tid == 0) {
        *reinterpret_cast<__half*>(tile + 56) = __float2half(l2_o);
        *reinterpret_cast<__half*>(tile + 58) = __float2half(l2_r);
        *reinterpret_cast<uint32_t*>(tile + 60) = 0;  // zero the 4-byte padding
    }
}

inline cudaError_t launch_quantize_write_a35(
    const __half*  kv_in,
    uint8_t*       tiles_out,
    const int32_t* outlier_idx,
    const int32_t* regular_idx,
    const float*   signs_out,
    const float*   signs_reg,
    uint32_t       num_tokens,
    uint32_t       num_kv_heads,
    cudaStream_t   stream = nullptr
) {
    if (num_tokens == 0 || num_kv_heads == 0) return cudaSuccess;
    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(32);
    quantize_write_a35_kernel<<<grid, block, 0, stream>>>(
        kv_in, tiles_out, outlier_idx, regular_idx,
        signs_out, signs_reg, num_tokens, num_kv_heads
    );
    return cudaGetLastError();
}

}  // namespace turboquant
