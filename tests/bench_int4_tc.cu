// HYP-019: INT4 tensor core QK dot product benchmark.
//
// Compares three QK dot product approaches:
//   1. Scalar FMA + warp shuffle (matches TurboQuant v4's inner loop)
//   2. INT4 tensor cores via PTX mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
//   3. INT8 tensor cores via PTX mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// Mathematical basis (uniform quantization):
//   For uniform codebook: centroid[i] = scale * (i - offset)
//   sum(q_j * centroid[k_j]) = scale_q * scale_k * sum(q_idx * k_idx) + corrections
//   The integer matmul gives the raw index dot product; post-matmul scaling corrects.
//
// A100 tensor core throughput:
//   FP16:  312 TFLOPS
//   INT8:  624 TOPS
//   INT4: 1248 TOPS
//
// Build (standalone, no dependencies on TurboQuant headers):
//   nvcc -std=c++17 -O2 -arch=sm_80 --generate-line-info \
//        -o build/bench_int4_tc tests/bench_int4_tc.cu -lcudart

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

// ────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────

static constexpr int HEAD_DIM = 128;
static constexpr int BLOCK_SEQ = 16;   // KV tokens per tile
static constexpr int WARP_SIZE = 32;

// Lloyd-Max 4-bit codebook centroids for N(0,1)
__device__ __constant__ float kCodebook4bit[16] = {
    -2.7326368f, -2.0693470f, -1.6180345f, -1.2562491f,
    -0.9423684f, -0.6567591f, -0.3880823f, -0.1284154f,
     0.1284154f,  0.3880823f,  0.6567591f,  0.9423684f,
     1.2562491f,  1.6180345f,  2.0693470f,  2.7326368f,
};


// ────────────────────────────────────────────────────────────────────
// Kernel 1: Scalar FMA QK dot product (baseline, matches v4)
// ────────────────────────────────────────────────────────────────────
//
// Each warp computes QK scores for BLOCK_SEQ KV tokens.
// 16 of 32 lanes each handle 8 dimensions (4 packed bytes).
// Warp shuffle reduces across threads.

__global__ void scalar_qk_kernel(
    const float* __restrict__ q,          // [HEAD_DIM]
    const uint8_t* __restrict__ k_packed, // [seq_len, HEAD_DIM/2]
    const float* __restrict__ k_norms,    // [seq_len]
    float* __restrict__ output,           // [seq_len]
    int seq_len,
    float codebook_scale                  // 1/sqrt(padded_dim)
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int base_token = warp_id * BLOCK_SEQ;

    // Lane mapping: lane 0..15 each handle 8 dimensions, lanes 16..31 mirror 0..15
    int dim_lane = lane % 16;

    // Load Q for this lane's 8 dimensions
    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        q_local[i] = q[dim_lane * 8 + i];
    }

    for (int t = 0; t < BLOCK_SEQ; t++) {
        int token = base_token + t;
        if (token >= seq_len) {
            if (lane == 0) output[token] = 0.0f;
            continue;
        }

        float norm_s = codebook_scale * k_norms[token];

        // Dequant + dot product
        float dot = 0.0f;
        const uint8_t* src = k_packed + token * (HEAD_DIM / 2) + dim_lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t p = src[i];
            float k_hi = kCodebook4bit[(p >> 4) & 0x0F] * norm_s;
            float k_lo = kCodebook4bit[p & 0x0F] * norm_s;
            dot += q_local[i * 2] * k_hi;
            dot += q_local[i * 2 + 1] * k_lo;
        }

        // Warp reduce
        if (lane >= 16) dot = 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        }

        if (lane == 0) {
            output[token] = dot;
        }
    }
}


// ────────────────────────────────────────────────────────────────────
// Helpers for integer quantization
// ────────────────────────────────────────────────────────────────────

__device__ __forceinline__ int8_t quantize_to_int4(float val, float inv_scale) {
    int v = __float2int_rn(val * inv_scale);
    return static_cast<int8_t>(max(-8, min(7, v)));
}

__device__ __forceinline__ int8_t quantize_to_int8(float val, float inv_scale) {
    int v = __float2int_rn(val * inv_scale);
    return static_cast<int8_t>(max(-128, min(127, v)));
}


// ────────────────────────────────────────────────────────────────────
// Kernel 2: INT4 tensor core QK dot product
// ────────────────────────────────────────────────────────────────────
//
// Uses mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
// M=16 (Q rows, only row 0 is real query), N=8 (KV tokens), K=64 (head dims)
//
// Strategy: Stage A and B in shared memory, load fragments via PTX ldmatrix
// or manual register packing, then call MMA.
//
// PTX fragment layout (m16n8k64, s4):
//   A (row-major [16,64] s4, 4 regs/thread):
//     groupID = lane / 4, tid = lane % 4
//     a[0] = A[tid,       groupID*8 .. groupID*8+7]   (8 s4 packed)
//     a[1] = A[tid+8,     groupID*8 .. groupID*8+7]
//     a[2] = A[tid,       groupID*8+32 .. groupID*8+39]
//     a[3] = A[tid+8,     groupID*8+32 .. groupID*8+39]
//
//   B (col-major [64,8] s4, 2 regs/thread):
//     b[0] = B[groupID*8 .. groupID*8+7, tid]   (8 s4 packed)
//     b[1] = B[groupID*8+32 .. groupID*8+39, tid]
//
//   D (accumulator, 4 x int32/thread):
//     d[0] = D[tid,   groupID]
//     d[1] = D[tid+4, groupID]
//     d[2] = D[tid+8, groupID]
//     d[3] = D[tid+12, groupID]
//     This covers all 16*8=128 elements across 32 threads.

__global__ void int4_tc_qk_kernel(
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_packed,
    const float* __restrict__ k_norms,
    float* __restrict__ output,
    int seq_len,
    float codebook_scale,
    float q_scale,
    float q_inv_scale
) {
    int block_base = blockIdx.x * BLOCK_SEQ;
    int lane = threadIdx.x;
    int tid = lane & 3;       // 0..3 (thread position within group)
    int groupID = lane >> 2;  // 0..7 (which group)

    // Shared memory for staging A and B in the correct layout for fragment loading
    // A: [16, 64] s4 row-major = 16 rows * 32 bytes/row = 512 bytes
    // B: [64, 8] s4 col-major = 8 cols * 32 bytes/col = 256 bytes
    __shared__ uint8_t a_smem[16 * 32];  // 512 bytes
    __shared__ uint8_t b_smem[8 * 32];   // 256 bytes
    __shared__ int32_t c_smem[16 * 8];   // 512 bytes for output readback

    // Process in 2 groups of N=8 tokens each
    for (int n_group = 0; n_group < 2; n_group++) {
        int n_offset = n_group * 8;
        int32_t c_regs[4] = {0, 0, 0, 0};

        // Accumulate across 2 k-chunks (HEAD_DIM=128, 64 dims per MMA)
        for (int k_chunk = 0; k_chunk < 2; k_chunk++) {
            int k_off = k_chunk * 64;

            // ── Fill A: quantize Q into row 0, zero-pad rows 1..15 ──
            // Each row is 32 bytes = 64 nibbles. Row r occupies a_smem[r*32 .. r*32+31].
            // Nibble packing: byte j of row r holds A[r, 2j] in low nibble, A[r, 2j+1] in high.
            for (int i = lane; i < 16 * 32; i += WARP_SIZE) {
                int row = i / 32;
                int byte_idx = i % 32;
                uint8_t val = 0;
                if (row == 0) {
                    int d0 = k_off + byte_idx * 2;
                    int d1 = d0 + 1;
                    int8_t q0 = (d0 < HEAD_DIM) ? quantize_to_int4(q[d0], q_inv_scale) : 0;
                    int8_t q1 = (d1 < HEAD_DIM) ? quantize_to_int4(q[d1], q_inv_scale) : 0;
                    val = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
                }
                a_smem[i] = val;
            }

            // ── Fill B: K data in col-major INT4 ──
            // B[64,8] col-major: column c occupies b_smem[c*32 .. c*32+31].
            // byte j of column c holds B[2j, c] in low nibble, B[2j+1, c] in high.
            for (int i = lane; i < 8 * 32; i += WARP_SIZE) {
                int col = i / 32;    // N-dimension token
                int byte_idx = i % 32;
                int k0 = k_off + byte_idx * 2;
                int k1 = k0 + 1;
                int token = block_base + n_offset + col;

                uint8_t val = 0;
                if (token < seq_len) {
                    // Read codebook indices from nibble-packed K
                    // K packing: high nibble = even dim, low nibble = odd dim
                    auto read_idx = [&](int dim) -> uint8_t {
                        if (dim >= HEAD_DIM) return 0;
                        uint8_t packed = k_packed[token * (HEAD_DIM / 2) + dim / 2];
                        return (dim % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    };
                    // Convert unsigned [0..15] to signed [-8..+7]: XOR with 0x8
                    uint8_t s0 = read_idx(k0) ^ 0x8;
                    uint8_t s1 = read_idx(k1) ^ 0x8;
                    val = (s0 & 0xF) | ((s1 & 0xF) << 4);
                }
                b_smem[i] = val;
            }
            __syncwarp();

            // ── Load fragments from shared memory ──
            // A fragment: 4 registers, following the PTX layout
            // a[0] = A[tid, groupID*8..groupID*8+7] -> row tid, 8 nibbles = 4 bytes
            //         at a_smem[tid * 32 + groupID * 4]
            // a[1] = A[tid+8, groupID*8..groupID*8+7]
            //         at a_smem[(tid+8) * 32 + groupID * 4]
            // a[2] = A[tid, groupID*8+32..groupID*8+39]
            //         at a_smem[tid * 32 + groupID * 4 + 16]
            // a[3] = A[tid+8, groupID*8+32..groupID*8+39]
            //         at a_smem[(tid+8) * 32 + groupID * 4 + 16]
            int32_t a_reg[4];
            memcpy(&a_reg[0], &a_smem[tid * 32 + groupID * 4], 4);
            memcpy(&a_reg[1], &a_smem[(tid + 8) * 32 + groupID * 4], 4);
            memcpy(&a_reg[2], &a_smem[tid * 32 + groupID * 4 + 16], 4);
            memcpy(&a_reg[3], &a_smem[(tid + 8) * 32 + groupID * 4 + 16], 4);

            // B fragment: 2 registers
            // b[0] = B[groupID*8..groupID*8+7, tid] -> column tid, 8 nibbles = 4 bytes
            //         at b_smem[tid * 32 + groupID * 4]
            // b[1] = B[groupID*8+32..groupID*8+39, tid]
            //         at b_smem[tid * 32 + groupID * 4 + 16]
            int32_t b_reg[2];
            memcpy(&b_reg[0], &b_smem[tid * 32 + groupID * 4], 4);
            memcpy(&b_reg[1], &b_smem[tid * 32 + groupID * 4 + 16], 4);

            // ── MMA instruction ──
#if __CUDA_ARCH__ >= 800
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(c_regs[0]), "=r"(c_regs[1]), "=r"(c_regs[2]), "=r"(c_regs[3])
                : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1]),
                  "r"(c_regs[0]), "r"(c_regs[1]), "r"(c_regs[2]), "r"(c_regs[3])
            );
#endif
        } // k_chunk

        // ── Write accumulator to shared memory for readback ──
        // D layout: d[0]=D[tid, groupID], d[1]=D[tid+4, groupID],
        //           d[2]=D[tid+8, groupID], d[3]=D[tid+12, groupID]
        c_smem[tid * 8 + groupID] = c_regs[0];
        c_smem[(tid + 4) * 8 + groupID] = c_regs[1];
        c_smem[(tid + 8) * 8 + groupID] = c_regs[2];
        c_smem[(tid + 12) * 8 + groupID] = c_regs[3];
        __syncwarp();

        // Read row 0: D[0, n] for n=0..7
        if (lane < 8) {
            int token = block_base + n_offset + lane;
            if (token < seq_len) {
                int32_t raw = c_smem[0 * 8 + lane];
                float k_norm = k_norms[token];
                // Post-matmul correction: output = q_scale * codebook_scale * norm * raw
                output[token] = q_scale * codebook_scale * k_norm * static_cast<float>(raw);
            }
        }
        __syncwarp();
    } // n_group
}


// ────────────────────────────────────────────────────────────────────
// Kernel 3: INT8 tensor core QK dot product
// ────────────────────────────────────────────────────────────────────
//
// Uses mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
// M=16, N=8, K=32: processes 32 dim-elements per MMA call.
// For HEAD_DIM=128: 4 MMA calls.
//
// PTX fragment layout (m16n8k32, s8):
//   A (row-major [16,32] s8, 4 regs/thread):
//     a[0] = A[tid, groupID*4..groupID*4+3]        (4 s8 = 1 int32)
//     a[1] = A[tid+8, groupID*4..groupID*4+3]
//     a[2] = A[tid, groupID*4+16..groupID*4+19]
//     a[3] = A[tid+8, groupID*4+16..groupID*4+19]
//
//   B (col-major [32,8] s8, 2 regs/thread):
//     b[0] = B[groupID*4..groupID*4+3, tid]
//     b[1] = B[groupID*4+16..groupID*4+19, tid]
//
//   D: same as INT4 (d[0]=D[tid,groupID], etc.)

__global__ void int8_tc_qk_kernel(
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_packed,
    const float* __restrict__ k_norms,
    float* __restrict__ output,
    int seq_len,
    float codebook_scale,
    float q_scale,
    float q_inv_scale
) {
    int block_base = blockIdx.x * BLOCK_SEQ;
    int lane = threadIdx.x;
    int tid = lane & 3;
    int groupID = lane >> 2;

    // A: [16, 32] int8 = 512 bytes, B: [32, 8] int8 col-major = 256 bytes
    __shared__ int8_t a_smem[16 * 32];
    __shared__ int8_t b_smem[8 * 32];  // stored as [8 cols, 32 rows]
    __shared__ int32_t c_smem[16 * 8];

    for (int n_group = 0; n_group < 2; n_group++) {
        int n_offset = n_group * 8;
        int32_t c_regs[4] = {0, 0, 0, 0};

        for (int k_chunk = 0; k_chunk < 4; k_chunk++) {
            int k_off = k_chunk * 32;

            // Fill A: row 0 = quantized Q, rows 1..15 = zero
            for (int i = lane; i < 16 * 32; i += WARP_SIZE) {
                int row = i / 32;
                int col = i % 32;
                int8_t val = 0;
                if (row == 0) {
                    int d = k_off + col;
                    val = (d < HEAD_DIM) ? quantize_to_int8(q[d], q_inv_scale) : 0;
                }
                a_smem[i] = val;
            }

            // Fill B: K indices as signed int8 in col-major
            for (int i = lane; i < 8 * 32; i += WARP_SIZE) {
                int col = i / 32;    // token within N=8 group
                int k_pos = i % 32;  // K-dimension position
                int token = block_base + n_offset + col;
                int dim = k_off + k_pos;

                int8_t val = 0;
                if (token < seq_len && dim < HEAD_DIM) {
                    uint8_t packed = k_packed[token * (HEAD_DIM / 2) + dim / 2];
                    uint8_t idx = (dim % 2 == 0) ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    val = static_cast<int8_t>(idx) - 8;
                }
                b_smem[col * 32 + k_pos] = val;
            }
            __syncwarp();

            // Load fragments
            // A: a[0] at row=tid, cols [groupID*4..+3] = a_smem[tid*32 + groupID*4]
            int32_t a_reg[4];
            memcpy(&a_reg[0], &a_smem[tid * 32 + groupID * 4], 4);
            memcpy(&a_reg[1], &a_smem[(tid + 8) * 32 + groupID * 4], 4);
            memcpy(&a_reg[2], &a_smem[tid * 32 + groupID * 4 + 16], 4);
            memcpy(&a_reg[3], &a_smem[(tid + 8) * 32 + groupID * 4 + 16], 4);

            // B: b[0] at col=tid, rows [groupID*4..+3] = b_smem[tid*32 + groupID*4]
            int32_t b_reg[2];
            memcpy(&b_reg[0], &b_smem[tid * 32 + groupID * 4], 4);
            memcpy(&b_reg[1], &b_smem[tid * 32 + groupID * 4 + 16], 4);

#if __CUDA_ARCH__ >= 800
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=r"(c_regs[0]), "=r"(c_regs[1]), "=r"(c_regs[2]), "=r"(c_regs[3])
                : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1]),
                  "r"(c_regs[0]), "r"(c_regs[1]), "r"(c_regs[2]), "r"(c_regs[3])
            );
#endif
        } // k_chunk

        // Write and read back accumulator
        c_smem[tid * 8 + groupID] = c_regs[0];
        c_smem[(tid + 4) * 8 + groupID] = c_regs[1];
        c_smem[(tid + 8) * 8 + groupID] = c_regs[2];
        c_smem[(tid + 12) * 8 + groupID] = c_regs[3];
        __syncwarp();

        if (lane < 8) {
            int token = block_base + n_offset + lane;
            if (token < seq_len) {
                int32_t raw = c_smem[0 * 8 + lane];
                float k_norm = k_norms[token];
                output[token] = q_scale * codebook_scale * k_norm * static_cast<float>(raw);
            }
        }
        __syncwarp();
    } // n_group
}


// ────────────────────────────────────────────────────────────────────
// Host: benchmark driver
// ────────────────────────────────────────────────────────────────────

struct BenchConfig {
    int seq_len;
    int warmup;
    int iters;
};

float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return static_cast<float>(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

void run_benchmark(const BenchConfig& cfg) {
    printf("\n-- seq_len=%d --\n", cfg.seq_len);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int seq_len = cfg.seq_len;
    float codebook_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    // Lloyd-Max boundaries scaled
    const float boundaries[15] = {
        -2.4009919f*codebook_scale, -1.8436908f*codebook_scale,
        -1.4371418f*codebook_scale, -1.0993087f*codebook_scale,
        -0.7995637f*codebook_scale, -0.5224207f*codebook_scale,
        -0.2582488f*codebook_scale,  0.0000000f*codebook_scale,
         0.2582488f*codebook_scale,  0.5224207f*codebook_scale,
         0.7995637f*codebook_scale,  1.0993087f*codebook_scale,
         1.4371418f*codebook_scale,  1.8436908f*codebook_scale,
         2.4009919f*codebook_scale,
    };
    const float centroids[16] = {
        -2.7326368f*codebook_scale, -2.0693470f*codebook_scale,
        -1.6180345f*codebook_scale, -1.2562491f*codebook_scale,
        -0.9423684f*codebook_scale, -0.6567591f*codebook_scale,
        -0.3880823f*codebook_scale, -0.1284154f*codebook_scale,
         0.1284154f*codebook_scale,  0.3880823f*codebook_scale,
         0.6567591f*codebook_scale,  0.9423684f*codebook_scale,
         1.2562491f*codebook_scale,  1.6180345f*codebook_scale,
         2.0693470f*codebook_scale,  2.7326368f*codebook_scale,
    };

    // Generate Q and K
    std::vector<float> h_q(HEAD_DIM);
    for (auto& v : h_q) v = dist(rng);

    // Q scale for INT4/INT8 uniform quantization
    float q_absmax = 0.0f;
    for (auto& v : h_q) q_absmax = fmaxf(q_absmax, fabsf(v));
    float q_scale = q_absmax / 7.0f;
    float q_inv_scale = (q_scale > 0.0f) ? 1.0f / q_scale : 0.0f;

    std::vector<uint8_t> h_k_packed(seq_len * HEAD_DIM / 2, 0);
    std::vector<float> h_k_norms(seq_len);
    std::vector<float> h_k_deq(seq_len * HEAD_DIM);

    for (int t = 0; t < seq_len; t++) {
        std::vector<float> k_vec(HEAD_DIM);
        for (auto& v : k_vec) v = dist(rng);

        float norm = 0.0f;
        for (auto& v : k_vec) norm += v * v;
        norm = sqrtf(norm);
        h_k_norms[t] = norm;

        float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

        for (int d = 0; d < HEAD_DIM; d++) {
            float val = k_vec[d] * inv_norm;
            uint8_t idx = 15;
            for (int b = 0; b < 15; b++) {
                if (val < boundaries[b]) { idx = b; break; }
            }
            h_k_deq[t * HEAD_DIM + d] = centroids[idx] * norm;

            // Pack: high nibble = even dim, low nibble = odd dim
            int byte_pos = t * (HEAD_DIM / 2) + d / 2;
            if (d % 2 == 0) {
                h_k_packed[byte_pos] = (h_k_packed[byte_pos] & 0x0F) | (idx << 4);
            } else {
                h_k_packed[byte_pos] = (h_k_packed[byte_pos] & 0xF0) | (idx & 0x0F);
            }
        }
    }

    // CPU reference
    std::vector<float> ref_scores(seq_len);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            dot += h_q[d] * h_k_deq[t * HEAD_DIM + d];
        ref_scores[t] = dot;
    }

    // GPU allocations
    float *d_q, *d_k_norms, *d_out_scalar, *d_out_int4, *d_out_int8;
    uint8_t *d_k_packed;

    CUDA_CHECK(cudaMalloc(&d_q, HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_packed, seq_len * HEAD_DIM / 2));
    CUDA_CHECK(cudaMalloc(&d_k_norms, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_scalar, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_int4, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_int8, seq_len * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_packed, h_k_packed.data(), seq_len * HEAD_DIM / 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_norms, h_k_norms.data(), seq_len * sizeof(float), cudaMemcpyHostToDevice));

    // Scalar kernel config
    int warps_per_block = 4;
    int scalar_threads = warps_per_block * WARP_SIZE;
    int tokens_per_block = warps_per_block * BLOCK_SEQ;
    int scalar_blocks = (seq_len + tokens_per_block - 1) / tokens_per_block;

    // TC kernel config: one warp per block, each block handles BLOCK_SEQ=16 tokens
    int tc_threads = WARP_SIZE;
    int tc_blocks = (seq_len + BLOCK_SEQ - 1) / BLOCK_SEQ;

    // ── Benchmark scalar ──
    for (int i = 0; i < cfg.warmup; i++)
        scalar_qk_kernel<<<scalar_blocks, scalar_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_scalar, seq_len, codebook_scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; i++)
        scalar_qk_kernel<<<scalar_blocks, scalar_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_scalar, seq_len, codebook_scale);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float scalar_ms;
    CUDA_CHECK(cudaEventElapsedTime(&scalar_ms, start, stop));
    float scalar_us = scalar_ms * 1000.0f / cfg.iters;

    // ── Benchmark INT4 TC ──
    for (int i = 0; i < cfg.warmup; i++)
        int4_tc_qk_kernel<<<tc_blocks, tc_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_int4, seq_len,
            codebook_scale, q_scale, q_inv_scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; i++)
        int4_tc_qk_kernel<<<tc_blocks, tc_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_int4, seq_len,
            codebook_scale, q_scale, q_inv_scale);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float int4_ms;
    CUDA_CHECK(cudaEventElapsedTime(&int4_ms, start, stop));
    float int4_us = int4_ms * 1000.0f / cfg.iters;

    // ── Benchmark INT8 TC ──
    for (int i = 0; i < cfg.warmup; i++)
        int8_tc_qk_kernel<<<tc_blocks, tc_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_int8, seq_len,
            codebook_scale, q_scale, q_inv_scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; i++)
        int8_tc_qk_kernel<<<tc_blocks, tc_threads>>>(
            d_q, d_k_packed, d_k_norms, d_out_int8, seq_len,
            codebook_scale, q_scale, q_inv_scale);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float int8_ms;
    CUDA_CHECK(cudaEventElapsedTime(&int8_ms, start, stop));
    float int8_us = int8_ms * 1000.0f / cfg.iters;

    // ── Correctness ──
    std::vector<float> h_scalar(seq_len), h_int4(seq_len), h_int8(seq_len);
    CUDA_CHECK(cudaMemcpy(h_scalar.data(), d_out_scalar, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_int4.data(), d_out_int4, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_int8.data(), d_out_int8, seq_len * sizeof(float), cudaMemcpyDeviceToHost));

    float cos_scalar = cosine_sim(h_scalar, ref_scores);
    float cos_int4 = cosine_sim(h_int4, ref_scores);
    float cos_int8 = cosine_sim(h_int8, ref_scores);

    printf("  %-20s %8.2f us  cos=%f\n", "Scalar FMA:", scalar_us, cos_scalar);
    printf("  %-20s %8.2f us  cos=%f  speedup=%.2fx\n",
           "INT4 TC (PTX):", int4_us, cos_int4, scalar_us / int4_us);
    printf("  %-20s %8.2f us  cos=%f  speedup=%.2fx\n",
           "INT8 TC (PTX):", int8_us, cos_int8, scalar_us / int8_us);

    // Sample scores
    printf("  First 4 scores (ref / scalar / int4 / int8):\n");
    for (int i = 0; i < 4 && i < seq_len; i++) {
        printf("    [%d] %8.3f / %8.3f / %8.3f / %8.3f\n",
               i, ref_scores[i], h_scalar[i], h_int4[i], h_int8[i]);
    }

    // Cleanup
    cudaFree(d_q); cudaFree(d_k_packed); cudaFree(d_k_norms);
    cudaFree(d_out_scalar); cudaFree(d_out_int4); cudaFree(d_out_int8);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("=== HYP-019: INT4 Tensor Core QK Benchmark ===\n");
    printf("HEAD_DIM=%d, BLOCK_SEQ=%d\n", HEAD_DIM, BLOCK_SEQ);
    printf("\nKernels:\n");
    printf("  1. Scalar FMA + warp shuffle (matches TQ v4)\n");
    printf("  2. INT4 tensor cores (mma.m16n8k64.s4.s4.s32)\n");
    printf("  3. INT8 tensor cores (mma.m16n8k32.s8.s8.s32)\n");
    printf("\nNote: TC kernels use uniform INT4/INT8 quantization of Q.\n");
    printf("Cosine measures agreement with Lloyd-Max dequantized reference.\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major < 8) {
        printf("ERROR: INT4/INT8 tensor cores require SM >= 8.0 (A100+)\n");
        return 1;
    }

    BenchConfig configs[] = {
        {128,  50, 500},
        {512,  50, 500},
        {1024, 50, 500},
        {4096, 30, 200},
    };

    for (const auto& cfg : configs) {
        run_benchmark(cfg);
    }

    printf("\n=== Done ===\n");
    return 0;
}
