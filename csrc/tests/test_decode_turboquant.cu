// Phase 3a: Test fused decode attention with TurboQuant dequantization.
// Injects dummy pre-compressed KV tiles and compares output against
// a CPU reference (brute-force attention on dequantized KV).

#include "turboquant/decode_turboquant.cuh"
#include "turboquant/quantize.h"
#include "turboquant/tile.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/pack.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace turboquant;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static constexpr uint32_t HEAD_DIM = 64;
static constexpr uint32_t PADDED_DIM = 64;  // power of 2
static constexpr uint32_t PAGE_SIZE = 16;
static constexpr uint32_t NUM_KV_HEADS = 1;
static constexpr uint32_t NUM_QO_HEADS = 1;

// CPU reference: brute-force attention
void cpu_attention(
    const float* q,      // [num_qo_heads, head_dim]
    const float* k,      // [seq_len, head_dim]
    const float* v,      // [seq_len, head_dim]
    float* o,            // [num_qo_heads, head_dim]
    uint32_t seq_len,
    uint32_t head_dim,
    float sm_scale
) {
    // Compute scores
    std::vector<float> scores(seq_len);
    for (uint32_t s = 0; s < seq_len; s++) {
        float dot = 0;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += q[d] * k[s * head_dim + d];
        }
        scores[s] = dot * sm_scale;
    }
    // Softmax
    float max_s = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0;
    for (auto& s : scores) {
        s = expf(s - max_s);
        sum_exp += s;
    }
    for (auto& s : scores) s /= sum_exp;

    // Weighted sum
    for (uint32_t d = 0; d < head_dim; d++) {
        float val = 0;
        for (uint32_t s = 0; s < seq_len; s++) {
            val += scores[s] * v[s * head_dim + d];
        }
        o[d] = val;
    }
}

void test_basic_decode() {
    printf("test_basic_decode (seq_len=16, head_dim=64)...\n");

    uint32_t seq_len = 16;
    uint32_t num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;  // 1 page
    uint32_t dim_chunks = PADDED_DIM / 64;  // 1
    uint32_t quant_bytes_per_chunk = 28;  // 16 (hi) + 12 (lo)
    uint32_t quant_bytes_per_token = dim_chunks * quant_bytes_per_chunk;

    // Generate random K, V as float
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_k(seq_len * HEAD_DIM), h_v(seq_len * HEAD_DIM);
    std::vector<float> h_q(NUM_QO_HEADS * HEAD_DIM);
    for (auto& v : h_k) v = dist(rng);
    for (auto& v : h_v) v = dist(rng);
    for (auto& v : h_q) v = dist(rng);

    // Quantize K and V using C++ reference
    // Quantize each token's 64 dims as one tile row
    std::vector<uint8_t> h_k_quant(num_pages * NUM_KV_HEADS * PAGE_SIZE * quant_bytes_per_token, 0);
    std::vector<uint8_t> h_v_quant(h_k_quant.size(), 0);
    std::vector<uint16_t> h_k_norms_u16(num_pages * NUM_KV_HEADS * PAGE_SIZE * dim_chunks, 0);
    std::vector<uint16_t> h_v_norms_u16(h_k_norms_u16.size(), 0);

    float scale = 1.0f / sqrtf(static_cast<float>(PADDED_DIM));

    // Pre-scale boundaries
    float boundaries_4bit[15], boundaries_3bit[7];
    for (int i = 0; i < 15; i++) boundaries_4bit[i] = kBoundaries4bit[i] * scale;
    for (int i = 0; i < 7; i++) boundaries_3bit[i] = kBoundaries3bit[i] * scale;

    auto bucketize = [](float val, const float* bounds, int n) -> uint8_t {
        for (int i = 0; i < n; i++) if (val < bounds[i]) return i;
        return n;
    };

    for (uint32_t token = 0; token < seq_len; token++) {
        for (int kv = 0; kv < 2; kv++) {
            const float* src = (kv == 0) ? h_k.data() + token * HEAD_DIM
                                         : h_v.data() + token * HEAD_DIM;
            uint8_t* quant_dst = (kv == 0) ? h_k_quant.data() + token * quant_bytes_per_token
                                           : h_v_quant.data() + token * quant_bytes_per_token;
            uint16_t* norm_dst = (kv == 0) ? h_k_norms_u16.data() + token * dim_chunks
                                           : h_v_norms_u16.data() + token * dim_chunks;

            // L2 norm
            float l2_sq = 0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) l2_sq += src[d] * src[d];
            float l2 = sqrtf(l2_sq);
            norm_dst[0] = float_to_fp16(l2);
            float norm = fp16_to_float(norm_dst[0]);
            float inv_norm = (norm > 0) ? 1.0f / norm : 0.0f;

            // Quantize hi dims (4-bit)
            uint8_t hi_indices[32];
            for (uint32_t d = 0; d < 32; d++) {
                hi_indices[d] = bucketize(src[d] * inv_norm, boundaries_4bit, 15);
            }
            pack_4bit(hi_indices, 32, quant_dst);

            // Quantize lo dims (3-bit)
            uint8_t lo_indices[32];
            for (uint32_t d = 0; d < 32; d++) {
                lo_indices[d] = bucketize(src[32 + d] * inv_norm, boundaries_3bit, 7);
            }
            pack_3bit(lo_indices, 32, quant_dst + 16);
        }
    }

    // Dequantize K, V on CPU for reference attention
    float centroids_4bit[16], centroids_3bit[8];
    for (int i = 0; i < 16; i++) centroids_4bit[i] = kCentroids4bit[i] * scale;
    for (int i = 0; i < 8; i++) centroids_3bit[i] = kCentroids3bit[i] * scale;

    std::vector<float> h_k_deq(seq_len * HEAD_DIM), h_v_deq(seq_len * HEAD_DIM);
    for (uint32_t token = 0; token < seq_len; token++) {
        for (int kv = 0; kv < 2; kv++) {
            const uint8_t* qdata = (kv == 0) ? h_k_quant.data() + token * quant_bytes_per_token
                                             : h_v_quant.data() + token * quant_bytes_per_token;
            const uint16_t* norm_u16 = (kv == 0) ? h_k_norms_u16.data() + token
                                                  : h_v_norms_u16.data() + token;
            float* dst = (kv == 0) ? h_k_deq.data() + token * HEAD_DIM
                                   : h_v_deq.data() + token * HEAD_DIM;
            float norm = fp16_to_float(norm_u16[0]);

            // Hi dims
            uint8_t hi_idx[32];
            unpack_4bit(qdata, 32, hi_idx);
            for (uint32_t d = 0; d < 32; d++) {
                dst[d] = centroids_4bit[hi_idx[d]] * norm;
            }
            // Lo dims
            uint8_t lo_idx[32];
            unpack_3bit(qdata + 16, 32, lo_idx);
            for (uint32_t d = 0; d < 32; d++) {
                dst[32 + d] = centroids_3bit[lo_idx[d]] * norm;
            }
        }
    }

    // CPU reference attention on dequantized KV
    float sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    std::vector<float> h_o_ref(NUM_QO_HEADS * HEAD_DIM);
    cpu_attention(h_q.data(), h_k_deq.data(), h_v_deq.data(), h_o_ref.data(),
                  seq_len, HEAD_DIM, sm_scale);

    // Allocate GPU memory
    __half* d_q; __half* d_o;
    uint8_t* d_k_quant; uint8_t* d_v_quant;
    __half* d_k_norms; __half* d_v_norms;
    int32_t* d_indices; int32_t* d_indptr; int32_t* d_last_page_len;
    float* d_lse;

    CUDA_CHECK(cudaMalloc(&d_q, NUM_QO_HEADS * HEAD_DIM * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_o, NUM_QO_HEADS * HEAD_DIM * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k_quant, h_k_quant.size()));
    CUDA_CHECK(cudaMalloc(&d_v_quant, h_v_quant.size()));
    CUDA_CHECK(cudaMalloc(&d_k_norms, h_k_norms_u16.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_v_norms, h_v_norms_u16.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_pages * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_indptr, 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_last_page_len, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_lse, NUM_QO_HEADS * sizeof(float)));

    // Convert Q to fp16 and upload
    std::vector<__half> h_q_fp16(NUM_QO_HEADS * HEAD_DIM);
    for (size_t i = 0; i < h_q_fp16.size(); i++) h_q_fp16[i] = __float2half(h_q[i]);
    CUDA_CHECK(cudaMemcpy(d_q, h_q_fp16.data(), h_q_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_k_quant, h_k_quant.data(), h_k_quant.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_quant, h_v_quant.data(), h_v_quant.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_norms, h_k_norms_u16.data(), h_k_norms_u16.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_norms, h_v_norms_u16.data(), h_v_norms_u16.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Page table: identity mapping (page 0 → physical page 0)
    std::vector<int32_t> h_indices(num_pages);
    for (uint32_t i = 0; i < num_pages; i++) h_indices[i] = i;
    int32_t h_indptr[2] = {0, (int32_t)num_pages};
    int32_t h_last_page_len = seq_len % PAGE_SIZE == 0 ? PAGE_SIZE : seq_len % PAGE_SIZE;

    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_pages * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_last_page_len, &h_last_page_len, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_o, 0, NUM_QO_HEADS * HEAD_DIM * sizeof(__half)));

    // Launch kernel
    constexpr uint32_t vec_size = 8;  // HEAD_DIM=64: max(16/2, 64/32) = 8
    constexpr uint32_t bdx = HEAD_DIM / vec_size;  // 8
    constexpr uint32_t bdy = 1;  // GQA group = 1
    constexpr uint32_t bdz = 1;  // single warp for simplicity
    constexpr uint32_t tile_size_per_bdx = 4;

    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;
    uint32_t smem_size = 2 * tile_tokens * 64 * sizeof(__half) + 2 * bdy * bdz * sizeof(float);

    dim3 grid(1, NUM_KV_HEADS);  // 1 batch
    dim3 block(bdx, bdy, bdz);

    // Construct paged_kv on host
    paged_kv_turbo_t<int32_t> h_paged_kv(
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM, 1,
        d_k_quant, d_v_quant, d_k_norms, d_v_norms,
        d_indices, d_indptr, d_last_page_len
    );

    auto kernel = BatchDecodeWithTurboQuantKVKernel<HEAD_DIM, vec_size, bdx, bdy, bdz, tile_size_per_bdx, int32_t>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<grid, block, smem_size>>>(
        h_paged_kv, d_q, d_o, d_lse,
        NUM_QO_HEADS, sm_scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back output
    std::vector<__half> h_o_fp16(NUM_QO_HEADS * HEAD_DIM);
    CUDA_CHECK(cudaMemcpy(h_o_fp16.data(), d_o, h_o_fp16.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    // Compare
    float dot = 0, na = 0, nb = 0;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        float a = h_o_ref[d];
        float b = __half2float(h_o_fp16[d]);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    float cos_sim = dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);

    float max_diff = 0;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        float diff = fabsf(h_o_ref[d] - __half2float(h_o_fp16[d]));
        max_diff = fmaxf(max_diff, diff);
    }

    printf("  cosine similarity: %.6f\n", cos_sim);
    printf("  max abs diff: %.6f\n", max_diff);
    printf("  %s (threshold: cos > 0.99, atol < 1e-2)\n",
           (cos_sim > 0.99f && max_diff < 1e-2f) ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_q); cudaFree(d_o);
    cudaFree(d_k_quant); cudaFree(d_v_quant);
    cudaFree(d_k_norms); cudaFree(d_v_norms);
    cudaFree(d_indices); cudaFree(d_indptr); cudaFree(d_last_page_len);
    cudaFree(d_lse);
}

int main() {
    printf("=== Phase 3a: TurboQuant Decode Fusion Test ===\n\n");
    test_basic_decode();
    printf("\n=== Done ===\n");
    return 0;
}
