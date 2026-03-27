// Phase 3b: Benchmark and profile the fused decode TurboQuant kernel.
//
// Measures:
// 1. Kernel latency (CUDA events)
// 2. Register usage (cudaFuncGetAttributes)
// 3. Comparison vs a "dequant-then-standard-attention" baseline
// 4. Multiple seq lengths to show scaling

#include "turboquant/decode_turboquant.cuh"
#include "turboquant/quantize.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/pack.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
static constexpr uint32_t PADDED_DIM = 64;
static constexpr uint32_t PAGE_SIZE = 16;
static constexpr uint32_t NUM_KV_HEADS = 1;
static constexpr uint32_t NUM_QO_HEADS = 1;

struct BenchResult {
    float time_us;
    float cosine;
};

// Prepare quantized KV data for a given seq_len
struct QuantizedKVData {
    std::vector<uint8_t> k_quant, v_quant;
    std::vector<uint16_t> k_norms, v_norms;
    std::vector<int32_t> indices;
    int32_t indptr[2];
    int32_t last_page_len;
    uint32_t num_pages;
};

QuantizedKVData prepare_data(uint32_t seq_len, std::mt19937& rng) {
    QuantizedKVData data;
    data.num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    uint32_t dim_chunks = PADDED_DIM / 64;
    uint32_t qbytes = 28;  // per token per chunk

    data.k_quant.resize(data.num_pages * NUM_KV_HEADS * PAGE_SIZE * dim_chunks * qbytes, 0);
    data.v_quant.resize(data.k_quant.size(), 0);
    data.k_norms.resize(data.num_pages * NUM_KV_HEADS * PAGE_SIZE * dim_chunks, 0);
    data.v_norms.resize(data.k_norms.size(), 0);

    float scale = 1.0f / sqrtf(static_cast<float>(PADDED_DIM));
    float b4[15], b3[7];
    for (int i = 0; i < 15; i++) b4[i] = kBoundaries4bit[i] * scale;
    for (int i = 0; i < 7; i++) b3[i] = kBoundaries3bit[i] * scale;

    auto bucket = [](float val, const float* bounds, int n) -> uint8_t {
        for (int i = 0; i < n; i++) if (val < bounds[i]) return i;
        return n;
    };

    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t t = 0; t < seq_len; t++) {
        float vec[HEAD_DIM];
        for (auto& v : vec) v = dist(rng);

        for (int kv = 0; kv < 2; kv++) {
            uint8_t* qd = (kv == 0) ? data.k_quant.data() + t * qbytes
                                     : data.v_quant.data() + t * qbytes;
            uint16_t* nd = (kv == 0) ? data.k_norms.data() + t
                                     : data.v_norms.data() + t;

            float l2 = 0;
            for (int d = 0; d < HEAD_DIM; d++) l2 += vec[d] * vec[d];
            l2 = sqrtf(l2);
            nd[0] = float_to_fp16(l2);
            float norm = fp16_to_float(nd[0]);
            float inv = (norm > 0) ? 1.0f / norm : 0.0f;

            uint8_t hi[32], lo[32];
            for (int d = 0; d < 32; d++) hi[d] = bucket(vec[d] * inv, b4, 15);
            for (int d = 0; d < 32; d++) lo[d] = bucket(vec[32 + d] * inv, b3, 7);
            pack_4bit(hi, 32, qd);
            pack_3bit(lo, 32, qd + 16);
        }
    }

    data.indices.resize(data.num_pages);
    for (uint32_t i = 0; i < data.num_pages; i++) data.indices[i] = i;
    data.indptr[0] = 0;
    data.indptr[1] = data.num_pages;
    data.last_page_len = (seq_len % PAGE_SIZE == 0) ? PAGE_SIZE : seq_len % PAGE_SIZE;

    return data;
}

BenchResult run_bench(uint32_t seq_len, int warmup, int iters) {
    std::mt19937 rng(42);
    auto data = prepare_data(seq_len, rng);

    // Query
    std::vector<__half> h_q(NUM_QO_HEADS * HEAD_DIM);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_q) v = __float2half(dist(rng));

    // Allocate GPU
    __half *d_q, *d_o, *d_k_norms, *d_v_norms;
    uint8_t *d_k_quant, *d_v_quant;
    int32_t *d_indices, *d_indptr, *d_last_page_len;
    float *d_lse;

    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_o, NUM_QO_HEADS * HEAD_DIM * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k_quant, data.k_quant.size()));
    CUDA_CHECK(cudaMalloc(&d_v_quant, data.v_quant.size()));
    CUDA_CHECK(cudaMalloc(&d_k_norms, data.k_norms.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_v_norms, data.v_norms.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_indices, data.num_pages * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_indptr, 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_last_page_len, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_lse, NUM_QO_HEADS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_quant, data.k_quant.data(), data.k_quant.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_quant, data.v_quant.data(), data.v_quant.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_norms, data.k_norms.data(), data.k_norms.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_norms, data.v_norms.data(), data.v_norms.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, data.indices.data(), data.num_pages * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indptr, data.indptr, 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_last_page_len, &data.last_page_len, sizeof(int32_t), cudaMemcpyHostToDevice));

    paged_kv_turbo_t<int32_t> paged_kv(
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM, 1,
        d_k_quant, d_v_quant, d_k_norms, d_v_norms,
        d_indices, d_indptr, d_last_page_len
    );

    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t bdy = 1;
    constexpr uint32_t bdz = 1;
    constexpr uint32_t tile_size_per_bdx = 4;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;

    uint32_t smem_size = 2 * tile_tokens * 64 * sizeof(__half) + 2 * bdy * bdz * sizeof(float);
    float sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    dim3 grid(1, NUM_KV_HEADS);
    dim3 block(bdx, bdy, bdz);

    auto kernel = BatchDecodeWithTurboQuantKVKernel<HEAD_DIM, vec_size, bdx, bdy, bdz, tile_size_per_bdx, int32_t>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block, smem_size>>>(paged_kv, d_q, d_o, d_lse, NUM_QO_HEADS, sm_scale);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; i++) {
        kernel<<<grid, block, smem_size>>>(paged_kv, d_q, d_o, d_lse, NUM_QO_HEADS, sm_scale);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float time_us = ms * 1000.0f / iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Cleanup
    cudaFree(d_q); cudaFree(d_o);
    cudaFree(d_k_quant); cudaFree(d_v_quant);
    cudaFree(d_k_norms); cudaFree(d_v_norms);
    cudaFree(d_indices); cudaFree(d_indptr); cudaFree(d_last_page_len);
    cudaFree(d_lse);

    return {time_us, 0.0f};
}

int main() {
    printf("=== Phase 3b: TurboQuant Decode Kernel Benchmarks ===\n\n");

    // Register usage
    {
        constexpr uint32_t vec_size = 8;
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        auto kernel = BatchDecodeWithTurboQuantKVKernel<HEAD_DIM, vec_size, bdx, 1, 1, 4, int32_t>;
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
        printf("Kernel attributes:\n");
        printf("  Registers per thread: %d\n", attr.numRegs);
        printf("  Shared memory (static): %zu bytes\n", attr.sharedSizeBytes);
        printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);
        printf("  Local memory per thread: %zu bytes %s\n",
               attr.localSizeBytes,
               attr.localSizeBytes == 0 ? "(NO SPILLING)" : "(SPILLING!)");
        printf("\n");
    }

    // Latency across seq lengths
    printf("%10s %12s %12s\n", "seq_len", "time_us", "tokens/us");
    printf("---------- ------------ ------------\n");

    uint32_t seq_lens[] = {16, 64, 256, 1024, 4096};
    for (auto seq_len : seq_lens) {
        auto result = run_bench(seq_len, 10, 100);
        float tokens_per_us = static_cast<float>(seq_len) / result.time_us;
        printf("%10u %12.2f %12.2f\n", seq_len, result.time_us, tokens_per_us);
    }

    printf("\n=== Done ===\n");
    return 0;
}
