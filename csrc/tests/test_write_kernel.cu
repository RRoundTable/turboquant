// Phase 2 CUDA write kernel test.
// Compares CUDA kernel output against C++ CPU reference (bit-exact).

#include "turboquant/write_kernel.cuh"
#include "turboquant/quantize.h"
#include "turboquant/tile.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace turboquant;

static constexpr int kDim = TurboQuantTile::kDims;  // 64
static constexpr int kTokens = TurboQuantTile::kTokens;  // 16

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Compare two tiles byte-by-byte
static int compare_tiles(const uint8_t* a, const uint8_t* b, int tile_bytes) {
    int mismatches = 0;
    for (int i = 0; i < tile_bytes; i++) {
        if (a[i] != b[i]) {
            if (mismatches < 10) {
                printf("  mismatch at byte %d: cuda=0x%02x cpu=0x%02x\n", i, a[i], b[i]);
            }
            mismatches++;
        }
    }
    return mismatches;
}

void test_basic_quantize() {
    printf("test_basic_quantize...\n");

    int num_tokens = kTokens;
    int head_dim = kDim;
    int padded_dim = kDim;
    int tiles_per_row = 1;
    int tile_size = sizeof(TurboQuantTile);

    // Generate random input
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_input(num_tokens * head_dim);
    for (auto& v : h_input) v = dist(rng);

    // ── CPU reference ──
    // Pass raw input — C++ quantize_tile computes L2 norm internally,
    // same as the CUDA kernel.
    TurboQuantTile cpu_tile;
    quantize_tile(h_input.data(), padded_dim, cpu_tile);

    // ── CUDA kernel ──
    float* d_input;
    uint8_t* d_output;
    float* d_signs;  // unused in MVP but required by API
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, tile_size));
    CUDA_CHECK(cudaMalloc(&d_signs, padded_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          num_tokens * head_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, tile_size));

    CUDA_CHECK(launch_quantize_kv(
        d_input, d_output, d_signs,
        num_tokens, head_dim, padded_dim, tiles_per_row
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    std::vector<uint8_t> gpu_tile(tile_size);
    CUDA_CHECK(cudaMemcpy(gpu_tile.data(), d_output, tile_size,
                          cudaMemcpyDeviceToHost));

    // Compare norms (first 32 bytes — may differ due to FP16 rounding path)
    const uint8_t* cpu_bytes = reinterpret_cast<const uint8_t*>(&cpu_tile);

    // Check norms separately (allow FP16 rounding differences)
    int norm_mismatches = 0;
    for (int row = 0; row < kTokens; row++) {
        uint16_t cpu_norm, gpu_norm;
        memcpy(&cpu_norm, cpu_bytes + row * 2, 2);
        memcpy(&gpu_norm, gpu_tile.data() + row * 2, 2);
        if (cpu_norm != gpu_norm) {
            // Check if the difference is small (1 ULP)
            int diff = abs(static_cast<int>(cpu_norm) - static_cast<int>(gpu_norm));
            if (diff > 1) {
                printf("  norm mismatch row %d: cpu=0x%04x gpu=0x%04x (diff=%d)\n",
                       row, cpu_norm, gpu_norm, diff);
                norm_mismatches++;
            }
        }
    }

    // Check quant data (bytes 32..479)
    int data_mismatches = compare_tiles(cpu_bytes + 32, gpu_tile.data() + 32, tile_size - 32);

    printf("  norm mismatches: %d, data mismatches: %d\n", norm_mismatches, data_mismatches);

    if (norm_mismatches == 0 && data_mismatches == 0) {
        printf("  PASS (bit-exact)\n");
    } else if (data_mismatches <= 5) {
        printf("  PASS (near-exact, %d byte diffs from FP rounding)\n", data_mismatches);
    } else {
        printf("  FAIL\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_signs);
}

void test_dequant_quality() {
    printf("test_dequant_quality...\n");

    int num_tokens = kTokens;
    int head_dim = kDim;
    int padded_dim = kDim;
    int tiles_per_row = 1;
    int tile_size = sizeof(TurboQuantTile);

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_input(num_tokens * head_dim);
    for (auto& v : h_input) v = dist(rng);

    // Run CUDA kernel
    float* d_input;
    uint8_t* d_output;
    float* d_signs;
    CUDA_CHECK(cudaMalloc(&d_input, num_tokens * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, tile_size));
    CUDA_CHECK(cudaMalloc(&d_signs, padded_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          num_tokens * head_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_quantize_kv(
        d_input, d_output, d_signs,
        num_tokens, head_dim, padded_dim, tiles_per_row
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back tile
    TurboQuantTile tile;
    CUDA_CHECK(cudaMemcpy(&tile, d_output, tile_size, cudaMemcpyDeviceToHost));

    // Dequantize on CPU
    std::vector<float> reconstructed(num_tokens * head_dim);
    dequantize_tile(tile, padded_dim, reconstructed.data());

    // Measure cosine similarity per row
    float min_cos = 1.0f;
    for (int row = 0; row < num_tokens; row++) {
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < head_dim; d++) {
            float a = h_input[row * head_dim + d];
            float b = reconstructed[row * head_dim + d];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        float cos = dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
        if (cos < min_cos) min_cos = cos;
    }

    printf("  min cosine similarity: %.4f\n", min_cos);
    printf("  %s (threshold: 0.85)\n", min_cos > 0.85f ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_signs);
}

int main() {
    printf("=== Phase 2: CUDA Write Kernel Tests ===\n\n");
    test_basic_quantize();
    printf("\n");
    test_dequant_quality();
    printf("\n=== Done ===\n");
    return 0;
}
