#include "turboquant/test_framework.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/pack.h"
#include "turboquant/quantize.h"
#include "turboquant/tile.h"

#include <cmath>
#include <random>
#include <vector>

using namespace turboquant;

static constexpr int kTileSize = TurboQuantTile::kTokens * TurboQuantTile::kDims;
static constexpr int kDim = TurboQuantTile::kDims;

static float row_relative_rmse(const float* src, const float* dst, int dims) {
    float mse = 0.0f, norm_sq = 0.0f;
    for (int d = 0; d < dims; d++) {
        float err = dst[d] - src[d];
        mse += err * err;
        norm_sq += src[d] * src[d];
    }
    if (norm_sq < 1e-12f) return 0.0f;
    return std::sqrt(mse / norm_sq);
}

static float row_cosine_similarity(const float* src, const float* dst, int dims) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int d = 0; d < dims; d++) {
        dot += src[d] * dst[d];
        norm_a += src[d] * src[d];
        norm_b += dst[d] * dst[d];
    }
    if (norm_a < 1e-12f || norm_b < 1e-12f) return 1.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

TEST(Roundtrip, AllZeros) {
    std::vector<float> input(kTileSize, 0.0f), output(kTileSize);
    TurboQuantTile tile{};

    quantize_tile(input.data(), kDim, tile);
    dequantize_tile(tile, kDim, output.data());
    for (int i = 0; i < kTileSize; i++) EXPECT_FLOAT_EQ(output[i], 0.0f);
}

TEST(Roundtrip, UniformRandom) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), kDim, tile);
    dequantize_tile(tile, kDim, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float cos = row_cosine_similarity(
            input.data() + row * kDim,
            output.data() + row * kDim,
            kDim);
        EXPECT_GT(cos, 0.85f);
    }
}

TEST(Roundtrip, GaussianRandom) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 5.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), kDim, tile);
    dequantize_tile(tile, kDim, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float cos = row_cosine_similarity(
            input.data() + row * kDim,
            output.data() + row * kDim,
            kDim);
        EXPECT_GT(cos, 0.80f);
    }
}

TEST(Roundtrip, MultipleSeeds100) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), kDim, tile);
        dequantize_tile(tile, kDim, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            float rel = row_relative_rmse(
                input.data() + row * kDim,
                output.data() + row * kDim,
                kDim);
            EXPECT_LT(rel, 0.25f);
        }
    }
}

TEST(Roundtrip, LargeMagnitude) {
    // Large magnitude uniform data: codebook is designed for post-Hadamard
    // N(0,1/d) distribution. Raw uniform data has heavier tails, so expect
    // lower quality. The fused kernel will Hadamard-rotate first.
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), kDim, tile);
    dequantize_tile(tile, kDim, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float cos = row_cosine_similarity(
            input.data() + row * kDim,
            output.data() + row * kDim,
            kDim);
        EXPECT_GT(cos, 0.60f);
    }
}

TEST(Roundtrip, NormIsL2) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), kDim, tile);

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float l2_sq = 0.0f;
        for (int d = 0; d < kDim; d++) {
            float v = input[row * kDim + d];
            l2_sq += v * v;
        }
        float l2_norm = std::sqrt(l2_sq);
        float stored_norm = fp16_to_float(tile.norms[row]);
        // FP16 rounding tolerance
        EXPECT_LT(std::fabs(stored_norm - l2_norm) / std::max(l2_norm, 1e-6f), 0.01f);
    }
}

TEST(Roundtrip, IndicesInRange) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : input) v = dist(rng);

    quantize_tile(input.data(), kDim, tile);

    // Check 4-bit indices are 0..15
    uint8_t hi_indices[TurboQuantTile::kHiDims];
    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        unpack_4bit(tile.quant_hi + row * TurboQuantTile::kHiBytesPerRow,
                    TurboQuantTile::kHiDims, hi_indices);
        for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
            EXPECT_LT(hi_indices[d], 16);
        }
    }

    // Check 3-bit indices are 0..7
    uint8_t lo_indices[TurboQuantTile::kLoDims];
    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        unpack_3bit(tile.quant_lo + row * TurboQuantTile::kLoBytesPerRow,
                    TurboQuantTile::kLoDims, lo_indices);
        for (int d = 0; d < TurboQuantTile::kLoDims; d++) {
            EXPECT_LT(lo_indices[d], 8);
        }
    }
}

TEST(Dequant, HiLoChannelAccuracy) {
    // 4-bit (16 levels) should have lower error than 3-bit (8 levels)
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    float hi_mse_total = 0.0f, hi_norm_total = 0.0f;
    float lo_mse_total = 0.0f, lo_norm_total = 0.0f;

    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed + 2000);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), kDim, tile);
        dequantize_tile(tile, kDim, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                int idx = row * kDim + d;
                float e = output[idx] - input[idx];
                hi_mse_total += e * e;
                hi_norm_total += input[idx] * input[idx];
            }
            for (int d = TurboQuantTile::kHiDims; d < kDim; d++) {
                int idx = row * kDim + d;
                float e = output[idx] - input[idx];
                lo_mse_total += e * e;
                lo_norm_total += input[idx] * input[idx];
            }
        }
    }

    float hi_rel_rmse = std::sqrt(hi_mse_total / hi_norm_total);
    float lo_rel_rmse = std::sqrt(lo_mse_total / lo_norm_total);

    // 4-bit (16 levels) must have lower error than 3-bit (8 levels)
    EXPECT_LT(hi_rel_rmse, lo_rel_rmse);
}
