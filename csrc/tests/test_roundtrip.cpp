#include "turboquant/test_framework.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/quantize.h"
#include "turboquant/tile.h"

#include <cmath>
#include <random>
#include <vector>

using namespace turboquant;

static constexpr int kTileSize = TurboQuantTile::kTokens * TurboQuantTile::kDims;

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

// Magnitude levels for a given dimension
static int mag_levels_for_dim(int d) {
    return (d < TurboQuantTile::kHiDims)
        ? TurboQuantTile::kHiMagLevels   // 8
        : TurboQuantTile::kLoMagLevels;  // 4
}

// Max per-value absolute error: norm * 0.5 / mag_levels + FP16 headroom
static float max_error_for_dim(int d, float norm) {
    int N = mag_levels_for_dim(d);
    float quant_err = norm * 0.5f / static_cast<float>(N);
    float fp16_err = norm * 0.001f;
    return quant_err + fp16_err;
}

// ── Basic roundtrip tests ───────────────────────────────────────────

TEST(Roundtrip, SignPreservation) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    for (int i = 0; i < kTileSize; i++)
        input[i] = static_cast<float>(i + 1) / kTileSize;
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());
    for (int i = 0; i < kTileSize; i++) EXPECT_GE(output[i], 0.0f);

    for (int i = 0; i < kTileSize; i++)
        input[i] = -static_cast<float>(i + 1) / kTileSize;
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());
    for (int i = 0; i < kTileSize; i++) EXPECT_LE(output[i], 0.0f);
}

TEST(Roundtrip, AllZeros) {
    std::vector<float> input(kTileSize, 0.0f), output(kTileSize);
    TurboQuantTile tile{};

    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());
    for (int i = 0; i < kTileSize; i++) EXPECT_FLOAT_EQ(output[i], 0.0f);
}

TEST(Roundtrip, UniformRandom) {
    // 3.5-bit has lower resolution than 4.5-bit; allow 15% relative RMSE.
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float rel = row_relative_rmse(
            input.data() + row * TurboQuantTile::kDims,
            output.data() + row * TurboQuantTile::kDims,
            TurboQuantTile::kDims);
        EXPECT_LT(rel, 0.15f);
    }
}

TEST(Roundtrip, GaussianRandom) {
    // Gaussian heavy tails with only 4 magnitude levels on 3-bit channels
    // can cause high RMSE on rows with large outliers. Tolerance: 25%.
    // (Lloyd-Max codebooks in Phase 2 will improve this significantly.)
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 5.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float rel = row_relative_rmse(
            input.data() + row * TurboQuantTile::kDims,
            output.data() + row * TurboQuantTile::kDims,
            TurboQuantTile::kDims);
        EXPECT_LT(rel, 0.25f);
    }
}

TEST(Roundtrip, MultipleSeeds100) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), tile);
        dequantize_tile(tile, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            float rel = row_relative_rmse(
                input.data() + row * TurboQuantTile::kDims,
                output.data() + row * TurboQuantTile::kDims,
                TurboQuantTile::kDims);
            EXPECT_LT(rel, 0.15f);
        }
    }
}

TEST(Roundtrip, LargeMagnitude) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-60000.0f, 60000.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float rel = row_relative_rmse(
            input.data() + row * TurboQuantTile::kDims,
            output.data() + row * TurboQuantTile::kDims,
            TurboQuantTile::kDims);
        EXPECT_LT(rel, 0.15f);
    }
}

TEST(Roundtrip, MixedScaleRows) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(777);

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float scale = std::pow(10.0f, -4.0f + 8.0f * row / 15.0f);
        std::uniform_real_distribution<float> dist(-scale, scale);
        for (int d = 0; d < TurboQuantTile::kDims; d++)
            input[row * TurboQuantTile::kDims + d] = dist(rng);
    }

    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float rel = row_relative_rmse(
            input.data() + row * TurboQuantTile::kDims,
            output.data() + row * TurboQuantTile::kDims,
            TurboQuantTile::kDims);
        EXPECT_LT(rel, 0.15f);
    }
}

// ── Granular dequant verification ───────────────────────────────────

TEST(Dequant, PerValueErrorBound) {
    // Every dequantized value must be within ±(norm * 0.5/N + epsilon)
    // where N = mag_levels (8 for 4-bit signed, 4 for 3-bit signed).
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float norm = fp16_to_float(tile.norms[row]);
        for (int d = 0; d < TurboQuantTile::kDims; d++) {
            int idx = row * TurboQuantTile::kDims + d;
            float err = std::fabs(output[idx] - input[idx]);
            float bound = max_error_for_dim(d, norm);
            EXPECT_LE(err, bound);
        }
    }
}

TEST(Dequant, PerValueErrorBound_MultiSeed) {
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed + 1000);
        std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), tile);
        dequantize_tile(tile, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            float norm = fp16_to_float(tile.norms[row]);
            for (int d = 0; d < TurboQuantTile::kDims; d++) {
                int idx = row * TurboQuantTile::kDims + d;
                float err = std::fabs(output[idx] - input[idx]);
                float bound = max_error_for_dim(d, norm);
                EXPECT_LE(err, bound);
            }
        }
    }
}

TEST(Dequant, CentroidValidity) {
    // Every dequantized |value| must land on a valid centroid:
    // (mag_idx + 0.5) / mag_levels * norm
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    for (auto& v : input) v = dist(rng);
    quantize_tile(input.data(), tile);
    dequantize_tile(tile, output.data());

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float norm = fp16_to_float(tile.norms[row]);
        if (norm == 0.0f) continue;

        for (int d = 0; d < TurboQuantTile::kDims; d++) {
            float val = output[row * TurboQuantTile::kDims + d];
            float mag = std::fabs(val);
            int N = mag_levels_for_dim(d);

            // Recover magnitude index: mag = (mag_idx + 0.5) / N * norm
            float ratio = mag / norm * static_cast<float>(N);
            int mag_idx = static_cast<int>(std::round(ratio - 0.5f));

            EXPECT_GE(mag_idx, 0);
            EXPECT_LT(mag_idx, N);

            float expected_mag = (static_cast<float>(mag_idx) + 0.5f)
                                 / static_cast<float>(N) * norm;
            EXPECT_LT(std::fabs(mag - expected_mag), norm * 1e-5f);
        }
    }
}

TEST(Dequant, HiLoChannelAccuracy) {
    // 4-bit signed channels (8 mag levels) must have lower RMSE than
    // 3-bit signed channels (4 mag levels).
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    float hi_mse_total = 0.0f, hi_norm_total = 0.0f;
    float lo_mse_total = 0.0f, lo_norm_total = 0.0f;

    for (int seed = 0; seed < 50; seed++) {
        std::mt19937 rng(seed + 2000);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), tile);
        dequantize_tile(tile, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                int idx = row * TurboQuantTile::kDims + d;
                float e = output[idx] - input[idx];
                hi_mse_total += e * e;
                hi_norm_total += input[idx] * input[idx];
            }
            for (int d = TurboQuantTile::kHiDims; d < TurboQuantTile::kDims; d++) {
                int idx = row * TurboQuantTile::kDims + d;
                float e = output[idx] - input[idx];
                lo_mse_total += e * e;
                lo_norm_total += input[idx] * input[idx];
            }
        }
    }

    float hi_rel_rmse = std::sqrt(hi_mse_total / hi_norm_total);
    float lo_rel_rmse = std::sqrt(lo_mse_total / lo_norm_total);

    // 4-bit signed (8 mag levels) must have lower error than 3-bit signed (4 mag levels)
    EXPECT_LT(hi_rel_rmse, lo_rel_rmse);

    // Ratio should be roughly 2x (step_lo / step_hi = 2). Allow ≥ 1.5x.
    float ratio = lo_rel_rmse / hi_rel_rmse;
    EXPECT_GT(ratio, 1.5f);
}

TEST(Dequant, ErrorDistribution) {
    // Verify quantization errors behave like uniform noise:
    // 1. Mean normalized error ≈ 0 (unbiased)
    // 2. Variance ≈ step^2 / 12
    // Checked separately for 4-bit signed (8 mag levels) and 3-bit signed (4 mag levels).
    std::vector<float> input(kTileSize), output(kTileSize);
    TurboQuantTile tile{};

    std::vector<float> hi_errors, lo_errors;

    for (int seed = 0; seed < 100; seed++) {
        std::mt19937 rng(seed + 3000);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (auto& v : input) v = dist(rng);

        quantize_tile(input.data(), tile);
        dequantize_tile(tile, output.data());

        for (int row = 0; row < TurboQuantTile::kTokens; row++) {
            float norm = fp16_to_float(tile.norms[row]);
            if (norm < 1e-6f) continue;

            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                int idx = row * TurboQuantTile::kDims + d;
                float normed_err = (output[idx] - input[idx]) / norm;
                hi_errors.push_back(normed_err);
            }
            for (int d = TurboQuantTile::kHiDims; d < TurboQuantTile::kDims; d++) {
                int idx = row * TurboQuantTile::kDims + d;
                float normed_err = (output[idx] - input[idx]) / norm;
                lo_errors.push_back(normed_err);
            }
        }
    }

    // 4-bit signed: 8 magnitude levels → step = 1/8
    {
        double sum = 0, sum_sq = 0;
        for (float e : hi_errors) { sum += e; sum_sq += e * e; }
        double n = static_cast<double>(hi_errors.size());
        double mean = sum / n;
        double variance = sum_sq / n - mean * mean;

        EXPECT_LT(std::fabs(mean), 0.01);

        double expected_var = (1.0 / 8.0) * (1.0 / 8.0) / 12.0;  // 0.001302
        EXPECT_LT(variance, expected_var * 2.0);
        EXPECT_GT(variance, expected_var * 0.2);
    }

    // 3-bit signed: 4 magnitude levels → step = 1/4
    {
        double sum = 0, sum_sq = 0;
        for (float e : lo_errors) { sum += e; sum_sq += e * e; }
        double n = static_cast<double>(lo_errors.size());
        double mean = sum / n;
        double variance = sum_sq / n - mean * mean;

        EXPECT_LT(std::fabs(mean), 0.02);

        double expected_var = (1.0 / 4.0) * (1.0 / 4.0) / 12.0;  // 0.005208
        EXPECT_LT(variance, expected_var * 2.0);
        EXPECT_GT(variance, expected_var * 0.2);
    }
}
