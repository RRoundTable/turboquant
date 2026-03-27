#include "turboquant/quantize.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/pack.h"
#include "turboquant/tile.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace turboquant {

// Find the codebook index for a value using boundary search.
static inline uint8_t bucketize(float value, const float* boundaries, int n_boundaries) {
    for (int i = 0; i < n_boundaries; i++) {
        if (value < boundaries[i]) {
            return static_cast<uint8_t>(i);
        }
    }
    return static_cast<uint8_t>(n_boundaries);
}

void quantize_tile(const float* input, int dim, TurboQuantTile& tile) {
    std::memset(&tile, 0, sizeof(tile));

    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // Pre-scale boundaries by 1/sqrt(dim)
    float boundaries_4bit[15];
    for (int i = 0; i < 15; i++) {
        boundaries_4bit[i] = kBoundaries4bit[i] * scale;
    }
    float boundaries_3bit[7];
    for (int i = 0; i < 7; i++) {
        boundaries_3bit[i] = kBoundaries3bit[i] * scale;
    }

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        const float* src = input + row * TurboQuantTile::kDims;

        // Compute L2 norm for the row
        float l2_sq = 0.0f;
        for (int d = 0; d < TurboQuantTile::kDims; d++) {
            l2_sq += src[d] * src[d];
        }
        float l2_norm = std::sqrt(l2_sq);

        // Store L2 norm as FP16
        tile.norms[row] = float_to_fp16(l2_norm);

        // Recover FP16-rounded norm for quantization consistency
        float norm = fp16_to_float(tile.norms[row]);
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

        // Quantize first 32 dims → 4-bit Lloyd-Max codebook
        {
            uint8_t indices[TurboQuantTile::kHiDims];
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                float normalized = src[d] * inv_norm;
                indices[d] = bucketize(normalized, boundaries_4bit, 15);
            }
            pack_4bit(indices, TurboQuantTile::kHiDims,
                      tile.quant_hi + row * TurboQuantTile::kHiBytesPerRow);
        }

        // Quantize last 32 dims → 3-bit Lloyd-Max codebook
        {
            uint8_t indices[TurboQuantTile::kLoDims];
            for (int d = 0; d < TurboQuantTile::kLoDims; d++) {
                float normalized = src[TurboQuantTile::kHiDims + d] * inv_norm;
                indices[d] = bucketize(normalized, boundaries_3bit, 7);
            }
            pack_3bit(indices, TurboQuantTile::kLoDims,
                      tile.quant_lo + row * TurboQuantTile::kLoBytesPerRow);
        }
    }
}

void dequantize_tile(const TurboQuantTile& tile, int dim, float* output) {
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    // Pre-scale centroids
    float centroids_4bit[16];
    for (int i = 0; i < 16; i++) {
        centroids_4bit[i] = kCentroids4bit[i] * scale;
    }
    float centroids_3bit[8];
    for (int i = 0; i < 8; i++) {
        centroids_3bit[i] = kCentroids3bit[i] * scale;
    }

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float* dst = output + row * TurboQuantTile::kDims;
        float norm = fp16_to_float(tile.norms[row]);

        // Dequantize first 32 dims (4-bit codebook)
        {
            uint8_t indices[TurboQuantTile::kHiDims];
            unpack_4bit(tile.quant_hi + row * TurboQuantTile::kHiBytesPerRow,
                        TurboQuantTile::kHiDims, indices);
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                dst[d] = centroids_4bit[indices[d]] * norm;
            }
        }

        // Dequantize last 32 dims (3-bit codebook)
        {
            uint8_t indices[TurboQuantTile::kLoDims];
            unpack_3bit(tile.quant_lo + row * TurboQuantTile::kLoBytesPerRow,
                        TurboQuantTile::kLoDims, indices);
            for (int d = 0; d < TurboQuantTile::kLoDims; d++) {
                dst[TurboQuantTile::kHiDims + d] = centroids_3bit[indices[d]] * norm;
            }
        }
    }
}

}  // namespace turboquant
