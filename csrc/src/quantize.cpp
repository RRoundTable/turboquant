#include "turboquant/quantize.h"
#include "turboquant/fp16_utils.h"
#include "turboquant/pack.h"
#include "turboquant/tile.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace turboquant {

// Signed quantization: MSB = sign, remaining bits = magnitude index.
// 4-bit signed: idx[3]=sign, idx[2:0]=mag → ±(mag + 0.5) / 8
// 3-bit signed: idx[2]=sign, idx[1:0]=mag → ±(mag + 0.5) / 4
static inline uint8_t encode_signed(float value, float inv_norm, int mag_levels) {
    float mag = std::fabs(value) * inv_norm;  // [0, 1]
    int mag_idx = static_cast<int>(mag * static_cast<float>(mag_levels));
    mag_idx = std::min(mag_idx, mag_levels - 1);
    uint8_t sign_bit = (value < 0.0f) ? 1 : 0;
    return static_cast<uint8_t>((sign_bit << (mag_levels == 8 ? 3 : 2)) | mag_idx);
}

static inline float decode_signed(uint8_t idx, float norm, int mag_levels) {
    int mag_bits = (mag_levels == 8) ? 3 : 2;
    uint8_t sign_bit = (idx >> mag_bits) & 1;
    uint8_t mag_idx = idx & (static_cast<uint8_t>(mag_levels) - 1);
    float centroid = (static_cast<float>(mag_idx) + 0.5f)
                     / static_cast<float>(mag_levels);
    return sign_bit ? -centroid * norm : centroid * norm;
}

void quantize_tile(const float* input, TurboQuantTile& tile) {
    std::memset(&tile, 0, sizeof(tile));

    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        const float* src = input + row * TurboQuantTile::kDims;

        // Compute max absolute value for the row
        float max_abs = 0.0f;
        for (int d = 0; d < TurboQuantTile::kDims; d++) {
            max_abs = std::max(max_abs, std::fabs(src[d]));
        }

        // Store norm as FP16
        tile.norms[row] = float_to_fp16(max_abs);

        // Recover norm in float for quantization (use the FP16-rounded value
        // so that dequantize sees the same scale)
        float norm = fp16_to_float(tile.norms[row]);
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

        // Quantize outlier channels (dims 0..31) → 4-bit signed
        {
            uint8_t indices[TurboQuantTile::kHiDims];
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                indices[d] = encode_signed(src[d], inv_norm,
                                           TurboQuantTile::kHiMagLevels);
            }
            pack_4bit(indices, TurboQuantTile::kHiDims,
                      tile.quant_hi + row * TurboQuantTile::kHiBytesPerRow);
        }

        // Quantize normal channels (dims 32..63) → 3-bit signed
        {
            uint8_t indices[TurboQuantTile::kLoDims];
            for (int d = 0; d < TurboQuantTile::kLoDims; d++) {
                indices[d] = encode_signed(src[TurboQuantTile::kHiDims + d],
                                           inv_norm,
                                           TurboQuantTile::kLoMagLevels);
            }
            pack_3bit(indices, TurboQuantTile::kLoDims,
                      tile.quant_lo + row * TurboQuantTile::kLoBytesPerRow);
        }
    }
}

void dequantize_tile(const TurboQuantTile& tile, float* output) {
    for (int row = 0; row < TurboQuantTile::kTokens; row++) {
        float* dst = output + row * TurboQuantTile::kDims;
        float norm = fp16_to_float(tile.norms[row]);

        // Dequantize outlier channels (4-bit signed)
        {
            uint8_t indices[TurboQuantTile::kHiDims];
            unpack_4bit(tile.quant_hi + row * TurboQuantTile::kHiBytesPerRow,
                        TurboQuantTile::kHiDims, indices);
            for (int d = 0; d < TurboQuantTile::kHiDims; d++) {
                dst[d] = decode_signed(indices[d], norm,
                                       TurboQuantTile::kHiMagLevels);
            }
        }

        // Dequantize normal channels (3-bit signed)
        {
            uint8_t indices[TurboQuantTile::kLoDims];
            unpack_3bit(tile.quant_lo + row * TurboQuantTile::kLoBytesPerRow,
                        TurboQuantTile::kLoDims, indices);
            for (int d = 0; d < TurboQuantTile::kLoDims; d++) {
                dst[TurboQuantTile::kHiDims + d] =
                    decode_signed(indices[d], norm,
                                  TurboQuantTile::kLoMagLevels);
            }
        }
    }
}

}  // namespace turboquant
