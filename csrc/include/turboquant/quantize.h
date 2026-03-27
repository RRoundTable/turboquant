#pragma once

#include "turboquant/tile.h"

namespace turboquant {

// Quantize a 16×64 float array into a TurboQuantTile.
// `input` is row-major: input[row * 64 + dim].
// Dims [0..31] → 4-bit (quant_hi), dims [32..63] → 3-bit (quant_lo).
// Phase 1: uniform quantization with max-abs scaling.
void quantize_tile(const float* input, TurboQuantTile& tile);

// Dequantize a TurboQuantTile back to a 16×64 float array.
// `output` is row-major: output[row * 64 + dim].
void dequantize_tile(const TurboQuantTile& tile, float* output);

}  // namespace turboquant
