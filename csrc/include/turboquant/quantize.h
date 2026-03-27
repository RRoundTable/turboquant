#pragma once

#include "turboquant/tile.h"

#include <cmath>

namespace turboquant {

// Lloyd-Max optimal centroids for N(0,1), precomputed to high precision.
// After Hadamard rotation of a unit vector, each coordinate ~N(0, 1/d).
// Scaled by 1/sqrt(d) at runtime.

constexpr float kCentroids3bit[8] = {
    -2.1519775207392190f,
    -1.3439709227384000f,
    -0.7560052489261838f,
    -0.2451209601065855f,
     0.2451209601065855f,
     0.7560052489261838f,
     1.3439709227384000f,
     2.1519775207392190f,
};

constexpr float kBoundaries3bit[7] = {
    -1.7479742217388095f,
    -1.0499880858322919f,
    -0.5005631045163847f,
     0.0000000000000000f,
     0.5005631045163847f,
     1.0499880858322919f,
     1.7479742217388095f,
};

constexpr float kCentroids4bit[16] = {
    -2.7326368237440100f,
    -2.0693470280837980f,
    -1.6180344937899930f,
    -1.2562490887894980f,
    -0.9423683558764893f,
    -0.6567591430551046f,
    -0.3880823046410021f,
    -0.1284153814744918f,
     0.1284153814744918f,
     0.3880823046410021f,
     0.6567591430551046f,
     0.9423683558764893f,
     1.2562490887894980f,
     1.6180344937899930f,
     2.0693470280837980f,
     2.7326368237440100f,
};

constexpr float kBoundaries4bit[15] = {
    -2.4009919259139040f,
    -1.8436907609368955f,
    -1.4371417912897455f,
    -1.0993087223329937f,
    -0.7995637494657970f,
    -0.5224207238480534f,
    -0.2582488430577470f,
     0.0000000000000000f,
     0.2582488430577470f,
     0.5224207238480534f,
     0.7995637494657970f,
     1.0993087223329937f,
     1.4371417912897455f,
     1.8436907609368955f,
     2.4009919259139040f,
};

// Quantize a 16×64 float array into a TurboQuantTile.
// Input should be Hadamard-rotated unit vectors rescaled by L2 norm.
// First 32 dims → 4-bit Lloyd-Max codebook, last 32 → 3-bit Lloyd-Max codebook.
// `dim` is the Hadamard padded dimension (for codebook scaling: 1/sqrt(dim)).
void quantize_tile(const float* input, int dim, TurboQuantTile& tile);

// Dequantize a TurboQuantTile back to a 16×64 float array.
void dequantize_tile(const TurboQuantTile& tile, int dim, float* output);

}  // namespace turboquant
