#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant {

// TurboQuantTile: stores one compressed tile of 16 tokens x 64 dimensions.
// 3.5-bit average: 32 dims at 4-bit + 32 dims at 3-bit.
//
// After Hadamard rotation, each coordinate follows ~N(0, 1/d). We use
// precomputed Lloyd-Max codebook indices (unsigned). Centroids are symmetric
// around 0 and carry the sign implicitly.
//
// Memory layout (480 bytes, 16-byte aligned):
//   offset   0: norms     [16]  — FP16 L2 norm per row               (32 B)
//   offset  32: quant_hi [256]  — 4-bit unsigned codebook idx, 16 B/row  (256 B)
//   offset 288: quant_lo [192]  — 3-bit unsigned codebook idx, 12 B/row  (192 B)
//
// 4-bit: index 0..15 → Lloyd-Max centroid for N(0,1) scaled by 1/√d
//   Packing: 2 values per byte (high nibble = even idx, low nibble = odd idx)
// 3-bit: index 0..7 → Lloyd-Max centroid for N(0,1) scaled by 1/√d
//   Packing: group-of-8 in 3 bytes (GGML-compatible, little-endian)
//
// Dequantize: value = centroids[index] * norm
//
// Effective bits per value: (4 + 3) / 2 = 3.5

struct alignas(16) TurboQuantTile {
    static constexpr int kTokens        = 16;
    static constexpr int kDims          = 64;
    static constexpr int kHiDims        = 32;  // first 32 dims (4-bit)
    static constexpr int kLoDims        = 32;  // last 32 dims (3-bit)
    static constexpr int kHiBits        = 4;
    static constexpr int kLoBits        = 3;
    static constexpr int kHiLevels      = 16;  // 2^4 codebook levels
    static constexpr int kLoLevels      = 8;   // 2^3 codebook levels
    static constexpr int kHiBytesPerRow = 16;  // 32 dims * 4 bits / 8
    static constexpr int kLoBytesPerRow = 12;  // 32 dims * 3 bits / 8

    // FP16 L2 norm per row — stored as raw uint16_t FP16 bits
    uint16_t norms[kTokens];                          // 32 bytes

    // 4-bit unsigned Lloyd-Max codebook indices (first 32 dims after rotation)
    // Row r at quant_hi[r * 16 .. r * 16 + 15]
    uint8_t quant_hi[kTokens * kHiBytesPerRow];       // 256 bytes

    // 3-bit unsigned Lloyd-Max codebook indices (last 32 dims after rotation)
    // Row r at quant_lo[r * 12 .. r * 12 + 11]
    uint8_t quant_lo[kTokens * kLoBytesPerRow];       // 192 bytes
};

// Compile-time layout verification
static_assert(sizeof(TurboQuantTile) == 480,
              "TurboQuantTile must be exactly 480 bytes");
static_assert(sizeof(TurboQuantTile) % 16 == 0,
              "TurboQuantTile size must be a multiple of 16 bytes (128 bits)");
static_assert(alignof(TurboQuantTile) == 16,
              "TurboQuantTile must be 16-byte aligned");

// Field offset verification
static_assert(offsetof(TurboQuantTile, norms)    == 0);
static_assert(offsetof(TurboQuantTile, quant_hi) == 32);
static_assert(offsetof(TurboQuantTile, quant_lo) == 288);

// All fields start at 16-byte aligned offsets within the struct
static_assert(offsetof(TurboQuantTile, quant_hi) % 16 == 0);
static_assert(offsetof(TurboQuantTile, quant_lo) % 16 == 0);

}  // namespace turboquant
