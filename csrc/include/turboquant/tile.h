#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant {

// TurboQuantTile: stores one compressed tile of 16 tokens x 64 dimensions.
// Uniform 4-bit: all 64 dims at 4-bit Lloyd-Max codebook (16 levels).
//
// After Hadamard rotation, each coordinate follows ~N(0, 1/d). We use
// precomputed Lloyd-Max codebook indices (unsigned). Centroids are symmetric
// around 0 and carry the sign implicitly.
//
// Memory layout (544 bytes, 16-byte aligned):
//   offset   0: norms    [16]  — FP16 L2 norm per row                (32 B)
//   offset  32: quant   [512]  — 4-bit unsigned codebook idx, 32 B/row (512 B)
//
// 4-bit: index 0..15 → Lloyd-Max centroid for N(0,1) scaled by 1/√d
//   Packing: 2 values per byte (high nibble = even idx, low nibble = odd idx)
//
// Dequantize: value = centroids[index] * norm
//
// Compression: 544 bytes vs 2048 bytes fp16 = 3.76x
// Effective bits per value: 4 + 2/64 (norm amortized) ≈ 4.03

struct alignas(16) TurboQuantTile {
    static constexpr int kTokens       = 16;
    static constexpr int kDims         = 64;
    static constexpr int kBits         = 4;
    static constexpr int kLevels       = 16;   // 2^4
    static constexpr int kBytesPerRow  = 32;   // 64 dims * 4 bits / 8

    // FP16 L2 norm per row — stored as raw uint16_t FP16 bits
    uint16_t norms[kTokens];                       // 32 bytes

    // 4-bit unsigned Lloyd-Max codebook indices (all 64 dims after rotation)
    // Row r at quant[r * 32 .. r * 32 + 31]
    uint8_t quant[kTokens * kBytesPerRow];         // 512 bytes
};

// Compile-time layout verification
static_assert(sizeof(TurboQuantTile) == 544,
              "TurboQuantTile must be exactly 544 bytes");
static_assert(sizeof(TurboQuantTile) % 16 == 0,
              "TurboQuantTile size must be a multiple of 16 bytes (128 bits)");
static_assert(alignof(TurboQuantTile) == 16,
              "TurboQuantTile must be 16-byte aligned");

// Field offset verification
static_assert(offsetof(TurboQuantTile, norms) == 0);
static_assert(offsetof(TurboQuantTile, quant) == 32);
static_assert(offsetof(TurboQuantTile, quant) % 16 == 0);

}  // namespace turboquant
