#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant {

// TurboQuantTile: stores one compressed tile of 16 tokens x 64 dimensions.
// 3.5-bit average: 32 outlier dims at 4-bit signed + 32 normal dims at 3-bit signed.
// Sign is folded into the index MSB (no separate sign field). Scales as FP16.
//
// Memory layout (480 bytes, 16-byte aligned):
//   offset   0: norms     [16]  — FP16 scale per row               (32 B)
//   offset  32: quant_hi [256]  — 4-bit signed outlier, 16 B/row   (256 B)
//   offset 288: quant_lo [192]  — 3-bit signed normal, 12 B/row    (192 B)
//
// 4-bit signed: idx[3]=sign, idx[2:0]=magnitude → ±(mag + 0.5) / 8
//   Packing: 2 values per byte (high nibble = even idx, low nibble = odd idx)
// 3-bit signed: idx[2]=sign, idx[1:0]=magnitude → ±(mag + 0.5) / 4
//   Packing: group-of-8 in 3 bytes (GGML-compatible, little-endian)
//
// Effective bits per value: (4 + 3) / 2 = 3.5 (sign included)

struct alignas(16) TurboQuantTile {
    static constexpr int kTokens        = 16;
    static constexpr int kDims          = 64;
    static constexpr int kHiDims        = 32;  // outlier channels (4-bit signed)
    static constexpr int kLoDims        = 32;  // normal channels (3-bit signed)
    static constexpr int kHiBits        = 4;   // bits per outlier value (sign included)
    static constexpr int kLoBits        = 3;   // bits per normal value (sign included)
    static constexpr int kHiMagLevels   = 8;   // 2^(kHiBits-1) magnitude levels
    static constexpr int kLoMagLevels   = 4;   // 2^(kLoBits-1) magnitude levels
    static constexpr int kHiBytesPerRow = 16;  // 32 dims * 4 bits / 8
    static constexpr int kLoBytesPerRow = 12;  // 32 dims * 3 bits / 8

    // FP16 norm (scale) per row — stored as raw uint16_t FP16 bits
    uint16_t norms[kTokens];                          // 32 bytes

    // 4-bit signed quantized outlier channels (dims 0..31)
    // Row r at quant_hi[r * 16 .. r * 16 + 15]
    uint8_t quant_hi[kTokens * kHiBytesPerRow];       // 256 bytes

    // 3-bit signed quantized normal channels (dims 32..63)
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
