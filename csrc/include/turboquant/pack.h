#pragma once

#include <cstddef>
#include <cstdint>

namespace turboquant {

// ── 4-bit packing (nibble) ──────────────────────────────────────────
// Packs `count` values (each 0..15) into bytes: 2 values per byte.
// byte = (values[2i] << 4) | values[2i+1]
// `count` must be even. `out` must have at least count/2 bytes.
void pack_4bit(const uint8_t* values, size_t count, uint8_t* out);
void unpack_4bit(const uint8_t* packed, size_t count, uint8_t* out);

// ── 3-bit packing (group-of-8) ─────────────────────────────────────
// Packs `count` values (each 0..7) into bytes: 8 values per 3 bytes.
// Uses GGML-compatible little-endian layout:
//   byte[0] = v0 | (v1 << 3) | (v2 << 6)
//   byte[1] = (v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)
//   byte[2] = (v5 >> 1) | (v6 << 2) | (v7 << 5)
// `count` must be a multiple of 8. `out` must have count*3/8 bytes.
void pack_3bit(const uint8_t* values, size_t count, uint8_t* out);
void unpack_3bit(const uint8_t* packed, size_t count, uint8_t* out);

}  // namespace turboquant
