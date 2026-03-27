#include "turboquant/pack.h"

namespace turboquant {

// ── 4-bit packing ───────────────────────────────────────────────────

void pack_4bit(const uint8_t* values, size_t count, uint8_t* out) {
    for (size_t i = 0; i < count; i += 2) {
        out[i / 2] = static_cast<uint8_t>((values[i] << 4) | (values[i + 1] & 0x0F));
    }
}

void unpack_4bit(const uint8_t* packed, size_t count, uint8_t* out) {
    for (size_t i = 0; i < count; i += 2) {
        out[i]     = (packed[i / 2] >> 4) & 0x0F;
        out[i + 1] = packed[i / 2] & 0x0F;
    }
}

// ── 3-bit packing (group-of-8 in 3 bytes, GGML-compatible) ─────────

void pack_3bit(const uint8_t* values, size_t count, uint8_t* out) {
    for (size_t g = 0; g < count; g += 8) {
        const uint8_t* v = values + g;
        uint8_t* o = out + (g / 8) * 3;

        o[0] = static_cast<uint8_t>(
            (v[0] & 0x07)
            | ((v[1] & 0x07) << 3)
            | ((v[2] & 0x03) << 6));  // v2: low 2 bits

        o[1] = static_cast<uint8_t>(
            ((v[2] >> 2) & 0x01)      // v2: high 1 bit
            | ((v[3] & 0x07) << 1)
            | ((v[4] & 0x07) << 4)
            | ((v[5] & 0x01) << 7));  // v5: low 1 bit

        o[2] = static_cast<uint8_t>(
            ((v[5] >> 1) & 0x03)      // v5: high 2 bits
            | ((v[6] & 0x07) << 2)
            | ((v[7] & 0x07) << 5));
    }
}

void unpack_3bit(const uint8_t* packed, size_t count, uint8_t* out) {
    for (size_t g = 0; g < count; g += 8) {
        const uint8_t* p = packed + (g / 8) * 3;
        uint8_t* o = out + g;

        o[0] = p[0] & 0x07;
        o[1] = (p[0] >> 3) & 0x07;
        o[2] = ((p[0] >> 6) & 0x03) | ((p[1] & 0x01) << 2);
        o[3] = (p[1] >> 1) & 0x07;
        o[4] = (p[1] >> 4) & 0x07;
        o[5] = ((p[1] >> 7) & 0x01) | ((p[2] & 0x03) << 1);
        o[6] = (p[2] >> 2) & 0x07;
        o[7] = (p[2] >> 5) & 0x07;
    }
}

}  // namespace turboquant
