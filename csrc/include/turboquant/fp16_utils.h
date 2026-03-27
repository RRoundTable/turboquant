#pragma once

#include <cmath>
#include <cstdint>

namespace turboquant {

// IEEE 754 half-precision (binary16) layout:
//   bit 15:      sign
//   bits 14-10:  exponent (biased by 15)
//   bits 9-0:    mantissa (implicit leading 1 for normals)

inline uint16_t float_to_fp16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    // Handle special cases
    if (exponent == 128) {
        // Inf or NaN
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00);  // Inf
        }
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));  // NaN
    }

    if (exponent > 15) {
        // Overflow → Inf
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    if (exponent < -14) {
        // Subnormal or underflow
        if (exponent < -24) {
            return static_cast<uint16_t>(sign);  // too small → zero
        }
        // Subnormal: shift mantissa with implicit leading 1
        mantissa |= 0x800000;  // add implicit 1
        int shift = -1 - exponent;  // shift = 14 + (-exponent) - 13
        // Round-to-nearest-even
        uint32_t round_bit = 1U << (shift - 1 + 13);
        uint32_t sticky = (mantissa & (round_bit - 1)) ? 1 : 0;
        uint32_t shifted = mantissa >> (shift + 13);
        if ((mantissa >> (shift + 13 - 1)) & 1) {
            // Round bit is set
            if (sticky || (shifted & 1)) {
                shifted++;
            }
        }
        return static_cast<uint16_t>(sign | shifted);
    }

    // Normal range: round-to-nearest-even
    uint32_t round_bit = mantissa & (1 << 12);
    uint32_t sticky = (mantissa & ((1 << 12) - 1)) ? 1 : 0;
    uint16_t result = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent + 15) << 10) | (mantissa >> 13));

    if (round_bit) {
        if (sticky || (result & 1)) {
            result++;
        }
    }

    return result;
}

inline float fp16_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;  // ±0
        } else {
            // Subnormal → normalize
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            bits = sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23)
                   | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal
        bits = sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23)
               | (mantissa << 13);
    }

    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace turboquant
