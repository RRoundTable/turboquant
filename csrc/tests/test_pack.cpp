#include "turboquant/test_framework.h"
#include "turboquant/pack.h"

#include <cstdint>

using namespace turboquant;

// ── 4-bit ───────────────────────────────────────────────────────────

TEST(Pack4Bit, RoundTrip) {
    uint8_t values[32];
    for (int i = 0; i < 32; i++) values[i] = static_cast<uint8_t>(i % 16);

    uint8_t packed[16];
    pack_4bit(values, 32, packed);

    uint8_t out[32];
    unpack_4bit(packed, 32, out);

    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], values[i]);
}

TEST(Pack4Bit, AllZeros) {
    uint8_t values[32] = {};
    uint8_t packed[16];
    pack_4bit(values, 32, packed);
    for (int i = 0; i < 16; i++) EXPECT_EQ(packed[i], 0u);

    uint8_t out[32];
    unpack_4bit(packed, 32, out);
    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], 0u);
}

TEST(Pack4Bit, AllMax) {
    uint8_t values[32];
    for (int i = 0; i < 32; i++) values[i] = 15;

    uint8_t packed[16];
    pack_4bit(values, 32, packed);
    for (int i = 0; i < 16; i++) EXPECT_EQ(packed[i], 0xFFu);

    uint8_t out[32];
    unpack_4bit(packed, 32, out);
    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], 15u);
}

TEST(Pack4Bit, NibbleOrder) {
    uint8_t values[2] = {0x0A, 0x05};
    uint8_t packed[1];
    pack_4bit(values, 2, packed);
    EXPECT_EQ(packed[0], 0xA5u);
}

TEST(Pack4Bit, SignedRange) {
    // 4-bit signed indices use full 0..15 range (MSB=sign, lower 3=magnitude)
    uint8_t values[32];
    for (int i = 0; i < 32; i++) values[i] = static_cast<uint8_t>(i % 16);

    uint8_t packed[16];
    pack_4bit(values, 32, packed);

    uint8_t out[32];
    unpack_4bit(packed, 32, out);
    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], values[i]);
}

// ── 3-bit ───────────────────────────────────────────────────────────

TEST(Pack3Bit, RoundTrip) {
    uint8_t values[32];
    for (int i = 0; i < 32; i++) values[i] = static_cast<uint8_t>(i % 8);

    uint8_t packed[12];
    pack_3bit(values, 32, packed);

    uint8_t out[32];
    unpack_3bit(packed, 32, out);

    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], values[i]);
}

TEST(Pack3Bit, AllZeros) {
    uint8_t values[8] = {};
    uint8_t packed[3];
    pack_3bit(values, 8, packed);
    for (int i = 0; i < 3; i++) EXPECT_EQ(packed[i], 0u);

    uint8_t out[8];
    unpack_3bit(packed, 8, out);
    for (int i = 0; i < 8; i++) EXPECT_EQ(out[i], 0u);
}

TEST(Pack3Bit, AllMax) {
    uint8_t values[8];
    for (int i = 0; i < 8; i++) values[i] = 7;

    uint8_t packed[3];
    pack_3bit(values, 8, packed);
    EXPECT_EQ(packed[0], 0xFFu);
    EXPECT_EQ(packed[1], 0xFFu);
    EXPECT_EQ(packed[2], 0xFFu);

    uint8_t out[8];
    unpack_3bit(packed, 8, out);
    for (int i = 0; i < 8; i++) EXPECT_EQ(out[i], 7u);
}

TEST(Pack3Bit, KnownPattern) {
    uint8_t values[8] = {1, 2, 3, 4, 5, 6, 7, 0};
    uint8_t packed[3];
    pack_3bit(values, 8, packed);

    uint8_t out[8];
    unpack_3bit(packed, 8, out);
    for (int i = 0; i < 8; i++) EXPECT_EQ(out[i], values[i]);
}

TEST(Pack3Bit, MultipleGroups) {
    uint8_t values[32];
    for (int i = 0; i < 32; i++) values[i] = static_cast<uint8_t>((i * 3 + 1) % 8);

    uint8_t packed[12];
    pack_3bit(values, 32, packed);

    uint8_t out[32];
    unpack_3bit(packed, 32, out);
    for (int i = 0; i < 32; i++) EXPECT_EQ(out[i], values[i]);
}

TEST(Pack3Bit, SignedRange) {
    // 3-bit signed indices use full 0..7 range (MSB=sign, lower 2=magnitude)
    uint8_t values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t packed[3];
    pack_3bit(values, 8, packed);

    uint8_t out[8];
    unpack_3bit(packed, 8, out);
    for (int i = 0; i < 8; i++) EXPECT_EQ(out[i], values[i]);
}
