#include "turboquant/test_framework.h"
#include "turboquant/fp16_utils.h"

#include <cmath>
#include <cstdint>

using namespace turboquant;

TEST(FP16, Zero) {
    uint16_t h = float_to_fp16(0.0f);
    EXPECT_EQ(h, 0x0000u);
    EXPECT_FLOAT_EQ(fp16_to_float(h), 0.0f);
}

TEST(FP16, NegativeZero) {
    uint16_t h = float_to_fp16(-0.0f);
    EXPECT_EQ(h, 0x8000u);
    float f = fp16_to_float(h);
    EXPECT_FLOAT_EQ(f, 0.0f);
    EXPECT_TRUE(std::signbit(f));
}

TEST(FP16, One) {
    uint16_t h = float_to_fp16(1.0f);
    EXPECT_EQ(h, 0x3C00u);
    EXPECT_FLOAT_EQ(fp16_to_float(h), 1.0f);
}

TEST(FP16, NegativeOne) {
    uint16_t h = float_to_fp16(-1.0f);
    EXPECT_EQ(h, 0xBC00u);
    EXPECT_FLOAT_EQ(fp16_to_float(h), -1.0f);
}

TEST(FP16, Infinity) {
    uint16_t h = float_to_fp16(INFINITY);
    EXPECT_EQ(h, 0x7C00u);
    EXPECT_TRUE(std::isinf(fp16_to_float(h)));
}

TEST(FP16, NegInfinity) {
    uint16_t h = float_to_fp16(-INFINITY);
    EXPECT_EQ(h, 0xFC00u);
    float f = fp16_to_float(h);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_TRUE(std::signbit(f));
}

TEST(FP16, NaN) {
    uint16_t h = float_to_fp16(NAN);
    EXPECT_TRUE(std::isnan(fp16_to_float(h)));
}

TEST(FP16, RoundTripNormals) {
    float vals[] = {
        0.5f, 1.0f, 2.0f, 0.1f, 0.25f, 3.14f, 10.0f, 100.0f,
        -0.5f, -1.0f, -2.0f, -0.1f, -0.25f, -3.14f, -10.0f, -100.0f,
        0.001f, 0.01f, 1000.0f, 65504.0f,
    };
    for (float v : vals) {
        uint16_t h = float_to_fp16(v);
        float r = fp16_to_float(h);
        float rel = std::fabs(r - v) / std::fabs(v);
        EXPECT_LT(rel, 0.001f);
    }
}

TEST(FP16, SmallSubnormals) {
    float tiny = 5.96e-8f;
    uint16_t h = float_to_fp16(tiny);
    float r = fp16_to_float(h);
    EXPECT_GE(r, 0.0f);
    EXPECT_LT(r, 1e-4f);
}

TEST(FP16, Overflow) {
    uint16_t h = float_to_fp16(100000.0f);
    EXPECT_TRUE(std::isinf(fp16_to_float(h)));
}
