#include "turboquant/test_framework.h"
#include "turboquant/tile.h"

using turboquant::TurboQuantTile;

// Compile-time checks (if this compiles, they pass)
static_assert(sizeof(TurboQuantTile) == 480);
static_assert(sizeof(TurboQuantTile) % 16 == 0);
static_assert(alignof(TurboQuantTile) == 16);
static_assert(offsetof(TurboQuantTile, norms)    == 0);
static_assert(offsetof(TurboQuantTile, quant_hi) == 32);
static_assert(offsetof(TurboQuantTile, quant_lo) == 288);
static_assert(offsetof(TurboQuantTile, quant_hi) % 16 == 0);
static_assert(offsetof(TurboQuantTile, quant_lo) % 16 == 0);

TEST(Tile, Size)           { EXPECT_EQ(sizeof(TurboQuantTile), 480u); }
TEST(Tile, Alignment)      { EXPECT_EQ(alignof(TurboQuantTile), 16u); }
TEST(Tile, SizeMod16)      { EXPECT_EQ(sizeof(TurboQuantTile) % 16, 0u); }
TEST(Tile, OffsetNorms)    { EXPECT_EQ(offsetof(TurboQuantTile, norms), 0u); }
TEST(Tile, OffsetQuantHi)  { EXPECT_EQ(offsetof(TurboQuantTile, quant_hi), 32u); }
TEST(Tile, OffsetQuantLo)  { EXPECT_EQ(offsetof(TurboQuantTile, quant_lo), 288u); }

TEST(Tile, Constants) {
    EXPECT_EQ(TurboQuantTile::kTokens, 16);
    EXPECT_EQ(TurboQuantTile::kDims, 64);
    EXPECT_EQ(TurboQuantTile::kHiDims, 32);
    EXPECT_EQ(TurboQuantTile::kLoDims, 32);
    EXPECT_EQ(TurboQuantTile::kHiBits, 4);
    EXPECT_EQ(TurboQuantTile::kLoBits, 3);
    EXPECT_EQ(TurboQuantTile::kHiLevels, 16);
    EXPECT_EQ(TurboQuantTile::kLoLevels, 8);
    EXPECT_EQ(TurboQuantTile::kHiBytesPerRow, 16);
    EXPECT_EQ(TurboQuantTile::kLoBytesPerRow, 12);
}

TEST(Tile, EffectiveBitsPerValue) {
    // (4-bit * 32 dims + 3-bit * 32 dims) / 64 dims = 3.5 bits/value
    float effective = static_cast<float>(
        TurboQuantTile::kHiBits * TurboQuantTile::kHiDims +
        TurboQuantTile::kLoBits * TurboQuantTile::kLoDims)
        / static_cast<float>(TurboQuantTile::kDims);
    EXPECT_FLOAT_EQ(effective, 3.5f);
}
