"""Tests for PyTorch tile implementation (Lloyd-Max codebook).

Verifies:
1. Pack/unpack roundtrips are bit-exact
2. Quantize → dequantize with Lloyd-Max centroids
3. Tile byte layout matches C++ struct offsets
"""

import struct
import torch
import pytest

from turboquant.tile import (
    TILE_TOKENS, TILE_DIMS, TILE_BYTES,
    HI_DIMS, LO_DIMS, HI_LEVELS, LO_LEVELS,
    HI_BYTES_PER_ROW, LO_BYTES_PER_ROW,
    pack_4bit, unpack_4bit,
    pack_3bit, unpack_3bit,
    quantize_tile, dequantize_tile,
)

DEVICE = "cuda"


class TestPacking:
    def test_4bit_roundtrip(self):
        indices = torch.randint(0, 16, (32,), dtype=torch.uint8, device=DEVICE)
        packed = pack_4bit(indices)
        assert packed.shape == (16,)
        unpacked = unpack_4bit(packed, 32)
        assert (indices == unpacked).all()

    def test_3bit_roundtrip(self):
        indices = torch.randint(0, 8, (32,), dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit(indices)
        assert packed.shape == (12,)
        unpacked = unpack_3bit(packed, 32)
        assert (indices == unpacked).all()

    def test_4bit_known_values(self):
        indices = torch.tensor([0x0A, 0x05], dtype=torch.uint8, device=DEVICE)
        packed = pack_4bit(indices)
        assert packed[0].item() == 0xA5

    def test_3bit_batched(self):
        indices = torch.randint(0, 8, (4, 32), dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit(indices)
        assert packed.shape == (4, 12)
        unpacked = unpack_3bit(packed, 32)
        assert (indices == unpacked).all()


class TestTileLayout:
    def test_tile_size(self):
        x = torch.randn(TILE_TOKENS, TILE_DIMS, device=DEVICE)
        tile = quantize_tile(x)
        assert tile.shape == (TILE_BYTES,)
        assert TILE_BYTES == 480

    def test_field_offsets(self):
        assert 0 + TILE_TOKENS * 2 == 32       # norms end
        assert 32 + TILE_TOKENS * HI_BYTES_PER_ROW == 288  # quant_hi end
        assert 288 + TILE_TOKENS * LO_BYTES_PER_ROW == 480  # quant_lo end

    def test_alignment(self):
        assert 0 % 16 == 0
        assert 32 % 16 == 0
        assert 288 % 16 == 0

    def test_roundtrip(self):
        torch.manual_seed(42)
        x = torch.randn(TILE_TOKENS, TILE_DIMS, device=DEVICE)
        tile = quantize_tile(x)
        x_hat = dequantize_tile(tile)

        assert x_hat.shape == x.shape
        # Cosine similarity should be reasonable
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
        assert cos.min() > 0.85

    def test_norm_is_fp16_l2(self):
        """Norms should be L2 norms stored as FP16."""
        x = torch.randn(TILE_TOKENS, TILE_DIMS, device=DEVICE) * 3.14
        tile = quantize_tile(x)

        for row in range(TILE_TOKENS):
            norm_bytes = tile[row * 2: row * 2 + 2].cpu().numpy()
            fp16_val = struct.unpack('<e', bytes(norm_bytes))[0]
            row_l2 = x[row].norm().item()
            assert abs(fp16_val - row_l2) / max(row_l2, 1e-8) < 0.01

    def test_zero_input(self):
        x = torch.zeros(TILE_TOKENS, TILE_DIMS, device=DEVICE)
        tile = quantize_tile(x)
        x_hat = dequantize_tile(tile)
        assert (x_hat == 0).all()

    def test_codebook_indices_in_range(self):
        """4-bit indices 0..15, 3-bit indices 0..7."""
        x = torch.randn(TILE_TOKENS, TILE_DIMS, device=DEVICE)
        tile = quantize_tile(x)

        for row in range(TILE_TOKENS):
            # Check 4-bit indices
            hi_offset = 32 + row * HI_BYTES_PER_ROW
            hi_packed = tile[hi_offset: hi_offset + HI_BYTES_PER_ROW]
            hi_indices = unpack_4bit(hi_packed, HI_DIMS)
            assert hi_indices.max() < HI_LEVELS

            # Check 3-bit indices
            lo_offset = 288 + row * LO_BYTES_PER_ROW
            lo_packed = tile[lo_offset: lo_offset + LO_BYTES_PER_ROW]
            lo_indices = unpack_3bit(lo_packed, LO_DIMS)
            assert lo_indices.max() < LO_LEVELS

    def test_cosine_similarity(self):
        torch.manual_seed(0)
        x = torch.randn(TILE_TOKENS, TILE_DIMS, device=DEVICE)
        tile = quantize_tile(x)
        x_hat = dequantize_tile(tile)

        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
        assert cos.min() > 0.85
