"""Tests for the fused write kernel (Phase 2).

Verifies the full pipeline: L2 normalize → Hadamard rotate → Lloyd-Max quantize → pack.
"""

import math
import torch
import pytest

from turboquant.write_kernel import TurboQuantWriter
from turboquant.tile import TILE_TOKENS, TILE_DIMS, TILE_BYTES

DEVICE = "cuda"


class TestTurboQuantWriter:
    def test_basic_roundtrip(self):
        """Quantize → dequantize preserves vectors reasonably."""
        writer = TurboQuantWriter(head_dim=64, device=DEVICE, seed=42)
        kv = torch.randn(16, 64, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])

        assert kv_hat.shape == kv.shape
        cos = torch.nn.functional.cosine_similarity(kv, kv_hat, dim=-1)
        assert cos.min() > 0.90

    def test_tile_shape(self):
        """Output tiles have correct shape."""
        writer = TurboQuantWriter(head_dim=64, device=DEVICE)
        kv = torch.randn(32, 64, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        # 32 tokens / 16 per tile = 2 token tiles, 64 dims / 64 = 1 tile per row
        assert tiles.shape == (2, 1, TILE_BYTES)
        assert meta["num_tokens"] == 32
        assert meta["tiles_per_row"] == 1

    def test_tile_shape_128(self):
        """head_dim=128 needs 2 tiles per row."""
        writer = TurboQuantWriter(head_dim=128, device=DEVICE)
        kv = torch.randn(16, 128, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        assert tiles.shape == (1, 2, TILE_BYTES)  # 1 token tile, 2 dim chunks
        assert meta["tiles_per_row"] == 2

    def test_padding(self):
        """Non-multiple-of-16 tokens are padded correctly."""
        writer = TurboQuantWriter(head_dim=64, device=DEVICE)
        kv = torch.randn(10, 64, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        assert tiles.shape == (1, 1, TILE_BYTES)  # 10 → padded to 16 → 1 token tile, 1 dim chunk
        kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])
        assert kv_hat.shape == (10, 64)

    def test_deterministic(self):
        """Same seed → same output."""
        kv = torch.randn(16, 64, device=DEVICE)
        w1 = TurboQuantWriter(head_dim=64, device=DEVICE, seed=42)
        w2 = TurboQuantWriter(head_dim=64, device=DEVICE, seed=42)

        t1, _ = w1.quantize_kv(kv)
        t2, _ = w2.quantize_kv(kv)
        assert (t1 == t2).all()

    def test_non_power_of_2_dim(self):
        """Head dims that aren't power-of-2 (e.g., 96) work via Hadamard padding."""
        writer = TurboQuantWriter(head_dim=96, device=DEVICE)
        kv = torch.randn(16, 96, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])

        assert kv_hat.shape == (16, 96)
        cos = torch.nn.functional.cosine_similarity(kv, kv_hat, dim=-1)
        assert cos.min() > 0.85

    def test_head_dim_128(self):
        """Standard Llama head_dim=128."""
        writer = TurboQuantWriter(head_dim=128, device=DEVICE, seed=42)
        kv = torch.randn(64, 128, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])

        assert kv_hat.shape == (64, 128)
        cos = torch.nn.functional.cosine_similarity(kv, kv_hat, dim=-1)
        assert cos.mean() > 0.95

    def test_quality_reasonable_across_dims(self):
        """All dims produce reasonable quality (norm MSE < 5%)."""
        for dim in [64, 128, 256]:
            writer = TurboQuantWriter(head_dim=dim, device=DEVICE, seed=42)
            kv = torch.randn(32, dim, device=DEVICE)

            tiles, meta = writer.quantize_kv(kv)
            kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])

            cos = torch.nn.functional.cosine_similarity(kv, kv_hat, dim=-1)
            assert cos.mean() > 0.95, f"dim={dim}: cos={cos.mean():.4f}"

    def test_zero_input(self):
        writer = TurboQuantWriter(head_dim=64, device=DEVICE)
        kv = torch.zeros(16, 64, device=DEVICE)

        tiles, meta = writer.quantize_kv(kv)
        kv_hat = writer.dequantize_kv(tiles, meta["num_tokens"])
        assert kv_hat.abs().max() < 1e-6

    def test_attention_quality(self):
        """Full pipeline: quantized KV cache → attention output quality."""
        d = 128
        writer = TurboQuantWriter(head_dim=d, device=DEVICE, seed=42)

        q = torch.randn(1, d, device=DEVICE)
        k = torch.randn(256, d, device=DEVICE)
        v = torch.randn(256, d, device=DEVICE)

        # Full precision attention
        scores_fp = torch.matmul(q, k.t()) / math.sqrt(d)
        weights_fp = torch.softmax(scores_fp, dim=-1)
        out_fp = torch.matmul(weights_fp, v)

        # Quantized attention
        k_tiles, k_meta = writer.quantize_kv(k)
        v_tiles, v_meta = writer.quantize_kv(v)
        k_hat = writer.dequantize_kv(k_tiles, k_meta["num_tokens"])
        v_hat = writer.dequantize_kv(v_tiles, v_meta["num_tokens"])

        scores_q = torch.matmul(q, k_hat.t()) / math.sqrt(d)
        weights_q = torch.softmax(scores_q, dim=-1)
        out_q = torch.matmul(weights_q, v_hat)

        cos = torch.nn.functional.cosine_similarity(out_fp, out_q, dim=-1)
        assert cos > 0.95
