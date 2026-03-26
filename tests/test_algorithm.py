"""Tests for TurboQuant algorithm correctness.

Run: pytest tests/test_algorithm.py -v
"""

import math

import torch
import pytest

from turboquant.hadamard import fwht, RandomHadamardRotation, next_power_of_2
from turboquant.codebook import Codebook
from turboquant.qjl import QJL
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd, _pack_bits, _unpack_bits

DEVICE = "cuda"


# ── Bit packing ───────────────────────────────────────────────────────


class TestBitPacking:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_roundtrip(self, bit_width):
        d = 128
        max_val = (1 << bit_width) - 1
        indices = torch.randint(0, max_val + 1, (10, d), dtype=torch.uint8, device=DEVICE)
        packed = _pack_bits(indices, bit_width)
        unpacked = _unpack_bits(packed, bit_width, d)
        assert (indices == unpacked).all()

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_compression_ratio(self, bit_width):
        d = 128
        indices = torch.zeros(1, d, dtype=torch.uint8, device=DEVICE)
        packed = _pack_bits(indices, bit_width)
        vals_per_byte = 8 // bit_width
        expected_packed_dim = (d + vals_per_byte - 1) // vals_per_byte
        assert packed.shape[-1] == expected_packed_dim


# ── Hadamard transform ─────────────────────────────────────────────────


class TestFWHT:
    def test_known_4x4(self):
        x = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
        y = fwht(x)
        expected = torch.tensor([0.5, 0.5, 0.5, 0.5], device=DEVICE)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_involution(self):
        x = torch.randn(128, device=DEVICE)
        assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)

    def test_norm_preservation(self):
        x = torch.randn(64, device=DEVICE)
        y = fwht(x)
        assert abs(x.norm().item() - y.norm().item()) < 1e-5

    def test_batched(self):
        x = torch.randn(2, 8, 32, device=DEVICE)
        y = fwht(x)
        assert y.shape == x.shape
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


class TestRandomHadamardRotation:
    def test_roundtrip(self):
        rot = RandomHadamardRotation(100, device=DEVICE, seed=0)
        x = torch.randn(5, 100, device=DEVICE)
        y = rot.rotate(x)
        x_hat = rot.inverse_rotate(y)
        assert torch.allclose(x_hat, x, atol=1e-5)

    def test_norm_preservation(self):
        rot = RandomHadamardRotation(128, device=DEVICE, seed=0)
        x = torch.randn(10, 128, device=DEVICE)
        y = rot.rotate(x)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    def test_padding(self):
        rot = RandomHadamardRotation(100, device=DEVICE, seed=0)
        assert rot.padded_dim == 128
        x = torch.randn(3, 100, device=DEVICE)
        y = rot.rotate(x)
        assert y.shape == (3, 128)
        x_hat = rot.inverse_rotate(y)
        assert x_hat.shape == (3, 100)
        assert torch.allclose(x_hat, x, atol=1e-5)


# ── Codebook ───────────────────────────────────────────────────────────


class TestCodebook:
    def test_quantize_dequantize_levels(self):
        for b in [1, 2, 3, 4]:
            cb = Codebook(128, b, device=DEVICE)
            x = torch.randn(100, device=DEVICE) / math.sqrt(128)
            idx = cb.quantize(x)
            assert idx.min() >= 0
            assert idx.max() < (1 << b)

    def test_centroids_symmetry(self):
        for b in [1, 2, 3, 4]:
            cb = Codebook(128, b, device=DEVICE)
            c = cb.centroids
            assert torch.allclose(c + c.flip(0), torch.zeros_like(c), atol=1e-6)

    def test_on_cuda(self):
        cb = Codebook(128, 3, device=DEVICE)
        assert cb.centroids.device.type == "cuda"
        assert cb.boundaries.device.type == "cuda"


# ── QJL ────────────────────────────────────────────────────────────────


class TestQJL:
    def test_unbiasedness(self):
        d = 128
        n_trials = 200
        x = torch.randn(d, device=DEVICE)
        x = x / x.norm()
        y = torch.randn(d, device=DEVICE)
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(n_trials):
            qjl = QJL(d, device=DEVICE, seed=seed)
            signs = qjl.quantize(x.unsqueeze(0))
            x_hat = qjl.dequantize_inner_product(signs, torch.ones(1, device=DEVICE))
            est = (y * x_hat.squeeze(0)).sum().item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        assert abs(mean_est - true_ip) < 0.15

    def test_on_cuda(self):
        qjl = QJL(128, device=DEVICE)
        assert qjl.S.device.type == "cuda"


# ── MSE Quantizer (Algorithm 1) ───────────────────────────────────────


class TestTurboQuantMSE:
    EXPECTED_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_distortion_bound(self, bit_width):
        d = 128
        q = TurboQuantMSE(d, bit_width, device=DEVICE, seed=42)
        x = torch.randn(500, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        packed, norms = q.quantize(x)
        x_hat = q.dequantize(packed, norms)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        expected = self.EXPECTED_MSE[bit_width]
        assert mse < expected * 2.5

    def test_deterministic(self):
        d = 128
        x = torch.randn(5, d, device=DEVICE)
        q1 = TurboQuantMSE(d, 3, device=DEVICE, seed=42)
        q2 = TurboQuantMSE(d, 3, device=DEVICE, seed=42)
        p1, n1 = q1.quantize(x)
        p2, n2 = q2.quantize(x)
        assert (p1 == p2).all()
        assert torch.allclose(n1, n2)

    def test_batched_shapes(self):
        d = 128
        q = TurboQuantMSE(d, 3, device=DEVICE)
        x = torch.randn(2, 32, 64, d, device=DEVICE)
        packed, norms = q.quantize(x)
        x_hat = q.dequantize(packed, norms)
        assert x_hat.shape == x.shape

    def test_mse_decreases_with_bits(self):
        d = 128
        x = torch.randn(200, d, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        prev_mse = float("inf")
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d, b, device=DEVICE, seed=42)
            p, n = q.quantize(x)
            xh = q.dequantize(p, n)
            mse = (x - xh).pow(2).sum(dim=-1).mean().item()
            assert mse < prev_mse
            prev_mse = mse

    def test_3bit_compression_ratio(self):
        d = 128
        q = TurboQuantMSE(d, 3, device=DEVICE)
        x = torch.randn(1, d, device=DEVICE)
        packed, norms = q.quantize(x)
        # 3-bit: 2 values per byte → 128/2 = 64 bytes for indices + 4 bytes for norm
        # vs fp16: 128 * 2 = 256 bytes
        quant_bytes = packed.nelement() + norms.nelement() * 4
        fp16_bytes = x.nelement() * 2
        ratio = fp16_bytes / quant_bytes
        assert ratio > 3.0  # should be ~3.7x with bit packing


# ── Inner Product Quantizer (Algorithm 2) ──────────────────────────────


class TestTurboQuantProd:
    def test_unbiasedness(self):
        d = 128
        n_trials = 100
        x = torch.randn(d, device=DEVICE)
        x = x / x.norm()
        y = torch.randn(d, device=DEVICE)
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(n_trials):
            q = TurboQuantProd(d, 3, device=DEVICE, seed=seed)
            data = q.quantize(x.unsqueeze(0))
            x_hat = q.dequantize(*data).squeeze(0)
            estimates.append((y * x_hat).sum().item())

        mean_est = sum(estimates) / len(estimates)
        assert abs(mean_est - true_ip) < 0.1

    def test_min_bit_width(self):
        with pytest.raises(AssertionError):
            TurboQuantProd(128, 1, device=DEVICE)


# ── Attention simulation ───────────────────────────────────────────────


class TestAttentionSimulation:
    @staticmethod
    def _attention(q, k, v):
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def test_attention_quality(self):
        d = 128
        q = torch.randn(1, 8, 1, d, device=DEVICE)
        k = torch.randn(1, 8, 256, d, device=DEVICE)
        v = torch.randn(1, 8, 256, d, device=DEVICE)

        out_fp = self._attention(q, k, v)

        quant = TurboQuantMSE(d, 3, device=DEVICE, seed=42)
        k_p, k_n = quant.quantize(k)
        v_p, v_n = quant.quantize(v)
        k_hat = quant.dequantize(k_p, k_n)
        v_hat = quant.dequantize(v_p, v_n)
        out_q = self._attention(q, k_hat, v_hat)

        rel_err = (out_fp - out_q).pow(2).sum().sqrt() / out_fp.pow(2).sum().sqrt()
        assert rel_err < 0.30
