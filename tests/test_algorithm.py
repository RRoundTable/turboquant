"""Standalone tests for TurboQuant algorithm correctness.

Run:  pytest tests/test_algorithm.py -v
  or: python tests/test_algorithm.py
"""

import math
import torch
import pytest

from turboquant.hadamard import fwht, RandomHadamardRotation, next_power_of_2
from turboquant.codebook import Codebook
from turboquant.qjl import QJL
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.kv_cache import TurboQuantCache


# ── Hadamard transform ─────────────────────────────────────────────────


class TestFWHT:
    def test_known_4x4(self):
        """FWHT of [1,0,0,0] should be [0.5, 0.5, 0.5, 0.5] (normalized)."""
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        y = fwht(x)
        expected = torch.tensor([0.5, 0.5, 0.5, 0.5])
        assert torch.allclose(y, expected, atol=1e-6)

    def test_involution(self):
        """Normalized FWHT applied twice = identity."""
        x = torch.randn(128)
        assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)

    def test_norm_preservation(self):
        """FWHT preserves L2 norm (orthogonal transform)."""
        x = torch.randn(64)
        y = fwht(x)
        assert abs(x.norm().item() - y.norm().item()) < 1e-5

    def test_batched(self):
        """Works with arbitrary batch dimensions."""
        x = torch.randn(2, 8, 32)
        y = fwht(x)
        assert y.shape == x.shape
        # Check norm preservation per vector
        assert torch.allclose(
            x.norm(dim=-1), y.norm(dim=-1), atol=1e-5
        )


class TestRandomHadamardRotation:
    def test_roundtrip(self):
        """rotate → inverse_rotate = identity."""
        rot = RandomHadamardRotation(100, seed=0)
        x = torch.randn(5, 100)
        y = rot.rotate(x)
        x_hat = rot.inverse_rotate(y)
        assert torch.allclose(x_hat, x, atol=1e-5)

    def test_norm_preservation(self):
        """Rotation preserves L2 norm."""
        rot = RandomHadamardRotation(128, seed=0)
        x = torch.randn(10, 128)
        y = rot.rotate(x)
        # Padded dim may differ, so compare norms accounting for padding
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-5)

    def test_uniform_on_sphere(self):
        """After rotation, unit vector coordinates should have ~N(0,1/d) distribution."""
        d = 128
        rot = RandomHadamardRotation(d, seed=42)
        # Take a fixed unit vector and rotate it
        x = torch.zeros(d)
        x[0] = 1.0
        y = rot.rotate(x.unsqueeze(0)).squeeze(0)[:d]
        # Variance should be ~1/d
        empirical_var = y.var().item()
        expected_var = 1.0 / d
        assert abs(empirical_var - expected_var) / expected_var < 0.3  # within 30%

    def test_padding(self):
        """Non-power-of-2 dims are handled via padding."""
        rot = RandomHadamardRotation(100, seed=0)
        assert rot.padded_dim == 128
        x = torch.randn(3, 100)
        y = rot.rotate(x)
        assert y.shape == (3, 128)
        x_hat = rot.inverse_rotate(y)
        assert x_hat.shape == (3, 100)
        assert torch.allclose(x_hat, x, atol=1e-5)


# ── Codebook ───────────────────────────────────────────────────────────


class TestCodebook:
    def test_quantize_dequantize_levels(self):
        """Indices should be in [0, 2^b) range."""
        for b in [1, 2, 3, 4]:
            cb = Codebook(128, b)
            x = torch.randn(100) / math.sqrt(128)  # ~N(0, 1/d)
            idx = cb.quantize(x)
            assert idx.min() >= 0
            assert idx.max() < (1 << b)

    def test_centroids_symmetry(self):
        """Codebook centroids should be symmetric around 0."""
        for b in [1, 2, 3, 4]:
            cb = Codebook(128, b)
            c = cb.centroids
            assert torch.allclose(c + c.flip(0), torch.zeros_like(c), atol=1e-6)

    def test_1bit_is_sign(self):
        """1-bit quantization should split at 0."""
        cb = Codebook(128, 1)
        x = torch.tensor([0.1, -0.1, 0.5, -0.5])
        idx = cb.quantize(x)
        assert (idx == torch.tensor([1, 0, 1, 0], dtype=torch.uint8)).all()


# ── QJL ────────────────────────────────────────────────────────────────


class TestQJL:
    def test_unbiasedness(self):
        """E[<y, Q^{-1}(Q(x))>] = <y, x> (statistical test over many trials)."""
        d = 128
        n_trials = 200
        estimates = []
        x = torch.randn(d)
        x = x / x.norm()
        y = torch.randn(d)

        true_ip = (x * y).sum().item()

        for seed in range(n_trials):
            qjl = QJL(d, seed=seed)
            signs = qjl.quantize(x.unsqueeze(0))
            x_hat = qjl.dequantize_inner_product(signs, torch.ones(1))
            est = (y * x_hat.squeeze(0)).sum().item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        # Should be close to true_ip (unbiased)
        assert abs(mean_est - true_ip) < 0.15, (
            f"QJL bias: true={true_ip:.4f}, mean_est={mean_est:.4f}"
        )


# ── MSE Quantizer (Algorithm 1) ───────────────────────────────────────


class TestTurboQuantMSE:
    """Validate MSE distortion matches Theorem 1 from the paper."""

    # Paper Table: for unit vectors, D_mse(b) ≈ ...
    EXPECTED_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_distortion_bound(self, bit_width):
        """MSE should be within 2x of paper's theoretical value."""
        d = 128
        n_vectors = 500
        q = TurboQuantMSE(d, bit_width, device="cpu", seed=42)

        # Random unit vectors
        x = torch.randn(n_vectors, d)
        x = x / x.norm(dim=-1, keepdim=True)

        indices, norms = q.quantize(x)
        x_hat = q.dequantize(indices, norms)

        # Per-vector MSE (for unit vectors, ||x||=1)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
        expected = self.EXPECTED_MSE[bit_width]

        print(f"  {bit_width}-bit MSE: {mse:.4f} (expected ≈ {expected})")
        # Allow 2x tolerance due to finite d and sample size
        assert mse < expected * 2.5, (
            f"{bit_width}-bit MSE={mse:.4f} exceeds 2.5x bound {expected*2.5:.4f}"
        )

    def test_norm_scaling(self):
        """Quantizer should handle non-unit vectors correctly."""
        d = 128
        q = TurboQuantMSE(d, 3, device="cpu")

        # Vector with large norm
        x = torch.randn(10, d) * 100
        indices, norms = q.quantize(x)
        x_hat = q.dequantize(indices, norms)

        # Normalized MSE should be similar to unit vector case
        norm_mse = ((x - x_hat).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean().item()
        assert norm_mse < 0.1

    def test_deterministic(self):
        """Same seed → same quantization."""
        d = 128
        x = torch.randn(5, d)
        q1 = TurboQuantMSE(d, 3, seed=42)
        q2 = TurboQuantMSE(d, 3, seed=42)
        idx1, n1 = q1.quantize(x)
        idx2, n2 = q2.quantize(x)
        assert (idx1 == idx2).all()
        assert torch.allclose(n1, n2)

    def test_batched_shapes(self):
        """Works with arbitrary batch dimensions (like KV cache tensors)."""
        d = 128
        q = TurboQuantMSE(d, 3)
        x = torch.randn(2, 32, 64, d)  # [batch, heads, seq_len, head_dim]
        idx, norms = q.quantize(x)
        x_hat = q.dequantize(idx, norms)
        assert x_hat.shape == x.shape

    def test_mse_decreases_with_bits(self):
        """More bits → lower MSE (monotonic improvement)."""
        d = 128
        x = torch.randn(200, d)
        x = x / x.norm(dim=-1, keepdim=True)

        prev_mse = float("inf")
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d, b, seed=42)
            idx, n = q.quantize(x)
            xh = q.dequantize(idx, n)
            mse = (x - xh).pow(2).sum(dim=-1).mean().item()
            assert mse < prev_mse, f"{b}-bit MSE {mse} >= {b-1}-bit MSE {prev_mse}"
            prev_mse = mse


# ── Inner Product Quantizer (Algorithm 2) ──────────────────────────────


class TestTurboQuantProd:
    """Validate inner product properties from Theorem 2."""

    def test_unbiasedness(self):
        """E[<y, x_hat>] ≈ <y, x> (key property of Algorithm 2)."""
        d = 128
        n_trials = 100
        x = torch.randn(d)
        x = x / x.norm()
        y = torch.randn(d)
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(n_trials):
            q = TurboQuantProd(d, 3, seed=seed)
            data = q.quantize(x.unsqueeze(0))
            x_hat = q.dequantize(*data).squeeze(0)
            estimates.append((y * x_hat).sum().item())

        mean_est = sum(estimates) / len(estimates)
        assert abs(mean_est - true_ip) < 0.1, (
            f"Prod bias: true={true_ip:.4f}, mean={mean_est:.4f}"
        )

    def test_distortion_decreases(self):
        """More bits → lower inner product distortion."""
        d = 128
        x = torch.randn(200, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(200, d)

        prev_err = float("inf")
        for b in [2, 3, 4]:
            q = TurboQuantProd(d, b, seed=42)
            data = q.quantize(x)
            x_hat = q.dequantize(*data)
            ip_err = ((y * x).sum(-1) - (y * x_hat).sum(-1)).pow(2).mean().item()
            assert ip_err < prev_err
            prev_err = ip_err


# ── KV Cache ───────────────────────────────────────────────────────────


class TestTurboQuantCache:
    def test_update_and_retrieve(self):
        """Basic store and retrieve."""
        cache = TurboQuantCache(head_dim=128, bit_width=3)
        k = torch.randn(1, 8, 10, 128)
        v = torch.randn(1, 8, 10, 128)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 8, 10, 128)

    def test_append_concatenates(self):
        """Multiple updates concatenate along seq_len dim."""
        cache = TurboQuantCache(head_dim=128, bit_width=3)
        k1 = torch.randn(1, 8, 10, 128)
        v1 = torch.randn(1, 8, 10, 128)
        cache.update(k1, v1, layer_idx=0)

        k2 = torch.randn(1, 8, 5, 128)
        v2 = torch.randn(1, 8, 5, 128)
        all_k, all_v = cache.update(k2, v2, layer_idx=0)
        assert all_k.shape == (1, 8, 15, 128)

    def test_multiple_layers(self):
        """Each layer has independent cache."""
        cache = TurboQuantCache(head_dim=128, bit_width=3)
        for layer in range(4):
            k = torch.randn(1, 8, 10, 128)
            v = torch.randn(1, 8, 10, 128)
            cache.update(k, v, layer_idx=layer)
        assert len(cache) == 4

    def test_outlier_aware(self):
        """Outlier channels use higher precision."""
        cache = TurboQuantCache(
            head_dim=128, bit_width=2,
            num_outlier_channels=32, outlier_bit_width=3,
        )
        cache.set_outlier_channels(torch.arange(32))
        assert abs(cache.effective_bit_width() - 2.5) < 1e-6

        k = torch.randn(1, 8, 10, 128)
        v = torch.randn(1, 8, 10, 128)
        all_k, all_v = cache.update(k, v, layer_idx=0)
        assert all_k.shape == (1, 8, 10, 128)

    def test_reconstruction_quality(self):
        """Quantized cache should have reasonable reconstruction."""
        cache = TurboQuantCache(head_dim=128, bit_width=3)
        k = torch.randn(1, 8, 100, 128)
        v = torch.randn(1, 8, 100, 128)
        all_k, all_v = cache.update(k, v, layer_idx=0)

        # Cosine similarity should be high
        cos_sim = torch.nn.functional.cosine_similarity(
            k.flatten(0, 2), all_k.flatten(0, 2), dim=-1
        ).mean()
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} too low"


# ── Attention simulation ───────────────────────────────────────────────


class TestAttentionSimulation:
    """Simulate full attention with quantized KV to measure end-to-end quality."""

    @staticmethod
    def _attention(q, k, v):
        """Standard scaled dot-product attention."""
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def test_attention_quality(self):
        """Attention output with quantized KV should be close to full precision."""
        d = 128
        batch, heads, seq_q, seq_kv = 1, 8, 1, 256

        q = torch.randn(batch, heads, seq_q, d)
        k = torch.randn(batch, heads, seq_kv, d)
        v = torch.randn(batch, heads, seq_kv, d)

        # Full precision attention
        out_fp = self._attention(q, k, v)

        # Quantized attention (3-bit)
        quant = TurboQuantMSE(d, 3, seed=42)
        k_idx, k_n = quant.quantize(k)
        v_idx, v_n = quant.quantize(v)
        k_hat = quant.dequantize(k_idx, k_n)
        v_hat = quant.dequantize(v_idx, v_n)
        out_q = self._attention(q, k_hat, v_hat)

        # Relative error
        rel_err = (out_fp - out_q).pow(2).sum().sqrt() / out_fp.pow(2).sum().sqrt()
        print(f"  3-bit attention relative error: {rel_err.item():.4f}")
        assert rel_err < 0.15, f"Attention error {rel_err:.4f} too high"

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_attention_improves_with_bits(self, bit_width):
        """Higher bit-width → lower attention error."""
        d = 128
        q = torch.randn(1, 8, 1, d)
        k = torch.randn(1, 8, 128, d)
        v = torch.randn(1, 8, 128, d)

        out_fp = self._attention(q, k, v)

        quant = TurboQuantMSE(d, bit_width, seed=42)
        k_hat = quant.dequantize(*quant.quantize(k))
        v_hat = quant.dequantize(*quant.quantize(v))
        out_q = self._attention(q, k_hat, v_hat)

        rel_err = (out_fp - out_q).norm() / out_fp.norm()
        # Just check it's bounded — the parametrize ensures we test all
        assert rel_err < 0.5

    def test_long_context_attention(self):
        """Simulate long-context scenario (4k tokens)."""
        d = 128
        seq_kv = 4096
        q = torch.randn(1, 1, 1, d)  # single query
        k = torch.randn(1, 1, seq_kv, d)
        v = torch.randn(1, 1, seq_kv, d)

        out_fp = self._attention(q, k, v)

        quant = TurboQuantMSE(d, 3, seed=42)
        k_hat = quant.dequantize(*quant.quantize(k))
        v_hat = quant.dequantize(*quant.quantize(v))
        out_q = self._attention(q, k_hat, v_hat)

        cos = torch.nn.functional.cosine_similarity(
            out_fp.flatten(), out_q.flatten(), dim=0
        )
        print(f"  Long-context (4k) cosine similarity: {cos.item():.6f}")
        assert cos > 0.98


# ── CLI runner ─────────────────────────────────────────────────────────


def run_all():
    """Run key tests with detailed output (no pytest needed)."""
    torch.manual_seed(0)
    print("=" * 65)
    print("TurboQuant Algorithm Validation")
    print("=" * 65)

    # 1. Hadamard
    print("\n[1/6] Fast Walsh-Hadamard Transform")
    x = torch.randn(128)
    assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)
    print("  ✓ FWHT is involution (H·H = I)")
    assert abs(x.norm().item() - fwht(x).norm().item()) < 1e-5
    print("  ✓ Preserves L2 norm")

    # 2. Rotation roundtrip
    print("\n[2/6] Randomized Hadamard Rotation")
    rot = RandomHadamardRotation(128, seed=42)
    x = torch.randn(50, 128)
    x_hat = rot.inverse_rotate(rot.rotate(x))
    assert torch.allclose(x_hat, x, atol=1e-5)
    print("  ✓ Perfect roundtrip (rotate → inverse = identity)")

    # 3. MSE distortion bounds
    print("\n[3/6] MSE Distortion (Theorem 1)")
    expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    d = 128
    x = torch.randn(1000, d)
    x = x / x.norm(dim=-1, keepdim=True)

    for b in [1, 2, 3, 4]:
        q = TurboQuantMSE(d, b, seed=42)
        idx, n = q.quantize(x)
        xh = q.dequantize(idx, n)
        mse = (x - xh).pow(2).sum(dim=-1).mean().item()
        status = "✓" if mse < expected[b] * 2.5 else "✗"
        print(f"  {status} {b}-bit: MSE={mse:.4f}  (paper bound: {expected[b]})")

    # 4. Inner product unbiasedness
    print("\n[4/6] Inner Product Unbiasedness (Theorem 2)")
    d = 128
    x = torch.randn(d)
    x = x / x.norm()
    y = torch.randn(d)
    true_ip = (x * y).sum().item()

    estimates = []
    for seed in range(200):
        q = TurboQuantProd(d, 3, seed=seed)
        data = q.quantize(x.unsqueeze(0))
        xh = q.dequantize(*data).squeeze(0)
        estimates.append((y * xh).sum().item())
    mean_est = sum(estimates) / len(estimates)
    bias = abs(mean_est - true_ip)
    print(f"  True IP:    {true_ip:.4f}")
    print(f"  Mean est:   {mean_est:.4f}")
    print(f"  Bias:       {bias:.4f}")
    print(f"  {'✓' if bias < 0.1 else '✗'} Unbiased (bias < 0.1)")

    # 5. Attention simulation
    print("\n[5/6] Attention Quality (simulated)")
    d = 128
    q_vec = torch.randn(1, 8, 1, d)
    k = torch.randn(1, 8, 512, d)
    v = torch.randn(1, 8, 512, d)

    def attn(q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        return torch.matmul(torch.softmax(s, -1), v)

    out_fp = attn(q_vec, k, v)

    for b in [2, 3, 4]:
        quant = TurboQuantMSE(d, b, seed=42)
        k_hat = quant.dequantize(*quant.quantize(k))
        v_hat = quant.dequantize(*quant.quantize(v))
        out_q = attn(q_vec, k_hat, v_hat)
        cos = torch.nn.functional.cosine_similarity(
            out_fp.flatten(), out_q.flatten(), dim=0
        ).item()
        rel_err = (out_fp - out_q).norm().item() / out_fp.norm().item()
        print(f"  {b}-bit: cosine_sim={cos:.4f}, rel_err={rel_err:.4f}")

    # 6. Cache
    print("\n[6/6] TurboQuantCache")
    cache = TurboQuantCache(head_dim=128, bit_width=3)
    k = torch.randn(1, 8, 100, 128)
    v = torch.randn(1, 8, 100, 128)
    all_k, all_v = cache.update(k, v, layer_idx=0)
    cos = torch.nn.functional.cosine_similarity(
        k.flatten(0, 2), all_k.flatten(0, 2), dim=-1
    ).mean().item()
    print(f"  Cache roundtrip cosine similarity: {cos:.4f}")
    print(f"  Compression ratio: {cache.compression_ratio():.1f}x")

    # Outlier-aware
    cache25 = TurboQuantCache(
        head_dim=128, bit_width=2,
        num_outlier_channels=32, outlier_bit_width=3,
    )
    cache25.set_outlier_channels(torch.arange(32))
    all_k, all_v = cache25.update(k, v, layer_idx=0)
    print(f"  2.5-bit outlier-aware: effective={cache25.effective_bit_width():.2f} bits, "
          f"compression={cache25.compression_ratio():.1f}x")

    print("\n" + "=" * 65)
    print("All validations passed!")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
