"""Unit tests for HYP-056 A_35_prime Python reference.

These gate the CUDA kernel port: the kernel MUST match this reference
bit-for-bit on the packed tile and within fp16 roundoff on the decoded
K/V values.

Run (GPU):   uv run pytest tests/test_outlier_aware_a35.py -v
Run (CPU):   TQ_FORCE_CPU=1 uv run pytest tests/test_outlier_aware_a35.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

from turboquant.outlier_aware_a35 import (
    A35PrimeQuantizer,
    mask_to_indices,
    pack_3bit_ggml,
    pack_4bit_nibble,
    unpack_3bit_ggml,
    unpack_4bit_nibble,
)

# Allow CPU fallback for local dev; Forge runs on CUDA
if os.environ.get("TQ_FORCE_CPU") == "1" or not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")


# ── Packing roundtrips ────────────────────────────────────────────────


class Test3BitGGMLPacking:
    @pytest.mark.parametrize("d", [8, 16, 64, 128])
    def test_roundtrip_1d(self, d):
        idx = torch.randint(0, 8, (d,), dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit_ggml(idx)
        assert packed.shape == (d * 3 // 8,)
        unpacked = unpack_3bit_ggml(packed, d)
        assert (unpacked == idx).all()

    def test_roundtrip_batched(self):
        idx = torch.randint(0, 8, (4, 8, 64), dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit_ggml(idx)
        assert packed.shape == (4, 8, 24)
        unpacked = unpack_3bit_ggml(packed, 64)
        assert (unpacked == idx).all()

    def test_compression_ratio(self):
        idx = torch.zeros(64, dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit_ggml(idx)
        # 64 × 3 bits = 192 bits = 24 bytes — exactly 3× compression vs one uint8 per index
        assert packed.shape == (24,)

    def test_all_values_cover_range(self):
        # Each of 8 positions and each of 8 values must roundtrip
        for base in range(8):
            idx = torch.tensor([(i + base) % 8 for i in range(64)],
                               dtype=torch.uint8, device=DEVICE)
            packed = pack_3bit_ggml(idx)
            unpacked = unpack_3bit_ggml(packed, 64)
            assert (unpacked == idx).all(), f"failed for base={base}"

    def test_rejects_odd_D(self):
        idx = torch.zeros(9, dtype=torch.uint8, device=DEVICE)
        with pytest.raises(AssertionError):
            pack_3bit_ggml(idx)

    def test_rejects_out_of_range(self):
        # Values >= 8 will silently corrupt — but we check dtype only;
        # this test documents that the caller is responsible for range.
        # (Bucketize always returns valid indices, so the upstream invariant holds.)
        idx = torch.full((8,), 9, dtype=torch.uint8, device=DEVICE)
        packed = pack_3bit_ggml(idx)
        unpacked = unpack_3bit_ggml(packed, 8)
        # value 9 wraps to value 1 (= 9 & 0x7)
        assert (unpacked == 1).all()


class Test4BitNibblePacking:
    @pytest.mark.parametrize("d", [2, 8, 64, 128])
    def test_roundtrip(self, d):
        idx = torch.randint(0, 16, (3, 4, d), dtype=torch.uint8, device=DEVICE)
        packed = pack_4bit_nibble(idx)
        assert packed.shape[-1] == d // 2
        unpacked = unpack_4bit_nibble(packed, d)
        assert (unpacked == idx).all()


# ── Outlier mask → index tables ───────────────────────────────────────


class TestMaskToIndices:
    def test_simple_half_half(self):
        # H=2 heads, first 64 of D=128 are outliers for head 0,
        # even dims are outliers for head 1
        H, D = 2, 128
        mask = torch.zeros(H, D, dtype=torch.bool, device=DEVICE)
        mask[0, :64] = True
        mask[1, 0::2] = True
        out_idx, reg_idx = mask_to_indices(mask, tier_size=64)
        assert out_idx.shape == (H, 64)
        assert reg_idx.shape == (H, 64)
        # Reconstruct mask from indices to verify
        recon = torch.zeros_like(mask)
        for h in range(H):
            recon[h, out_idx[h].long()] = True
        assert (recon == mask).all()

    def test_rejects_wrong_count(self):
        mask = torch.zeros(2, 128, dtype=torch.bool, device=DEVICE)
        mask[0, :65] = True  # wrong count
        mask[1, :64] = True
        with pytest.raises(AssertionError):
            mask_to_indices(mask, tier_size=64)


# ── FWHT on 64-dim sub-vector ─────────────────────────────────────────


class TestFWHT64:
    def test_norm_preserved(self):
        from turboquant.hadamard import fwht

        x = torch.randn(10, 64, device=DEVICE)
        y = fwht(x)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

    def test_involution(self):
        from turboquant.hadamard import fwht

        x = torch.randn(64, device=DEVICE)
        assert torch.allclose(fwht(fwht(x)), x, atol=1e-5)


# ── End-to-end A_35_prime quantize/dequantize ─────────────────────────


def _random_mask(H=8, D=128, tier_size=64, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    mask = torch.zeros(H, D, dtype=torch.bool)
    for h in range(H):
        perm = torch.randperm(D, generator=g)
        mask[h, perm[:tier_size]] = True
    return mask.to(DEVICE)


class TestA35PrimeQuantizer:
    def test_tile_byte_layout(self):
        mask = _random_mask(H=4)
        q = A35PrimeQuantizer(mask, head_dim=128)
        assert q.out_bytes == 32
        assert q.reg_bytes == 24
        assert q.norm_bytes == 4
        assert q.tile_bytes_raw == 60
        assert q.tile_bytes_aligned == 64
        assert abs(q.bits_per_dim - 3.5) < 1e-9

    def test_shape_contract(self):
        mask = _random_mask(H=8)
        q = A35PrimeQuantizer(mask, head_dim=128)
        x = torch.randn(2, 5, 8, 128, device=DEVICE, dtype=torch.float16)
        tile = q.quantize(x)
        assert tile.shape == (2, 5, 8, 64)
        assert tile.dtype == torch.uint8
        x_hat = q.dequantize(tile)
        assert x_hat.shape == x.shape
        assert x_hat.dtype == torch.float32

    def test_reconstruction_gaussian(self):
        # After random rotation, each coord ~ N(0, 1/sqrt(d)). Per-tier
        # independent quant should achieve cos >= 0.90 on synthetic unit-norm vectors.
        mask = _random_mask(H=8, seed=1)
        q = A35PrimeQuantizer(mask, head_dim=128)
        torch.manual_seed(0)
        x = torch.randn(16, 8, 128, device=DEVICE, dtype=torch.float32)
        tile = q.quantize(x)
        x_hat = q.dequantize(tile)
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, 128), x_hat.reshape(-1, 128), dim=-1
        )
        # Paper bound for MSE at 3.5 bits/dim is ~0.00266. Cos should be well above 0.9.
        mean_cos = cos.mean().item()
        assert mean_cos >= 0.90, f"cos={mean_cos:.4f} below 0.90"

    def test_deterministic(self):
        mask = _random_mask(H=8, seed=2)
        q1 = A35PrimeQuantizer(mask, head_dim=128, seed=42)
        q2 = A35PrimeQuantizer(mask, head_dim=128, seed=42)
        x = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.float16)
        t1 = q1.quantize(x)
        t2 = q2.quantize(x)
        assert (t1 == t2).all()

    def test_norm_bytes_roundtrip(self):
        # A tile's norms must survive the uint8→fp16 view roundtrip.
        mask = _random_mask(H=2)
        q = A35PrimeQuantizer(mask, head_dim=128)
        x = torch.randn(3, 2, 128, device=DEVICE, dtype=torch.float32)
        # Compute norms the same way as quantize
        prefix = x.shape[:-2]
        out_sub = torch.gather(
            x, -1, q.outlier_idx.long().expand(*prefix, -1, -1)
        )
        reg_sub = torch.gather(
            x, -1, q.regular_idx.long().expand(*prefix, -1, -1)
        )
        out_norms_expected = out_sub.norm(dim=-1)
        reg_norms_expected = reg_sub.norm(dim=-1)

        tile = q.quantize(x)
        # Extract norms from tile bytes
        reg_end = q.out_bytes + q.reg_bytes
        out_norm_bytes = tile[..., reg_end:reg_end + 2].contiguous()
        reg_norm_bytes = tile[..., reg_end + 2:reg_end + 4].contiguous()
        out_norms = out_norm_bytes.view(torch.float16).to(torch.float32).squeeze(-1)
        reg_norms = reg_norm_bytes.view(torch.float16).to(torch.float32).squeeze(-1)
        # fp16 roundtrip tolerance on a tensor of magnitude ~O(1)
        assert torch.allclose(out_norms, out_norms_expected, atol=1e-2, rtol=1e-2)
        assert torch.allclose(reg_norms, reg_norms_expected, atol=1e-2, rtol=1e-2)

    def test_packed_alignment_padding_is_zero(self):
        mask = _random_mask(H=2)
        q = A35PrimeQuantizer(mask, head_dim=128)
        x = torch.randn(2, 128, device=DEVICE, dtype=torch.float16)
        x = x.unsqueeze(0).expand(1, 2, 128)
        tile = q.quantize(x)
        # bytes 60..63 are padding
        padding = tile[..., 60:64]
        assert (padding == 0).all()


# ── CUDA kernel vs Python reference parity ───────────────────────────
# Bit-exact on the packed tile is the gate for the CUDA kernel. fp16 norm
# bytes may differ by one ULP (independent computation order); allow that.


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA kernel parity test requires GPU",
)
class TestCudaKernelParity:
    def _build(self, H=8, seed=3):
        from turboquant.a35_kernel import quantize_write_a35 as kern_quant

        mask = _random_mask(H=H, seed=seed)
        q = A35PrimeQuantizer(mask, head_dim=128, device=DEVICE)
        return q, kern_quant

    def test_tile_bitexact_outlier_and_regular(self):
        q, kern_quant = self._build(H=4, seed=3)
        torch.manual_seed(0)
        x = torch.randn(8, 4, 128, device=DEVICE, dtype=torch.float16)
        tile_py = q.quantize(x)
        tile_cu = kern_quant(x, q)

        # Compare outlier region [0..32) and regular region [32..56) byte-for-byte.
        # Norm bytes [56..60) may differ by 1 ULP in fp16 (independent reduction).
        assert (tile_py[..., :32] == tile_cu[..., :32]).all(), (
            "4-bit outlier nibbles mismatch kernel vs reference"
        )
        assert (tile_py[..., 32:56] == tile_cu[..., 32:56]).all(), (
            "3-bit regular GGML bytes mismatch kernel vs reference"
        )
        # Padding is zero
        assert (tile_cu[..., 60:64] == 0).all()

    def test_tile_norms_within_fp16_ulp(self):
        q, kern_quant = self._build(H=4, seed=4)
        torch.manual_seed(1)
        x = torch.randn(4, 4, 128, device=DEVICE, dtype=torch.float16)
        tile_py = q.quantize(x)
        tile_cu = kern_quant(x, q)
        norms_py = tile_py[..., 56:60].view(torch.float16).to(torch.float32)
        norms_cu = tile_cu[..., 56:60].view(torch.float16).to(torch.float32)
        diff = (norms_py - norms_cu).abs()
        # Tolerance: 2× fp16 relative epsilon (~2e-3) on values of magnitude ~10
        assert diff.max().item() < 5e-2, f"norm bytes diverge: max |Δ|={diff.max().item()}"

    def test_decoded_matches_reference(self):
        q, kern_quant = self._build(H=4, seed=5)
        torch.manual_seed(2)
        x = torch.randn(8, 4, 128, device=DEVICE, dtype=torch.float16)
        tile_py = q.quantize(x)
        tile_cu = kern_quant(x, q)
        # Decode both through the Python reference; compare reconstruction cos.
        x_py = q.dequantize(tile_py)
        x_cu = q.dequantize(tile_cu)
        cos = torch.nn.functional.cosine_similarity(
            x_py.reshape(-1, 128), x_cu.reshape(-1, 128), dim=-1
        )
        assert cos.mean() >= 0.9998, f"decode cos mean={cos.mean():.4f} < 0.9998"
