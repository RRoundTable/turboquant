"""HYP-056 A_35_prime Python reference — two-tier MSE-only quantization.

Layout (per head, head_dim=128):
  outlier tier: 64 dims × 4-bit MSE (Lloyd-Max 16 levels), nibble-packed → 32 B
  regular tier: 64 dims × 3-bit MSE (Lloyd-Max  8 levels), GGML-packed  → 24 B
  norms:        2 × fp16 (one per tier)                                 →  4 B
  total raw: 60 B  (kernel pads to 64 B)

The outlier mask is per-head, per-layer, offline-calibrated. This module is
mask-layer-agnostic: construct one A35PrimeQuantizer per layer with its mask.

Reference contract for the CUDA kernel:
  - quantize / dequantize are bit-exact round-trip vs the kernel's decode path
  - FWHT is applied per-tier (64 dims), not per-head, which matches the
    kernel's per-tier dequant-then-inverse-rotate layout.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .codebook import Codebook
from .hadamard import RandomHadamardRotation


# ── GGML-style 3-bit packing (8 indices → 3 bytes) ─────────────────────

def pack_3bit_ggml(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices (uint8 in [0,7]) into bytes, 8 vals per 3 bytes.

    Args:
      indices: [..., D] uint8 with D % 8 == 0
    Returns:
      packed:  [..., D * 3 // 8] uint8
    """
    assert indices.dtype == torch.uint8
    d = indices.shape[-1]
    assert d % 8 == 0, f"D must be multiple of 8 for 3-bit GGML packing, got {d}"
    g = d // 8

    prefix = indices.shape[:-1]
    groups = indices.view(*prefix, g, 8).to(torch.int64)

    packed24 = (
        (groups[..., 0])
        | (groups[..., 1] << 3)
        | (groups[..., 2] << 6)
        | (groups[..., 3] << 9)
        | (groups[..., 4] << 12)
        | (groups[..., 5] << 15)
        | (groups[..., 6] << 18)
        | (groups[..., 7] << 21)
    )
    b0 = (packed24 & 0xFF).to(torch.uint8)
    b1 = ((packed24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed24 >> 16) & 0xFF).to(torch.uint8)
    packed = torch.stack([b0, b1, b2], dim=-1).view(*prefix, g * 3)
    return packed


def unpack_3bit_ggml(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """Inverse of pack_3bit_ggml."""
    assert packed.dtype == torch.uint8
    assert orig_dim % 8 == 0
    g = orig_dim // 8
    assert packed.shape[-1] == g * 3, (
        f"packed last dim {packed.shape[-1]} != expected {g * 3}"
    )
    prefix = packed.shape[:-1]
    groups = packed.view(*prefix, g, 3).to(torch.int64)
    b0, b1, b2 = groups[..., 0], groups[..., 1], groups[..., 2]
    p24 = b0 | (b1 << 8) | (b2 << 16)
    idx = torch.stack(
        [
            (p24 >> 0) & 0x7,
            (p24 >> 3) & 0x7,
            (p24 >> 6) & 0x7,
            (p24 >> 9) & 0x7,
            (p24 >> 12) & 0x7,
            (p24 >> 15) & 0x7,
            (p24 >> 18) & 0x7,
            (p24 >> 21) & 0x7,
        ],
        dim=-1,
    ).to(torch.uint8).view(*prefix, orig_dim)
    return idx


# ── 4-bit nibble packing (2 indices per byte) ──────────────────────────

def pack_4bit_nibble(indices: torch.Tensor) -> torch.Tensor:
    assert indices.dtype == torch.uint8
    d = indices.shape[-1]
    assert d % 2 == 0
    lo = indices[..., 0::2] & 0x0F
    hi = (indices[..., 1::2] & 0x0F) << 4
    return lo | hi


def unpack_4bit_nibble(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] * 2 == orig_dim
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).view(*packed.shape[:-1], orig_dim)
    return out


# ── Index tables from a boolean outlier mask ───────────────────────────

def mask_to_indices(mask: torch.Tensor, tier_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-head outlier mask into outlier/regular column index tensors.

    Args:
      mask:      [H, D] bool — True = outlier dim
      tier_size: int — expected number of True per head (default 64 for A_35_prime)
    Returns:
      outlier_idx: [H, tier_size] int32 — column indices of outliers per head
      regular_idx: [H, D - tier_size] int32 — column indices of regulars per head
    """
    assert mask.dtype == torch.bool, f"mask must be bool, got {mask.dtype}"
    H, D = mask.shape
    counts = mask.sum(dim=-1)
    assert (counts == tier_size).all(), (
        f"Every head must have exactly {tier_size} outliers, "
        f"got per-head counts in [{counts.min().item()}, {counts.max().item()}]"
    )
    reg_size = D - tier_size
    outlier_idx = torch.zeros(H, tier_size, dtype=torch.int32, device=mask.device)
    regular_idx = torch.zeros(H, reg_size, dtype=torch.int32, device=mask.device)
    for h in range(H):
        o = mask[h].nonzero(as_tuple=False).flatten().to(torch.int32)
        r = (~mask[h]).nonzero(as_tuple=False).flatten().to(torch.int32)
        outlier_idx[h] = o
        regular_idx[h] = r
    return outlier_idx, regular_idx


# ── Two-tier MSE quantizer (A_35_prime) ────────────────────────────────

class A35PrimeQuantizer:
    """Two-tier MSE quantizer for a single attention layer.

    Both tiers are 64-dim sub-vectors of a head (head_dim=128). Each tier
    independently: L2-normalize → FWHT-64 rotate → scalar quantize → pack.

    One instance per layer. Sharing rotations across layers is safe (fixed
    seed), but splitting per layer keeps the kernel-side bindings uniform.
    """

    def __init__(
        self,
        outlier_mask: torch.Tensor,
        head_dim: int = 128,
        outlier_bits: int = 4,
        regular_bits: int = 3,
        outlier_tier_size: int = 64,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        if device is None:
            device = outlier_mask.device
        self.device = device
        self.head_dim = head_dim
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.outlier_tier_size = outlier_tier_size
        self.regular_tier_size = head_dim - outlier_tier_size

        assert outlier_mask.shape[-1] == head_dim
        self.num_heads = outlier_mask.shape[0]
        self.outlier_idx, self.regular_idx = mask_to_indices(
            outlier_mask.to(device), outlier_tier_size
        )

        self.out_rot = RandomHadamardRotation(
            self.outlier_tier_size, device=device, seed=seed
        )
        self.reg_rot = RandomHadamardRotation(
            self.regular_tier_size, device=device, seed=seed + 1
        )
        self.out_book = Codebook(self.outlier_tier_size, outlier_bits, device=device)
        self.reg_book = Codebook(self.regular_tier_size, regular_bits, device=device)

        self.out_bytes = (outlier_tier_size * outlier_bits) // 8  # 32 for A_35_prime
        self.reg_bytes = (self.regular_tier_size * regular_bits) // 8  # 24
        self.norm_bytes = 4  # 2 × fp16
        self.tile_bytes_raw = self.out_bytes + self.reg_bytes + self.norm_bytes  # 60
        self.tile_bytes_aligned = ((self.tile_bytes_raw + 63) // 64) * 64  # 64

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize K or V for one layer.

        Args:
          x: [..., H, D] float (fp16/fp32/bf16). Last two dims must be (H, D).
        Returns:
          tile: [..., H, tile_bytes_aligned] uint8
        """
        assert x.shape[-2] == self.num_heads and x.shape[-1] == self.head_dim
        prefix = x.shape[:-2]

        outlier_idx_long = self.outlier_idx.to(torch.long)
        regular_idx_long = self.regular_idx.to(torch.long)

        # Gather per-head into [..., H, tier_size]. `gather` over last dim,
        # broadcasting the prefix.
        x_f32 = x.to(torch.float32)
        out_sub = torch.gather(
            x_f32,
            -1,
            outlier_idx_long.expand(*prefix, -1, -1),
        )
        reg_sub = torch.gather(
            x_f32,
            -1,
            regular_idx_long.expand(*prefix, -1, -1),
        )

        out_packed, out_norms = self._quant_tier(
            out_sub, self.out_rot, self.out_book, self.outlier_bits
        )
        reg_packed, reg_norms = self._quant_tier(
            reg_sub, self.reg_rot, self.reg_book, self.regular_bits
        )

        # fp16 norms, packed as 2 bytes each → 4 bytes
        out_norm_bytes = out_norms.to(torch.float16).view(torch.uint8).view(
            *prefix, self.num_heads, 2
        )
        reg_norm_bytes = reg_norms.to(torch.float16).view(torch.uint8).view(
            *prefix, self.num_heads, 2
        )

        tile_raw = torch.cat([out_packed, reg_packed, out_norm_bytes, reg_norm_bytes], dim=-1)
        if self.tile_bytes_aligned > self.tile_bytes_raw:
            pad = torch.zeros(
                *tile_raw.shape[:-1],
                self.tile_bytes_aligned - self.tile_bytes_raw,
                dtype=torch.uint8,
                device=tile_raw.device,
            )
            tile_raw = torch.cat([tile_raw, pad], dim=-1)
        return tile_raw

    @torch.no_grad()
    def dequantize(self, tile: torch.Tensor) -> torch.Tensor:
        """Inverse of quantize.

        Args:
          tile: [..., H, tile_bytes_aligned] uint8
        Returns:
          x_hat: [..., H, D] float32
        """
        assert tile.shape[-2] == self.num_heads
        assert tile.shape[-1] == self.tile_bytes_aligned
        prefix = tile.shape[:-2]

        out_end = self.out_bytes
        reg_end = out_end + self.reg_bytes
        out_norm_end = reg_end + 2
        reg_norm_end = out_norm_end + 2

        out_packed = tile[..., :out_end].contiguous()
        reg_packed = tile[..., out_end:reg_end].contiguous()
        out_norm_bytes = tile[..., reg_end:out_norm_end].contiguous()
        reg_norm_bytes = tile[..., out_norm_end:reg_norm_end].contiguous()

        out_norms = out_norm_bytes.view(torch.float16).to(torch.float32).squeeze(-1)
        reg_norms = reg_norm_bytes.view(torch.float16).to(torch.float32).squeeze(-1)

        out_sub = self._dequant_tier(
            out_packed, out_norms, self.out_rot, self.out_book,
            self.outlier_bits, self.outlier_tier_size,
        )
        reg_sub = self._dequant_tier(
            reg_packed, reg_norms, self.reg_rot, self.reg_book,
            self.regular_bits, self.regular_tier_size,
        )

        # Scatter back into [..., H, D]
        x_hat = torch.zeros(
            *prefix, self.num_heads, self.head_dim,
            dtype=torch.float32, device=tile.device,
        )
        outlier_idx_long = self.outlier_idx.to(torch.long).expand(
            *prefix, -1, -1
        )
        regular_idx_long = self.regular_idx.to(torch.long).expand(
            *prefix, -1, -1
        )
        x_hat.scatter_(-1, outlier_idx_long, out_sub)
        x_hat.scatter_(-1, regular_idx_long, reg_sub)
        return x_hat

    def _quant_tier(self, sub, rot, book, bits):
        norms = sub.norm(dim=-1).clamp(min=1e-8)
        unit = sub / norms.unsqueeze(-1)
        y = rot.rotate(unit)
        idx = book.quantize(y)  # uint8, same shape as y
        if bits == 4:
            packed = pack_4bit_nibble(idx)
        elif bits == 3:
            packed = pack_3bit_ggml(idx)
        else:
            raise ValueError(f"Unsupported bits={bits}")
        return packed, norms

    def _dequant_tier(self, packed, norms, rot, book, bits, tier_size):
        if bits == 4:
            idx = unpack_4bit_nibble(packed, tier_size)
        elif bits == 3:
            idx = unpack_3bit_ggml(packed, tier_size)
        else:
            raise ValueError(f"Unsupported bits={bits}")
        y = book.dequantize(idx)
        unit = rot.inverse_rotate(y)
        return unit * norms.unsqueeze(-1)

    @property
    def bits_per_dim(self) -> float:
        total_bits = (
            self.outlier_tier_size * self.outlier_bits
            + self.regular_tier_size * self.regular_bits
        )
        return total_bits / self.head_dim
