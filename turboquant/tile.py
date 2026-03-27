"""PyTorch reference implementation of TurboQuantTile (Lloyd-Max codebook).

Exact port of the C++ tile layout (csrc/include/turboquant/tile.h) to PyTorch.
This serves as the ground truth for verifying CUDA kernels.

Tile layout: 16 tokens × 64 dims, 480 bytes, 16-byte aligned.
  - norms:    [16] FP16 L2 norm per row              (32 B, offset 0)
  - quant_hi: [16 × 16] 4-bit unsigned codebook idx  (256 B, offset 32)
  - quant_lo: [16 × 12] 3-bit unsigned codebook idx  (192 B, offset 288)

After Hadamard rotation, coordinates ~N(0, 1/d). Lloyd-Max centroids for N(0,1)
are scaled by 1/√d. Indices are unsigned — centroids carry the sign.
"""

import math

import torch

from .codebook import _CENTROIDS_N01, _BOUNDARIES_N01


# Tile constants (must match csrc/include/turboquant/tile.h)
TILE_TOKENS = 16
TILE_DIMS = 64
HI_DIMS = 32        # first 32 dims → 4-bit codebook
LO_DIMS = 32        # last 32 dims → 3-bit codebook
HI_BITS = 4
LO_BITS = 3
HI_LEVELS = 16      # 2^4
LO_LEVELS = 8       # 2^3
HI_BYTES_PER_ROW = 16  # 32 dims × 4 bits / 8
LO_BYTES_PER_ROW = 12  # 32 dims × 3 bits / 8
TILE_BYTES = 32 + 256 + 192  # 480


@torch.no_grad()
def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit values: 2 per byte, high nibble = even, low nibble = odd."""
    even = indices[..., 0::2]
    odd = indices[..., 1::2]
    return ((even << 4) | (odd & 0x0F)).to(torch.uint8)


@torch.no_grad()
def unpack_4bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack 4-bit nibble-packed bytes."""
    p = packed.to(torch.int32)
    even = (p >> 4) & 0x0F
    odd = p & 0x0F
    batch_shape = packed.shape[:-1]
    out = torch.zeros(*batch_shape, count, dtype=torch.int32, device=packed.device)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out.to(torch.uint8)


@torch.no_grad()
def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit values: 8 per 3 bytes, GGML-compatible little-endian."""
    batch_shape = indices.shape[:-1]
    n = indices.shape[-1]
    assert n % 8 == 0
    v = indices.view(*batch_shape, n // 8, 8).to(torch.int32)

    b0 = (v[..., 0] & 7) | ((v[..., 1] & 7) << 3) | ((v[..., 2] & 3) << 6)
    b1 = ((v[..., 2] >> 2) & 1) | ((v[..., 3] & 7) << 1) | ((v[..., 4] & 7) << 4) | ((v[..., 5] & 1) << 7)
    b2 = ((v[..., 5] >> 1) & 3) | ((v[..., 6] & 7) << 2) | ((v[..., 7] & 7) << 5)

    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return packed.view(*batch_shape, -1)


@torch.no_grad()
def unpack_3bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack GGML-compatible 3-bit packed bytes."""
    batch_shape = packed.shape[:-1]
    p = packed.view(*batch_shape, -1, 3).to(torch.int32)

    v0 = p[..., 0] & 7
    v1 = (p[..., 0] >> 3) & 7
    v2 = ((p[..., 0] >> 6) & 3) | ((p[..., 1] & 1) << 2)
    v3 = (p[..., 1] >> 1) & 7
    v4 = (p[..., 1] >> 4) & 7
    v5 = ((p[..., 1] >> 7) & 1) | ((p[..., 2] & 3) << 1)
    v6 = (p[..., 2] >> 2) & 7
    v7 = (p[..., 2] >> 5) & 7

    out = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).to(torch.uint8)
    return out.view(*batch_shape, -1)[..., :count]


def _make_codebook_tensors(bit_width: int, dim: int, device):
    """Create scaled centroids and boundaries tensors for a given bit-width and dim."""
    scale = 1.0 / math.sqrt(dim)
    centroids = torch.tensor(
        [c * scale for c in _CENTROIDS_N01[bit_width]],
        dtype=torch.float32, device=device,
    )
    boundaries = torch.tensor(
        [b * scale for b in _BOUNDARIES_N01[bit_width]],
        dtype=torch.float32, device=device,
    )
    return centroids, boundaries


@torch.no_grad()
def quantize_tile(x: torch.Tensor, dim: int = TILE_DIMS) -> torch.Tensor:
    """Quantize a [16, 64] float tensor into a flat 480-byte TurboQuantTile.

    Input should be Hadamard-rotated vectors. L2 norms are extracted and stored.
    Lloyd-Max codebook indices are computed using precomputed boundaries.

    Args:
        x: [16, 64] float tensor (one tile, already Hadamard-rotated)
        dim: padded dimension for codebook scaling (default: 64)
    Returns:
        [480] uint8 tensor matching C++ TurboQuantTile memory layout
    """
    assert x.shape == (TILE_TOKENS, TILE_DIMS), f"Expected [{TILE_TOKENS}, {TILE_DIMS}], got {list(x.shape)}"

    hi_centroids, hi_boundaries = _make_codebook_tensors(HI_BITS, dim, x.device)
    lo_centroids, lo_boundaries = _make_codebook_tensors(LO_BITS, dim, x.device)

    tile = torch.zeros(TILE_BYTES, dtype=torch.uint8, device=x.device)

    for row in range(TILE_TOKENS):
        src = x[row]

        # Compute L2 norm
        l2_norm = src.norm()

        # Store as FP16
        norm_fp16 = l2_norm.to(torch.float16).reshape(1)
        norm_bytes = norm_fp16.view(torch.uint8)
        tile[row * 2] = norm_bytes[0]
        tile[row * 2 + 1] = norm_bytes[1]

        # Recover FP16-rounded norm for quantization consistency
        norm = norm_fp16.float().squeeze()
        inv_norm = (1.0 / norm) if norm > 0 else torch.tensor(0.0, device=x.device)

        # Normalize by L2 norm
        normalized = src * inv_norm

        # First 32 dims → 4-bit codebook
        hi_indices = torch.bucketize(normalized[:HI_DIMS], hi_boundaries).to(torch.uint8)
        hi_packed = pack_4bit(hi_indices)
        hi_offset = 32 + row * HI_BYTES_PER_ROW
        tile[hi_offset: hi_offset + HI_BYTES_PER_ROW] = hi_packed

        # Last 32 dims → 3-bit codebook
        lo_indices = torch.bucketize(normalized[HI_DIMS:], lo_boundaries).to(torch.uint8)
        lo_packed = pack_3bit(lo_indices)
        lo_offset = 288 + row * LO_BYTES_PER_ROW
        tile[lo_offset: lo_offset + LO_BYTES_PER_ROW] = lo_packed

    return tile


@torch.no_grad()
def dequantize_tile(tile: torch.Tensor, dim: int = TILE_DIMS) -> torch.Tensor:
    """Dequantize a 480-byte TurboQuantTile back to [16, 64] float.

    Args:
        tile: [480] uint8 tensor
        dim: padded dimension for codebook scaling (default: 64)
    Returns:
        [16, 64] float tensor
    """
    assert tile.shape == (TILE_BYTES,), f"Expected [{TILE_BYTES}], got {list(tile.shape)}"

    hi_centroids, _ = _make_codebook_tensors(HI_BITS, dim, tile.device)
    lo_centroids, _ = _make_codebook_tensors(LO_BITS, dim, tile.device)

    output = torch.zeros(TILE_TOKENS, TILE_DIMS, dtype=torch.float32, device=tile.device)

    for row in range(TILE_TOKENS):
        # Read FP16 L2 norm
        norm_bytes = tile[row * 2: row * 2 + 2].contiguous()
        norm = norm_bytes.view(torch.float16).float().squeeze()

        # Dequantize first 32 dims (4-bit codebook)
        hi_offset = 32 + row * HI_BYTES_PER_ROW
        hi_packed = tile[hi_offset: hi_offset + HI_BYTES_PER_ROW]
        hi_indices = unpack_4bit(hi_packed, HI_DIMS)
        output[row, :HI_DIMS] = hi_centroids[hi_indices.long()] * norm

        # Dequantize last 32 dims (3-bit codebook)
        lo_offset = 288 + row * LO_BYTES_PER_ROW
        lo_packed = tile[lo_offset: lo_offset + LO_BYTES_PER_ROW]
        lo_indices = unpack_3bit(lo_packed, LO_DIMS)
        output[row, HI_DIMS:] = lo_centroids[lo_indices.long()] * norm

    return output
