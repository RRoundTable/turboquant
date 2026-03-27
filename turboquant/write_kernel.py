"""Phase 2 Write Kernel: Fused quantize-and-write for TurboQuantTile.

PyTorch eager implementation of the fused write path:
  Input KV tensor → L2 normalize → Hadamard rotate → Lloyd-Max quantize → bit-pack → TurboQuantTile

This runs entirely on GPU. It is the reference that the CUDA/Triton kernel must match.
RoPE is NOT included here — vLLM applies RoPE before the KV cache write.
"""

import math
from typing import Tuple

import torch

from .codebook import _CENTROIDS_N01, _BOUNDARIES_N01
from .hadamard import RandomHadamardRotation
from .tile import (
    TILE_TOKENS, TILE_DIMS, TILE_BYTES,
    HI_DIMS, LO_DIMS, HI_BITS, LO_BITS,
    HI_BYTES_PER_ROW, LO_BYTES_PER_ROW,
    pack_4bit, pack_3bit,
    unpack_4bit, unpack_3bit,
)


class TurboQuantWriter:
    """Fused quantize-and-write for KV cache vectors.

    Handles the full pipeline: L2 normalize → Hadamard rotate → tile quantize.
    Produces TurboQuantTile-format byte tensors ready for VRAM storage.
    """

    def __init__(self, head_dim: int, device: torch.device = "cuda", seed: int = 42):
        self.head_dim = head_dim
        self.device = device
        self.rotation = RandomHadamardRotation(head_dim, device=device, seed=seed)
        self.padded_dim = self.rotation.padded_dim

        scale = 1.0 / math.sqrt(self.padded_dim)
        self.hi_boundaries = torch.tensor(
            [b * scale for b in _BOUNDARIES_N01[HI_BITS]],
            dtype=torch.float32, device=device,
        )
        self.lo_boundaries = torch.tensor(
            [b * scale for b in _BOUNDARIES_N01[LO_BITS]],
            dtype=torch.float32, device=device,
        )
        self.hi_centroids = torch.tensor(
            [c * scale for c in _CENTROIDS_N01[HI_BITS]],
            dtype=torch.float32, device=device,
        )
        self.lo_centroids = torch.tensor(
            [c * scale for c in _CENTROIDS_N01[LO_BITS]],
            dtype=torch.float32, device=device,
        )

    @torch.no_grad()
    def quantize_kv(
        self, kv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV vectors into TurboQuantTile byte tensors.

        Full pipeline: L2 normalize → Hadamard rotate → codebook quantize → bit-pack.

        For head_dim > TILE_DIMS (64), the rotated vector is split into chunks
        of TILE_DIMS, each stored as a separate tile. E.g., head_dim=128 →
        2 tiles per token row, with norms stored in the first tile only.

        Args:
            kv: [num_tokens, head_dim] float tensor
        Returns:
            tiles: [num_token_tiles, tiles_per_row, TILE_BYTES] uint8
            metadata: dict with 'num_tokens', 'padded_dim', 'tiles_per_row'
        """
        num_tokens = kv.shape[0]
        assert kv.shape[-1] == self.head_dim

        # Pad tokens to multiple of TILE_TOKENS
        pad_tokens = (TILE_TOKENS - num_tokens % TILE_TOKENS) % TILE_TOKENS
        if pad_tokens > 0:
            kv = torch.nn.functional.pad(kv, (0, 0, 0, pad_tokens))

        num_padded = kv.shape[0]
        num_token_tiles = num_padded // TILE_TOKENS

        # L2 normalize
        norms = kv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        kv_unit = kv / norms

        # Hadamard rotate
        rotated = self.rotation.rotate(kv_unit)  # [N, padded_dim]

        # Split rotated dims into chunks of TILE_DIMS
        tiles_per_row = (self.padded_dim + TILE_DIMS - 1) // TILE_DIMS

        # Reshape into token tiles
        rotated_token_tiles = rotated.view(num_token_tiles, TILE_TOKENS, self.padded_dim)
        norms_token_tiles = norms.squeeze(-1).view(num_token_tiles, TILE_TOKENS)

        # Allocate output: [num_token_tiles, tiles_per_row, TILE_BYTES]
        tiles = torch.zeros(
            num_token_tiles, tiles_per_row, TILE_BYTES,
            dtype=torch.uint8, device=kv.device,
        )

        for tt in range(num_token_tiles):
            tile_norms = norms_token_tiles[tt]
            norms_fp16 = tile_norms.to(torch.float16)
            norms_bytes = norms_fp16.view(torch.uint8)

            for chunk_idx in range(tiles_per_row):
                dim_start = chunk_idx * TILE_DIMS
                dim_end = min(dim_start + TILE_DIMS, self.padded_dim)
                chunk = rotated_token_tiles[tt, :, dim_start:dim_end]

                # Pad chunk to TILE_DIMS if needed (last chunk may be smaller)
                if chunk.shape[-1] < TILE_DIMS:
                    chunk = torch.nn.functional.pad(chunk, (0, TILE_DIMS - chunk.shape[-1]))

                # Store norms in every chunk tile (needed for dequant)
                tiles[tt, chunk_idx, :TILE_TOKENS * 2] = norms_bytes

                # Quantize
                hi_vals = chunk[:, :HI_DIMS].contiguous()
                lo_vals = chunk[:, HI_DIMS:TILE_DIMS].contiguous()
                hi_indices = torch.bucketize(hi_vals, self.hi_boundaries).to(torch.uint8)
                lo_indices = torch.bucketize(lo_vals, self.lo_boundaries).to(torch.uint8)

                for row in range(TILE_TOKENS):
                    hi_packed = pack_4bit(hi_indices[row])
                    hi_offset = 32 + row * HI_BYTES_PER_ROW
                    tiles[tt, chunk_idx, hi_offset: hi_offset + HI_BYTES_PER_ROW] = hi_packed

                    lo_packed = pack_3bit(lo_indices[row])
                    lo_offset = 288 + row * LO_BYTES_PER_ROW
                    tiles[tt, chunk_idx, lo_offset: lo_offset + LO_BYTES_PER_ROW] = lo_packed

        return tiles, {
            "num_tokens": num_tokens,
            "padded_dim": self.padded_dim,
            "tiles_per_row": tiles_per_row,
        }

    @torch.no_grad()
    def dequantize_kv(
        self, tiles: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Dequantize TurboQuantTile byte tensors back to KV vectors.

        Reverse pipeline: unpack → centroid lookup → reassemble chunks → inverse Hadamard → rescale.

        Args:
            tiles: [num_token_tiles, tiles_per_row, TILE_BYTES] uint8
            num_tokens: original number of tokens (before padding)
        Returns:
            kv: [num_tokens, head_dim] float tensor
        """
        num_token_tiles = tiles.shape[0]
        tiles_per_row = tiles.shape[1]
        device = tiles.device

        total_tokens = num_token_tiles * TILE_TOKENS
        all_rotated = torch.zeros(
            total_tokens, self.padded_dim, dtype=torch.float32, device=device,
        )
        all_norms = torch.zeros(total_tokens, dtype=torch.float32, device=device)

        for tt in range(num_token_tiles):
            # Read norms from first chunk tile
            norms_bytes = tiles[tt, 0, :TILE_TOKENS * 2].contiguous()
            norms = norms_bytes.view(torch.float16).float()
            all_norms[tt * TILE_TOKENS: (tt + 1) * TILE_TOKENS] = norms

            for chunk_idx in range(tiles_per_row):
                dim_start = chunk_idx * TILE_DIMS
                dim_end = min(dim_start + TILE_DIMS, self.padded_dim)
                chunk_dim = dim_end - dim_start

                base = tt * TILE_TOKENS
                for row in range(TILE_TOKENS):
                    # 4-bit dequant
                    hi_offset = 32 + row * HI_BYTES_PER_ROW
                    hi_packed = tiles[tt, chunk_idx, hi_offset: hi_offset + HI_BYTES_PER_ROW]
                    hi_indices = unpack_4bit(hi_packed, HI_DIMS)
                    hi_vals = self.hi_centroids[hi_indices.long()]
                    hi_end = min(HI_DIMS, chunk_dim)
                    all_rotated[base + row, dim_start: dim_start + hi_end] = hi_vals[:hi_end]

                    # 3-bit dequant
                    lo_offset = 288 + row * LO_BYTES_PER_ROW
                    lo_packed = tiles[tt, chunk_idx, lo_offset: lo_offset + LO_BYTES_PER_ROW]
                    lo_indices = unpack_3bit(lo_packed, LO_DIMS)
                    lo_vals = self.lo_centroids[lo_indices.long()]
                    lo_end = min(LO_DIMS, chunk_dim - HI_DIMS) if chunk_dim > HI_DIMS else 0
                    if lo_end > 0:
                        all_rotated[base + row, dim_start + HI_DIMS: dim_start + HI_DIMS + lo_end] = lo_vals[:lo_end]

        # Inverse Hadamard rotation
        kv_unit = self.rotation.inverse_rotate(all_rotated)

        # Rescale by norms
        kv = kv_unit * all_norms.unsqueeze(-1)

        return kv[:num_tokens]
