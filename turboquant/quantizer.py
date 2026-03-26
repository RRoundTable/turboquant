"""Core TurboQuant quantizer implementations.

Two variants:
  TurboQuantMSE  — Algorithm 1: minimizes mean-squared error
  TurboQuantProd — Algorithm 2: unbiased inner-product estimation

Both are data-oblivious (online) and accelerator-friendly.
"""

import torch

from .codebook import Codebook
from .hadamard import RandomHadamardRotation
from .qjl import QJL


def _pack_bits(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Pack low-bit indices into bytes.

    Packs multiple indices into each byte for true compression.
    E.g., 3-bit: not evenly divisible, so pack 2 values per byte (6 bits used, 2 wasted).
         2-bit: 4 values per byte.
         1-bit: 8 values per byte.
         4-bit: 2 values per byte.
    """
    vals_per_byte = 8 // bit_width
    d = indices.shape[-1]
    pad_to = (d + vals_per_byte - 1) // vals_per_byte * vals_per_byte
    if pad_to > d:
        indices = torch.nn.functional.pad(indices, (0, pad_to - d))

    batch_shape = indices.shape[:-1]
    indices = indices.view(*batch_shape, -1, vals_per_byte).to(torch.uint8)

    packed = torch.zeros(*batch_shape, indices.shape[-2], dtype=torch.uint8, device=indices.device)
    for i in range(vals_per_byte):
        packed |= indices[..., i] << (i * bit_width)

    return packed


def _unpack_bits(packed: torch.Tensor, bit_width: int, orig_dim: int) -> torch.Tensor:
    """Unpack bytes back to individual indices."""
    vals_per_byte = 8 // bit_width
    mask = (1 << bit_width) - 1

    batch_shape = packed.shape[:-1]
    unpacked = torch.zeros(
        *batch_shape, packed.shape[-1] * vals_per_byte,
        dtype=torch.uint8,
        device=packed.device,
    )

    for i in range(vals_per_byte):
        unpacked[..., i::vals_per_byte] = (packed >> (i * bit_width)) & mask

    return unpacked[..., :orig_dim]


class TurboQuantMSE:
    """TurboQuant optimized for MSE (Algorithm 1).

    Steps:
      Quantize:   norm → normalize → random Hadamard rotate → scalar quantize → bit-pack
      Dequantize: unpack → centroid lookup → inverse rotate → rescale by norm
    """

    def __init__(self, dim: int, bit_width: int, device: torch.device = "cuda", seed: int = 42):
        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        self.rotation = RandomHadamardRotation(dim, device=device, seed=seed)
        self.codebook = Codebook(self.rotation.padded_dim, bit_width, device=device)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """Quantize vectors.

        Args:
            x: [..., d] float tensor
        Returns:
            packed: [..., packed_dim] uint8 tensor of bit-packed codebook indices
            norms:  [...] float tensor of L2 norms
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x / norms
        y = self.rotation.rotate(x_norm)
        indices = self.codebook.quantize(y)
        packed = _pack_bits(indices, self.bit_width)
        return packed, norms.squeeze(-1)

    @torch.no_grad()
    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize back to vectors.

        Args:
            packed: [..., packed_dim] uint8
            norms:  [...] float
        Returns:
            x_hat: [..., d] reconstructed vectors
        """
        indices = _unpack_bits(packed, self.bit_width, self.rotation.padded_dim)
        y_hat = self.codebook.dequantize(indices)
        x_hat = self.rotation.inverse_rotate(y_hat)
        return x_hat * norms.unsqueeze(-1)


class TurboQuantProd:
    """TurboQuant optimized for inner product (Algorithm 2).

    Two-stage quantizer:
      1. MSE quantizer with (b-1) bits → minimizes ||residual||
      2. QJL (1-bit) on the residual → makes inner-product unbiased
    """

    def __init__(self, dim: int, bit_width: int, device: torch.device = "cuda", seed: int = 42):
        assert bit_width >= 2, "Inner-product variant needs bit_width >= 2"

        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        self.mse = TurboQuantMSE(dim, bit_width - 1, device=device, seed=seed)
        self.qjl = QJL(dim, device=device, seed=seed + 10000)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """Quantize for unbiased inner-product estimation.

        Returns:
            mse_packed:     [..., packed_dim] uint8
            mse_norms:      [...] float
            qjl_signs:      [..., d] int8  ({-1, +1})
            residual_norms: [...] float
        """
        mse_packed, mse_norms = self.mse.quantize(x)
        x_hat = self.mse.dequantize(mse_packed, mse_norms)

        residual = x - x_hat
        residual_norms = residual.norm(dim=-1).clamp(min=1e-8)
        residual_unit = residual / residual_norms.unsqueeze(-1)

        qjl_signs = self.qjl.quantize(residual_unit)
        return mse_packed, mse_norms, qjl_signs, residual_norms

    @torch.no_grad()
    def dequantize(
        self,
        mse_packed: torch.Tensor,
        mse_norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize — sum of MSE reconstruction and QJL residual."""
        x_mse = self.mse.dequantize(mse_packed, mse_norms)
        x_qjl = self.qjl.dequantize_inner_product(qjl_signs, residual_norms)
        return x_mse + x_qjl
