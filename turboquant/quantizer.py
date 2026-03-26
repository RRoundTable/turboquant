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


class TurboQuantMSE:
    """TurboQuant optimized for MSE (Algorithm 1).

    Steps:
      Quantize:   norm → normalize → random Hadamard rotate → scalar quantize
      Dequantize: centroid lookup → inverse rotate → rescale by norm

    Distortion: D_mse <= sqrt(3*pi)/2 * 4^{-b}  for unit vectors.
    """

    def __init__(self, dim: int, bit_width: int, device=None, seed: int = 42):
        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        self.rotation = RandomHadamardRotation(dim, device=device, seed=seed)
        # Codebook operates in the padded dimension
        self.codebook = Codebook(self.rotation.padded_dim, bit_width, device=device)

    def to(self, device):
        self.device = device
        self.rotation.to(device)
        self.codebook.to(device)
        return self

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """Quantize vectors.

        Args:
            x: [..., d] float tensor
        Returns:
            indices: [..., d_padded] uint8 tensor of codebook indices
            norms:   [...] float tensor of L2 norms
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x / norms
        y = self.rotation.rotate(x_norm)
        indices = self.codebook.quantize(y)
        return indices, norms.squeeze(-1)

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """Dequantize back to vectors.

        Args:
            indices: [..., d_padded] uint8
            norms:   [...] float
        Returns:
            x_hat: [..., d] reconstructed vectors
        """
        y_hat = self.codebook.dequantize(indices)
        x_hat = self.rotation.inverse_rotate(y_hat)
        return x_hat * norms.unsqueeze(-1)


class TurboQuantProd:
    """TurboQuant optimized for inner product (Algorithm 2).

    Two-stage quantizer:
      1. MSE quantizer with (b-1) bits → minimizes ||residual||
      2. QJL (1-bit) on the residual → makes inner-product unbiased

    Guarantees:
      E[<y, x_hat>] = <y, x>                     (unbiased)
      D_prod <= sqrt(3)*pi^2 / (2d) * 4^{-b}     (low distortion)
    """

    def __init__(self, dim: int, bit_width: int, device=None, seed: int = 42):
        assert bit_width >= 2, "Inner-product variant needs bit_width >= 2"

        self.dim = dim
        self.bit_width = bit_width
        self.device = device

        self.mse = TurboQuantMSE(dim, bit_width - 1, device=device, seed=seed)
        self.qjl = QJL(dim, device=device, seed=seed + 10000)

    def to(self, device):
        self.device = device
        self.mse.to(device)
        self.qjl.to(device)
        return self

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """Quantize for unbiased inner-product estimation.

        Returns:
            mse_indices:    [..., d_padded] uint8
            mse_norms:      [...] float
            qjl_signs:      [..., d] int8  ({-1, +1})
            residual_norms: [...] float
        """
        mse_indices, mse_norms = self.mse.quantize(x)
        x_hat = self.mse.dequantize(mse_indices, mse_norms)

        residual = x - x_hat
        residual_norms = residual.norm(dim=-1).clamp(min=1e-8)
        residual_unit = residual / residual_norms.unsqueeze(-1)

        qjl_signs = self.qjl.quantize(residual_unit)
        return mse_indices, mse_norms, qjl_signs, residual_norms

    @torch.no_grad()
    def dequantize(
        self,
        mse_indices: torch.Tensor,
        mse_norms: torch.Tensor,
        qjl_signs: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize — sum of MSE reconstruction and QJL residual.

        Returns:
            x_hat: [..., d] unbiased inner-product reconstruction
        """
        x_mse = self.mse.dequantize(mse_indices, mse_norms)
        x_qjl = self.qjl.dequantize_inner_product(qjl_signs, residual_norms)
        return x_mse + x_qjl
