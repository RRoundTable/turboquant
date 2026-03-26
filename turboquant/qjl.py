"""Quantized Johnson-Lindenstrauss (QJL) transform.

1-bit inner product quantizer providing unbiased inner product estimation
with variance <= pi/(2d) * ||y||^2.

Reference: Zandieh et al., "QJL: 1-bit Quantized JL Transform for KV Cache
Quantization with Zero Overhead", 2024.
"""

import math

import torch


class QJL:
    """QJL: 1-bit quantized Johnson-Lindenstrauss transform.

    For x on the unit sphere S^{d-1}:
        Quantize:    Q(x) = sign(S * x)
        Dequantize:  Q^{-1}(z) = sqrt(pi/2) / d * S^T * z
    """

    def __init__(self, dim: int, device: torch.device = "cuda", seed: int = 1042):
        self.dim = dim
        self.dequant_scale = math.sqrt(math.pi / 2.0) / dim

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.S = torch.randn(dim, dim, generator=gen, device="cpu").to(device)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """sign(S * x) — returns int8 tensor of {-1, +1}.

        Args:
            x: [..., d] tensor (should be on unit sphere for guarantees)
        """
        projected = torch.matmul(x.float(), self.S.t())
        signs = projected.sign()
        signs[signs == 0] = 1.0
        return signs.to(torch.int8)

    @torch.no_grad()
    def dequantize_inner_product(
        self, signs: torch.Tensor, residual_norm: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize for inner product: sqrt(pi/2)/d * gamma * S^T * z.

        Args:
            signs: [..., d] int8 tensor of {-1, +1}
            residual_norm: [...] tensor of residual L2 norms (gamma)
        Returns:
            [..., d] contribution to the dequantized vector
        """
        result = torch.matmul(signs.float(), self.S)
        return result * (self.dequant_scale * residual_norm.unsqueeze(-1))
