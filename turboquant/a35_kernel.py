"""HYP-056: Python wrapper around the A_35_prime CUDA write kernel.

JIT-compiles `csrc/src/quantize_write_a35_binding.cu` on first use and
exposes `quantize_write_a35(k_or_v, quantizer) -> packed_tiles`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .outlier_aware_a35 import A35PrimeQuantizer

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load
    csrc = Path(__file__).resolve().parent.parent / "csrc"
    _MODULE = load(
        name="turboquant_write_a35",
        sources=[str(csrc / "src" / "quantize_write_a35_binding.cu")],
        extra_include_paths=[str(csrc / "include")],
        extra_cuda_cflags=[
            "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        ],
        verbose=False,
    )
    return _MODULE


def _idx_and_signs(quantizer: A35PrimeQuantizer):
    return (
        quantizer.outlier_idx.to(torch.int32).contiguous(),
        quantizer.regular_idx.to(torch.int32).contiguous(),
        quantizer.out_rot.signs.to(torch.float32).contiguous(),
        quantizer.reg_rot.signs.to(torch.float32).contiguous(),
    )


@torch.no_grad()
def quantize_write_a35(
    x: torch.Tensor,
    quantizer: A35PrimeQuantizer,
) -> torch.Tensor:
    """Call the CUDA write kernel with quantizer's index tables + FWHT signs.

    Args:
      x:         [..., H, 128] fp16 tensor (K or V).
      quantizer: A35PrimeQuantizer with matching num_heads and head_dim=128.

    Returns:
      tiles:     [..., H, 64] uint8.
    """
    assert x.is_cuda, "kernel requires CUDA tensor"
    assert x.dtype == torch.float16, f"kernel requires fp16, got {x.dtype}"
    assert x.shape[-2] == quantizer.num_heads
    assert x.shape[-1] == quantizer.head_dim == 128

    mod = _get_module()
    outlier_idx, regular_idx, signs_out, signs_reg = _idx_and_signs(quantizer)
    return mod.quantize_write_a35(x, outlier_idx, regular_idx, signs_out, signs_reg)


@torch.no_grad()
def dequantize_a35(
    tiles: torch.Tensor,
    quantizer: A35PrimeQuantizer,
) -> torch.Tensor:
    """Inverse of `quantize_write_a35` via CUDA kernel.

    Args:
      tiles:     [..., H, 64] uint8.
      quantizer: A35PrimeQuantizer with matching num_heads.

    Returns:
      kv_out:    [..., H, 128] fp16.
    """
    assert tiles.is_cuda
    assert tiles.dtype == torch.uint8
    assert tiles.shape[-2] == quantizer.num_heads
    assert tiles.shape[-1] == 64
    mod = _get_module()
    outlier_idx, regular_idx, signs_out, signs_reg = _idx_and_signs(quantizer)
    return mod.dequantize_a35(tiles, outlier_idx, regular_idx, signs_out, signs_reg)
