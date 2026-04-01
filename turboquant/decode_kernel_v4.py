"""TurboQuant v4 decode kernel — generalized for any model architecture.

JIT-compiles the v4 CUDA kernel (fused inline dequant, no fp16 smem)
and auto-dispatches based on model config (head_dim, GQA ratio).

Usage:
    from turboquant.decode_kernel_v4 import TurboQuantDecoderV4
    from turboquant.kernel_config import MODEL_PRESETS

    model = MODEL_PRESETS["llama-3-8b"]
    decoder = TurboQuantDecoderV4.from_model_config(model, page_size=16)
    output = decoder.decode(q, k_quant, v_quant, k_norms, v_norms,
                            indices, indptr, last_page_len)
"""

import functools
import math
import os
from pathlib import Path
from typing import Optional

import torch

from .kernel_config import ModelConfig, KernelConfig, compute_kernel_config

_CSRC_DIR = Path(__file__).parent.parent / "csrc"
_INCLUDE_DIR = _CSRC_DIR / "include"


def _find_flashinfer_include() -> Path:
    """Find FlashInfer include directory."""
    # Check environment variable first
    env_path = os.environ.get("FLASHINFER_INCLUDE_DIR")
    if env_path:
        return Path(env_path)

    # Try flashinfer-python package
    try:
        import flashinfer
        pkg_include = Path(flashinfer.__file__).parent / "data" / "include"
        if (pkg_include / "flashinfer" / "attention" / "decode.cuh").exists():
            return pkg_include
    except ImportError:
        pass

    # Fallback to local clone
    local = Path(os.path.expanduser("~/workdir/flashinfer/include"))
    if local.exists():
        return local

    raise RuntimeError(
        "FlashInfer headers not found. Set FLASHINFER_INCLUDE_DIR or "
        "install flashinfer-python."
    )


@functools.cache
def _get_module():
    """JIT-compile the v4 decode kernel."""
    from torch.utils.cpp_extension import load

    fi_include = _find_flashinfer_include()
    binding_src = _CSRC_DIR / "src" / "decode_v4_binding.cu"

    return load(
        name="turboquant_decode_v4",
        sources=[str(binding_src)],
        extra_include_paths=[str(_INCLUDE_DIR), str(fi_include)],
        extra_cuda_cflags=[
            "-std=c++17", "-O3", "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


class TurboQuantDecoderV4:
    """Fused decode attention with inline dequant (v4 kernel).

    Supports any model architecture via generalized dispatch.
    Auto-selects optimal thread config based on head_dim and GQA ratio.
    """

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
    ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size

        self.model_config = ModelConfig(num_qo_heads, num_kv_heads, head_dim)
        self.kernel_config = compute_kernel_config(self.model_config)
        self.padded_dim = self.model_config.padded_dim

    @classmethod
    def from_model_config(cls, config: ModelConfig, page_size: int = 16):
        return cls(config.num_qo_heads, config.num_kv_heads,
                   config.head_dim, page_size)

    @classmethod
    def from_model_name(cls, name: str, page_size: int = 16):
        from .kernel_config import MODEL_PRESETS
        config = MODEL_PRESETS[name]
        return cls.from_model_config(config, page_size)

    @torch.no_grad()
    def decode(
        self,
        q: torch.Tensor,                   # [batch, num_qo_heads, head_dim] fp16
        k_quant: torch.Tensor,             # flat uint8
        v_quant: torch.Tensor,             # flat uint8
        k_norms: torch.Tensor,             # flat fp16
        v_norms: torch.Tensor,             # flat fp16
        kv_indices: torch.Tensor,          # [total_pages] int32
        kv_indptr: torch.Tensor,           # [batch + 1] int32
        kv_last_page_len: torch.Tensor,    # [batch] int32
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run fused decode attention on quantized KV cache.

        Returns: [batch, num_qo_heads, head_dim] fp16
        """
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(self.head_dim)

        module = _get_module()
        return module.decode_v4(
            q, k_quant, v_quant, k_norms, v_norms,
            kv_indices, kv_indptr, kv_last_page_len,
            self.num_kv_heads, self.page_size,
            self.head_dim, self.padded_dim, sm_scale,
        )

    def __repr__(self):
        kc = self.kernel_config
        return (f"TurboQuantDecoderV4("
                f"qo={self.num_qo_heads}, kv={self.num_kv_heads}, "
                f"hd={self.head_dim}, "
                f"bdx={kc.bdx}, bdy={kc.bdy}, bdz={kc.bdz}, "
                f"threads={kc.threads_per_block})")
