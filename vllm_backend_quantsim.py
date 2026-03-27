# TurboQuant backend: measures quantization quality impact.
# Stores fp16 in cache (same as FlashAttention) but simulates quantization
# by quantizing+dequantizing K/V BEFORE writing to cache.
# This isolates the quality effect of 3.5-bit quantization.

import math
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

TILE_DIMS = 64
HI_DIMS, LO_DIMS = 32, 32

_C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
        0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
_B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
        0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]
_C3 = [-2.1519775,-1.3439709,-0.7560052,-0.2451210, 0.2451210, 0.7560052, 1.3439709, 2.1519775]
_B3 = [-1.7479742,-1.0499881,-0.5005631, 0.0, 0.5005631, 1.0499881, 1.7479742]


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class TurboQuantBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "float16", "bfloat16", "turboquant",
    ]

    @staticmethod
    def get_name():
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls():
        return TurboQuantImpl

    @staticmethod
    def get_builder_cls():
        return TurboQuantMetadataBuilder


class TurboQuantMetadataBuilder(FlashAttentionMetadataBuilder):
    pass


class TurboQuantImpl(FlashAttentionImpl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pd = _next_pow2(self.head_size)
        self._dc = self._pd // TILE_DIMS
        self._hi_b = None

    def _ensure(self, dev):
        if self._hi_b is not None:
            return
        s = 1.0 / math.sqrt(self._pd)
        self._hi_b = torch.tensor([b*s for b in _B4], device=dev, dtype=torch.float32)
        self._lo_b = torch.tensor([b*s for b in _B3], device=dev, dtype=torch.float32)
        self._hi_c = torch.tensor([c*s for c in _C4], device=dev, dtype=torch.float32)
        self._lo_c = torch.tensor([c*s for c in _C3], device=dev, dtype=torch.float32)

    def _ensure_rotation(self, device):
        """Initialize Hadamard rotation signs (deterministic, per head_dim)."""
        if hasattr(self, '_signs') and self._signs is not None:
            return
        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        signs = torch.sign(torch.randn(self._pd, generator=gen))
        signs[signs == 0] = 1.0
        self._signs = signs.to(device)

    def _fwht(self, x):
        """Fast Walsh-Hadamard Transform along last dim."""
        d = x.shape[-1]
        x = x.clone()
        shape = x.shape
        h = 1
        while h < d:
            x = x.view(*shape[:-1], d // (2 * h), 2, h)
            a = x[..., 0, :].clone()
            b = x[..., 1, :].clone()
            x[..., 0, :] = a + b
            x[..., 1, :] = a - b
            x = x.view(shape)
            h *= 2
        return x * (1.0 / math.sqrt(d))

    def _hadamard_rotate(self, x):
        """Forward: y = FWHT(signs * pad(x))"""
        self._ensure_rotation(x.device)
        d = x.shape[-1]
        if d < self._pd:
            x = F.pad(x, (0, self._pd - d))
        return self._fwht(x * self._signs)

    def _hadamard_inverse(self, y):
        """Inverse: x = unpad(signs * FWHT(y))"""
        x = self._fwht(y) * self._signs
        if self.head_size < self._pd:
            x = x[..., :self.head_size]
        return x

    def _quantize_dequantize(self, x):
        """Simulate 4-bit quantization with Hadamard rotation.

        Fully vectorized — no Python loops. Operates on entire batch at once.
        """
        self._ensure(x.device)
        xf = x.float()

        # L2 normalize + snap norms to fp16
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms = norms.to(torch.float16).float()

        # Hadamard rotate (vectorized FWHT on full padded_dim)
        rotated = self._hadamard_rotate(normalized)

        # 4-bit codebook quantize-dequant (fully vectorized, no chunk loop)
        indices = torch.bucketize(rotated.contiguous(), self._hi_b)
        quantized = self._hi_c[indices]

        # Inverse Hadamard rotate
        reconstructed = self._hadamard_inverse(quantized)

        return (reconstructed * norms).to(x.dtype)

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        num_tokens = slot_mapping.shape[0]
        key_q = self._quantize_dequantize(key[:num_tokens])
        value_q = self._quantize_dequantize(value[:num_tokens])

        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key_q, value_q, key_cache, value_cache, slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale,
        )
