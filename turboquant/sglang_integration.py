"""TurboQuant integration for SGLang.

Provides monkey-patching approach to apply TurboQuant KV cache
quantization to SGLang's attention layers.

Usage:
    import sglang as sgl
    from turboquant.sglang_integration import apply_turboquant_to_sglang

    engine = sgl.Engine("meta-llama/Llama-3.1-8B-Instruct")
    apply_turboquant_to_sglang(engine, bit_width=3)
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Optional

import torch
import torch.nn as nn

from .quantizer import TurboQuantMSE

logger = logging.getLogger(__name__)


def apply_turboquant_to_sglang(
    engine,
    bit_width: int = 3,
    head_dim: Optional[int] = None,
    seed: int = 42,
):
    """Apply TurboQuant KV cache quantization to an SGLang engine.

    Args:
        engine: SGLang Engine or Runtime instance
        bit_width: bits per coordinate (1-4, default 3)
        head_dim: dimension per attention head (auto-detected if None)
        seed: random seed
    """
    # Navigate SGLang's internal structure to find the model
    model = _extract_model(engine)
    if model is None:
        logger.warning("Could not find model in SGLang engine. "
                       "TurboQuant not applied.")
        return

    if head_dim is None:
        head_dim = _infer_head_dim(model)

    quantizer = TurboQuantMSE(head_dim, bit_width, seed=seed)
    _patch_attention_layers(model, quantizer)

    logger.info(
        f"TurboQuant applied to SGLang: {bit_width}-bit KV cache "
        f"(head_dim={head_dim}, ~{16.0/bit_width:.1f}x compression)"
    )


def _extract_model(engine) -> Optional[nn.Module]:
    """Try to extract the nn.Module from various SGLang objects."""
    # SGLang Engine → Runtime → ModelRunner → Model
    candidates = [engine]
    for attr_chain in [
        ["runtime"],
        ["runtime", "model_runner"],
        ["runtime", "model_runner", "model"],
        ["model_runner"],
        ["model_runner", "model"],
        ["model"],
        ["tp_worker", "model_runner", "model"],
    ]:
        obj = engine
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.Module):
            return obj
    return None


def _infer_head_dim(model: nn.Module) -> int:
    """Infer head_dim from model config or parameters."""
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "head_dim"):
            return config.head_dim
        hidden = getattr(config, "hidden_size", None)
        n_heads = getattr(config, "num_attention_heads", None)
        if hidden and n_heads:
            return hidden // n_heads

    for name, param in model.named_parameters():
        if "k_proj" in name and "weight" in name:
            return param.shape[0] // max(1, param.shape[0] // 128)

    return 128


def _patch_attention_layers(model: nn.Module, quantizer: TurboQuantMSE):
    """Patch attention layers in the model."""

    for name, module in model.named_modules():
        # SGLang attention layers typically have RadixAttention or similar
        cls_name = type(module).__name__.lower()

        # Check for standard attention patterns
        has_kv_proj = hasattr(module, "k_proj") and hasattr(module, "v_proj")
        is_attention = any(kw in cls_name for kw in ("attention", "attn", "radix"))

        if not (has_kv_proj or is_attention):
            continue

        # Skip if already patched
        if getattr(module, "_turboquant_patched", False):
            continue

        # Try to patch the forward method
        if has_kv_proj:
            _patch_kv_proj_layer(module, quantizer, name)
        elif is_attention and hasattr(module, "forward"):
            _patch_attention_forward(module, quantizer, name)


def _patch_kv_proj_layer(module: nn.Module, quantizer: TurboQuantMSE, name: str):
    """Patch a layer that has k_proj and v_proj."""
    original_forward = module.forward

    @wraps(original_forward)
    def patched_forward(*args, _orig=original_forward, _q=quantizer, **kwargs):
        result = _orig(*args, **kwargs)

        if isinstance(result, tuple) and len(result) >= 2:
            kv = result[-1]
            if isinstance(kv, tuple) and len(kv) == 2:
                k, v = kv
                if isinstance(k, torch.Tensor) and k.dim() >= 3:
                    _q.to(k.device)
                    k_hat = _q.dequantize(*_q.quantize(k))
                    v_hat = _q.dequantize(*_q.quantize(v))
                    result = (*result[:-1], (k_hat, v_hat))
        return result

    module.forward = patched_forward
    module._turboquant_patched = True
    logger.debug(f"Patched SGLang attention layer: {name}")


def _patch_attention_forward(module: nn.Module, quantizer: TurboQuantMSE, name: str):
    """Patch a generic attention module's forward."""
    original_forward = module.forward

    @wraps(original_forward)
    def patched_forward(*args, _orig=original_forward, _q=quantizer, **kwargs):
        # Intercept key_states and value_states if they appear in kwargs
        for kw in ("key", "key_states"):
            if kw in kwargs and isinstance(kwargs[kw], torch.Tensor):
                _q.to(kwargs[kw].device)
                k = kwargs[kw]
                kwargs[kw] = _q.dequantize(*_q.quantize(k))

        for kw in ("value", "value_states"):
            if kw in kwargs and isinstance(kwargs[kw], torch.Tensor):
                _q.to(kwargs[kw].device)
                v = kwargs[kw]
                kwargs[kw] = _q.dequantize(*_q.quantize(v))

        return _orig(*args, **kwargs)

    module.forward = patched_forward
    module._turboquant_patched = True
    logger.debug(f"Patched SGLang attention forward: {name}")


class TurboQuantSGLangConfig:
    """Configuration helper for SGLang + TurboQuant."""

    def __init__(
        self,
        bit_width: int = 3,
        num_outlier_channels: int = 32,
        mode: str = "mse",
    ):
        self.bit_width = bit_width
        self.num_outlier_channels = num_outlier_channels
        self.mode = mode

    @property
    def effective_bit_width(self) -> float:
        if self.num_outlier_channels == 0:
            return float(self.bit_width)
        n_o = self.num_outlier_channels
        return (n_o * (self.bit_width + 1) + (128 - n_o) * self.bit_width) / 128

    @property
    def compression_ratio(self) -> float:
        return 16.0 / self.effective_bit_width
