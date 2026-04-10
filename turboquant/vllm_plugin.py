"""TurboQuant vLLM plugin — auto-registers attention backend via entry_points.

When turboquant is pip-installed, vLLM discovers this plugin at startup
and registers the TurboQuantBackend as CUSTOM attention backend.

Usage:
  LLM(..., attention_backend="CUSTOM", kv_cache_dtype="fp8")

The fp8 cache dtype gives uint8 allocation (2× memory savings vs fp16).
TurboQuant stores 4-bit quantized data in the first 68 bytes of each
128-byte per-head allocation (for head_dim=128). True compression: 1.88×.
"""

_registered = False


def register():
    """Called by vLLM's plugin discovery system (vllm.general_plugins entry point)."""
    global _registered
    if _registered:
        return
    _registered = True

    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "turboquant.vllm_backend_fused.TurboQuantBackend",
        )
    except ImportError:
        pass
