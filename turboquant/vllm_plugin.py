"""TurboQuant vLLM plugin — registers attention backend via entry_points.

Usage: LLM(..., attention_backend="CUSTOM", kv_cache_dtype="fp8")
  fp8 gives uint8 cache (2× memory savings vs fp16).
  Full 3.76× savings requires vLLM upstream PR (custom page_size_bytes).
"""

_registered = False


def register():
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
