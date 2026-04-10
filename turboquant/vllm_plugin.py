"""TurboQuant vLLM plugin — registers attention backend via entry_points.

Usage: LLM(..., attention_backend="CUSTOM", kv_cache_dtype="fp8")
  fp8 gives uint8 cache allocation (2× memory savings vs fp16).
  TQ quantizes K,V into 4-bit and stores in the fp8 cache.
  Prefill uses FlashAttention with fresh fp16 K,V.
  Decode uses the fused v4 CUDA kernel.
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
