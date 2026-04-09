"""TurboQuant vLLM plugin — auto-registers attention backend via entry_points.

When turboquant is pip-installed, vLLM discovers this plugin at startup
and registers the TurboQuantBackend for --kv-cache-dtype turboquant.
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
        from turboquant.vllm_backend_fused import TurboQuantBackend

        register_backend(AttentionBackendEnum.CUSTOM, TurboQuantBackend)
    except ImportError:
        # vLLM not installed or incompatible version — skip silently
        pass
