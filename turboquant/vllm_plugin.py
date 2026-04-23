"""TurboQuant vLLM plugin — monkey-patches upstream Triton kernels.

At vLLM startup (via the `vllm.plugins` entry point in `pyproject.toml`),
`register()` is called once. If `TQ_PATCH_DECODE=1` (or any other
HYP-062+ env toggle) is set, `_patch_triton_kernels()` replaces specific
attributes on upstream modules with our optimized variants. Otherwise
the plugin is a no-op.

Toggles (default: all off, so `register()` does nothing):
- `TQ_PATCH_DECODE=1` → swap `triton_turboquant_decode_attention`
  with `turboquant.kernels.decode_stage1.make_patched_launcher()`.

Launch-config env vars (read by the patched launcher on every call):
- `TQ_DECODE_NUM_WARPS`, `TQ_DECODE_NUM_STAGES`, `TQ_DECODE_BLOCK_KV`.

Pre-pivot CUSTOM backend (`vllm_backend_fused.TurboQuantBackend`) is
archived — the registration branch below is kept no-op to avoid any
conflict with the upstream TurboQuant path.
"""

from __future__ import annotations

import os
import sys

_registered = False
_patched = []  # list of (module_name, attr, old_value) for diagnostics


def _patch_triton_kernels() -> None:
    """Monkey-patch upstream Triton kernel entry points."""
    global _patched

    if os.environ.get("TQ_PATCH_DECODE", "").strip() in ("1", "true", "True"):
        try:
            from vllm.v1.attention.ops import triton_turboquant_decode as _mod
            from turboquant.kernels.decode_stage1 import make_patched_launcher
        except ImportError as e:
            print(f"[tq-plugin] TQ_PATCH_DECODE=1 but import failed: {e}",
                  flush=True)
            return

        patched_fn = make_patched_launcher()
        old_fn = _mod.triton_turboquant_decode_attention
        patched_fn.__name__ = old_fn.__name__
        patched_fn.__qualname__ = old_fn.__qualname__

        # 1. Patch the defining module.
        _mod.triton_turboquant_decode_attention = patched_fn
        _patched.append(
            ("vllm.v1.attention.ops.triton_turboquant_decode",
             "triton_turboquant_decode_attention", old_fn)
        )

        # 2. Patch any already-imported modules that bound the symbol locally
        #    (from X import Y pattern). vLLM's turboquant_attn.py imports it;
        #    catch everything under sys.modules defensively.
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is _mod:
                continue
            # Only look inside vllm's turboquant-related modules.
            if "turboquant" not in mod_name:
                continue
            bound = getattr(mod, "triton_turboquant_decode_attention", None)
            if bound is old_fn:
                setattr(mod, "triton_turboquant_decode_attention", patched_fn)
                _patched.append(
                    (mod_name, "triton_turboquant_decode_attention", old_fn)
                )

        nw = os.environ.get("TQ_DECODE_NUM_WARPS", "1")
        ns = os.environ.get("TQ_DECODE_NUM_STAGES", "1")
        bk = os.environ.get("TQ_DECODE_BLOCK_KV", "4")
        print(
            f"[tq-plugin] patched triton_turboquant_decode_attention in "
            f"{len(_patched)} module(s) "
            f"(num_warps={nw}, num_stages={ns}, BLOCK_KV={bk})",
            flush=True,
        )


def register() -> None:
    """vLLM plugin entry point — called once at vLLM startup."""
    global _registered
    if _registered:
        return
    _registered = True

    _patch_triton_kernels()
