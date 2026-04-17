"""Runtime decode-op selection by compute capability (Phase 15a).

The v5 production kernel is A100-specific (SM80 ldmatrix/mma.sync). Hopper
(SM90) async wgmma and Blackwell (SM100) FP4 tensor cores would land as v6
and v7 respectively. Rather than hardcode `torch.ops.turboquant_v5.*` in
the hot-path `vllm_backend_fused.py`, this module picks the best available
kernel once per process based on `torch.cuda.get_device_capability()`.

Zero behavior change on today's SM80 hardware — the dispatcher returns
exactly the same v5 op the backend currently hardcodes. The indirection
exists so future kernel versions slot in without touching the backend.

Overrides for testing / A/B:
    TQ_FORCE_KERNEL=v4|v5|v6|v7     force a specific kernel variant
    TQ_DEBUG_KERNEL=1               print selected kernel at dispatch time

The selected op's Python signature is identical across every version:
    op(q, kv_cache, indices, indptr, last_page_len, seq_lens,
       num_kv_heads, page_size, head_dim, padded_dim,
       sm_scale, hadamard_signs, qbytes, nbytes,
       k_quant_ws, v_quant_ws, k_norms_ws, v_norms_ws, o_ws,
       max_len,
       partition_o_ws, partition_lse_ws,
       request_indices, kv_tile_indices, split_indptr, kv_chunk_size,
       num_splits) -> Tensor
"""
import os
from functools import lru_cache

import torch


def _has_op(lib: str, name: str) -> bool:
    """True if torch.ops.<lib>.<name> is registered."""
    try:
        ns = getattr(torch.ops, lib)
        return hasattr(ns, name)
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def _device_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


# --- op factories ----------------------------------------------------------
# Each factory returns the op callable or raises RuntimeError if unavailable.
# The v4 fallback uses a shim that adapts the older signature (no workspace
# args, no partition scratch) to the v5/v6/v7 unified signature.

def _v5_op():
    return torch.ops.turboquant_v5.decode_v5_from_cache_paged_splitkv_ws


def _v6_op():
    # Future: Hopper wgmma + TMA kernel (see ROADMAP Phase 15b).
    return torch.ops.turboquant_v6.decode_v6_wgmma_paged_splitkv_ws


def _v7_op():
    # Future: Blackwell FP4 kernel (see ROADMAP Phase 15c).
    return torch.ops.turboquant_v7.decode_v7_fp4_paged_splitkv_ws


def _v4_fallback_op(*_args, **_kw):
    # v4's `decode_v4_from_cache` takes only the non-workspace args. The
    # dispatcher adapter accepts the v5-style args and forwards only the
    # subset v4 needs, allocating any fp16 output on the fly. This path is
    # only expected on pre-Ampere GPUs (SM70/75) where we can't run WMMA
    # kernels — effectively a correctness safety net, not a perf target.
    raise NotImplementedError(
        "v4 fallback through the unified dispatcher is not yet implemented. "
        "Pre-SM80 hardware should use the older backend path (vllm_backend_quantsim) "
        "or set TQ_FORCE_KERNEL=v5 after verifying your GPU actually supports WMMA."
    )


# --- main dispatcher -------------------------------------------------------

@lru_cache(maxsize=1)
def pick_decode_op():
    """Return (op_callable, kernel_name) for this process.

    Cached — compute capability is fixed per process. Clear via
    `pick_decode_op.cache_clear()` in tests that change env vars.
    """
    forced = os.environ.get("TQ_FORCE_KERNEL", "").strip().lower()
    if forced:
        op, name = _resolve_forced(forced)
        if os.environ.get("TQ_DEBUG_KERNEL"):
            print(f"[TQ dispatch] TQ_FORCE_KERNEL={forced!r} → {name}", flush=True)
        return op, name

    cc = _device_capability()

    if cc >= (10, 0) and _has_op("turboquant_v7", "decode_v7_fp4_paged_splitkv_ws"):
        op, name = _v7_op(), "v7_fp4_blackwell"
    elif cc >= (9, 0) and _has_op("turboquant_v6", "decode_v6_wgmma_paged_splitkv_ws"):
        op, name = _v6_op(), "v6_wgmma_hopper"
    elif cc >= (8, 0) and _has_op("turboquant_v5", "decode_v5_from_cache_paged_splitkv_ws"):
        op, name = _v5_op(), "v5_paged_split_ampere"
    else:
        op, name = _v4_fallback_op, "v4_fallback_generic"

    if os.environ.get("TQ_DEBUG_KERNEL"):
        print(f"[TQ dispatch] cc={cc} → {name}", flush=True)
    return op, name


def _resolve_forced(name: str):
    """Look up a kernel by name, raising if unavailable."""
    table = {
        "v4": (_v4_fallback_op,                     "v4_fallback_generic"),
        "v5": (lambda: _v5_op(),                    "v5_paged_split_ampere"),
        "v6": (lambda: _v6_op(),                    "v6_wgmma_hopper"),
        "v7": (lambda: _v7_op(),                    "v7_fp4_blackwell"),
    }
    if name not in table:
        raise ValueError(
            f"TQ_FORCE_KERNEL={name!r} not recognised. "
            f"Choose from {sorted(table)} (v4 is a fallback stub)."
        )
    factory, display = table[name]
    if name == "v4":
        return factory, display
    # Try to resolve the op; if it's not registered, give a useful error
    # rather than the bare AttributeError torch.ops raises.
    try:
        op = factory()
    except (AttributeError, RuntimeError) as e:
        raise RuntimeError(
            f"TQ_FORCE_KERNEL={name!r} was requested but the op is not "
            f"registered. Is the kernel compiled for this arch? "
            f"(original error: {e})"
        ) from e
    return op, display
