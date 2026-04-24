"""Arch-aware dispatch helpers for the TurboQuant plugin.

Currently exposes `pick_adaptive_splits()` for HYP-065 (adaptive
`NUM_KV_SPLITS`). Designed so each new SM tier slots in by adding a
single entry to `_SM_TARGETS` once the per-arch sweep finishes.

Usage in `turboquant.kernels.decode_stage1`:

    from turboquant.dispatch import pick_adaptive_splits
    splits = pick_adaptive_splits(B, Hq, max_kv_len)

Env var override: `TQ_SPLIT_FACTOR` overrides the SM-table lookup
(useful for the HYP-065 sweep).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch


# Per-(SM major, SM minor) oversubscription factor.
# Sweep determined: A100 SM80 → 4 (HYP-065, 2026-04-24).
# Other arches stay TBD until hardware lands and the sweep is rerun.
_SM_TARGETS: dict[tuple[int, int], int] = {
    (8, 0): 4,    # A100 — locked by HYP-065 sweep
    # (9, 0): TBD,  # H100/H200 — sweep when access lands
    # (10, 0): TBD, # B200 — sweep when access lands
}

_DEFAULT_FACTOR = 4   # fallback for unknown arches; safe-ish guess


@lru_cache(maxsize=8)
def _device_meta(device_index: int) -> tuple[int, int]:
    """Cached (num_sms, factor) for a device. Re-keyed if device changes."""
    cap = torch.cuda.get_device_capability(device_index)
    factor_env = os.environ.get("TQ_SPLIT_FACTOR", "").strip()
    if factor_env:
        try:
            factor = int(factor_env)
        except ValueError:
            factor = _SM_TARGETS.get(cap, _DEFAULT_FACTOR)
    else:
        factor = _SM_TARGETS.get(cap, _DEFAULT_FACTOR)
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return (num_sms, factor)


def pick_adaptive_splits(
    batch: int,
    hq: int,
    max_kv_len: int,
    *,
    upstream_max_splits: int = 32,
    min_chunk: int = 32,
    device_index: int = 0,
) -> int:
    """Return a NUM_KV_SPLITS value tuned for current arch + workload.

    Formula:
        target_ctas = factor * num_sms                (~ 432 on A100 SM80)
        naive       = ceil(target_ctas / (batch * hq))
        splits      = clamp(naive, 1, upstream_max_splits)
        splits      = min(splits, max_kv_len // min_chunk)   # avoid sub-warp work

    `factor` is read from `TQ_SPLIT_FACTOR` env var if set, else from
    `_SM_TARGETS[get_device_capability()]`.

    Returns the upstream constant (`upstream_max_splits`) when factor==0
    so the sweep can trivially reproduce the unpatched baseline.
    """
    if hq <= 0 or batch <= 0:
        return upstream_max_splits

    num_sms, factor = _device_meta(device_index)

    if factor == 0:
        # Sentinel — caller wants exact upstream behavior, no adaptive override.
        return upstream_max_splits

    target_ctas = factor * num_sms
    naive = math.ceil(target_ctas / (batch * hq))
    splits = max(1, min(naive, upstream_max_splits))

    if max_kv_len > 0:
        splits = min(splits, max(1, max_kv_len // min_chunk))

    return splits
