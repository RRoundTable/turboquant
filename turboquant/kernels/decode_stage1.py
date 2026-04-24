"""HYP-062 — `_tq_decode_stage1` joint launch retune.

Replaces upstream vLLM v0.20.0's `triton_turboquant_decode_attention`
launcher with a version that reads `(num_warps, num_stages, BLOCK_KV)`
from environment variables instead of hardcoding them. The Triton
`@triton.jit` kernel body is unchanged — we reuse upstream's
`_tq_decode_stage1` directly, so SHA-256 parity is preserved by
construction.

Env vars (defaults match upstream):
- `TQ_DECODE_NUM_WARPS`   int, default 1
- `TQ_DECODE_NUM_STAGES`  int, default 1
- `TQ_DECODE_BLOCK_KV`    int, default 4

The monkey-patch is activated by `TQ_PATCH_DECODE=1`; otherwise the
plugin is a no-op and vLLM runs upstream defaults.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def make_patched_launcher():
    """Construct the patched `triton_turboquant_decode_attention` launcher.

    Imports upstream helpers at call time (lazy) so this module is importable
    without vLLM installed — the plugin's `register()` gates the import.
    """
    from vllm.v1.attention.ops import triton_turboquant_decode as _upstream

    # Bind locals for speed.
    _tq_decode_stage1 = _upstream._tq_decode_stage1
    _get_layout = _upstream._get_layout
    _use_fp8_e4b15 = _upstream._use_fp8_e4b15
    _fwd_kernel_stage2 = _upstream._fwd_kernel_stage2

    def patched_launcher(
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        scale: float,
        mse_bits: int,
        key_packed_size: int,
        value_quant_bits: int,
        key_fp8: bool = False,
        norm_correction: bool = False,
        PiT: torch.Tensor | None = None,
        mid_o_buf: torch.Tensor | None = None,
        output_buf: torch.Tensor | None = None,
        lse_buf: torch.Tensor | None = None,
        buf_holder: Any = None,
        max_num_kv_splits: int = 32,
    ) -> torch.Tensor:
        num_warps = _env_int("TQ_DECODE_NUM_WARPS", 1)
        num_stages = _env_int("TQ_DECODE_NUM_STAGES", 1)
        block_kv = _env_int("TQ_DECODE_BLOCK_KV", 4)

        B, Hq, D = query.shape

        # HYP-065: adaptive NUM_KV_SPLITS picker (arch-aware).
        # Gated on TQ_ADAPTIVE_SPLITS=1. Requires a host sync on seq_lens,
        # so must run under --enforce-eager (sweep harness does this).
        if os.environ.get("TQ_ADAPTIVE_SPLITS", "").strip() in ("1", "true", "True"):
            from turboquant.dispatch import pick_adaptive_splits
            try:
                max_kv = int(seq_lens.max().item())
            except RuntimeError:
                max_kv = 0
            max_num_kv_splits = pick_adaptive_splits(
                B, Hq, max_kv, upstream_max_splits=max_num_kv_splits,
            )
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        kv_group_size = Hq // Hk
        device = query.device

        cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

        if key_fp8:
            q_rot = query.contiguous()
        else:
            q_float = query.float()
            if PiT is None:
                PiT = Pi.T.contiguous()
            q_rot = (q_float @ PiT).contiguous()

        NUM_KV_SPLITS = max_num_kv_splits

        if (
            mid_o_buf is not None
            and mid_o_buf.shape[0] >= B
            and mid_o_buf.shape[2] >= NUM_KV_SPLITS
        ):
            mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
        else:
            mid_o = torch.empty(
                B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
            )
            if buf_holder is not None:
                buf_holder._tq_mid_o_buf = mid_o

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=block_kv,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        if output_buf is not None and output_buf.shape[0] >= B:
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_output_buf = output
        if lse_buf is not None and lse_buf.shape[0] >= B:
            lse = lse_buf[:B, :Hq]
        else:
            lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._tq_lse_buf = lse

        grid2 = (B, Hq)
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=cfg["BLOCK_D"],
            Lv=D,
            num_warps=4,
            num_stages=2,
        )

        return output.to(query.dtype)

    return patched_launcher
