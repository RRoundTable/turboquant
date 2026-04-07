"""Triton port of TurboQuant v4 decode attention kernel.

Fused decode: loads packed 4-bit KV from paged cache, dequantizes via codebook
lookup, computes QK with online softmax, and accumulates V — all in one kernel.

Grid: (batch * num_kv_heads,) — one program per (batch, kv_head).
Each program handles ALL seq tokens for its KV head, computing GQA_RATIO
QO heads that share that KV head.

Data layout matches paged_kv_turbo_t (HND, uniform 4-bit).
"""

import math
import torch
import triton
import triton.language as tl

# Lloyd-Max 4-bit codebook centroids for N(0,1).
CODEBOOK_4BIT = [
    -2.7326368, -2.0693470, -1.6180345, -1.2562491,
    -0.9423684, -0.6567591, -0.3880823, -0.1284154,
     0.1284154,  0.3880823,  0.6567591,  0.9423684,
     1.2562491,  1.6180345,  2.0693470,  2.7326368,
]


def _get_autotune_configs():
    """Autotune configurations for BLOCK_SEQ and num_warps."""
    configs = []
    for block_seq in [16, 32, 64, 128]:
        for num_warps in [2, 4, 8]:
            configs.append(
                triton.Config({"BLOCK_SEQ": block_seq}, num_warps=num_warps)
            )
    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["seq_len", "head_dim"])
@triton.jit
def _turboquant_decode_kernel(
    # Q tensor: [batch, num_qo_heads, head_dim]
    Q,
    # Output: [batch, num_qo_heads, head_dim]
    O,
    # Packed KV quant data (flat uint8)
    K_quant, V_quant,
    # KV norms (flat float16)
    K_norms, V_norms,
    # Codebook table (16 entries, float32)
    Codebook,
    # Page table
    kv_indices,   # [total_pages] int32
    kv_indptr,    # [batch + 1] int32
    last_page_len,  # [batch] int32
    # Dimensions
    num_qo_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_dim: tl.constexpr,
    seq_len,
    page_size,
    # Strides for paged layout
    quant_stride_page,
    quant_stride_h,
    quant_stride_n,
    norm_stride_page,
    norm_stride_h,
    norm_stride_n,
    # Attention scale
    sm_scale,
    # GQA ratio
    GQA_RATIO: tl.constexpr,
    # Tile size
    BLOCK_SEQ: tl.constexpr,
):
    """TurboQuant decode attention kernel.

    Each program handles one (batch, kv_head) pair and computes GQA_RATIO
    QO head outputs by iterating over all seq tokens.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head_idx = pid % num_kv_heads

    # Sequence length for this batch element
    indptr_start = tl.load(kv_indptr + batch_idx)
    indptr_end = tl.load(kv_indptr + batch_idx + 1)
    last_len = tl.load(last_page_len + batch_idx)
    total_pages = indptr_end - indptr_start
    actual_seq_len = (total_pages - 1) * page_size + last_len

    dim_chunks: tl.constexpr = padded_dim // 64
    codebook_scale = 1.0 / tl.sqrt(float(padded_dim))

    # Process each QO head that maps to this KV head
    for qo_offset in range(GQA_RATIO):
        qo_head_idx = kv_head_idx * GQA_RATIO + qo_offset

        # Load Q vector: [head_dim]
        q_base = batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim
        q_offsets = tl.arange(0, head_dim)
        q_vec = tl.load(Q + q_base + q_offsets).to(tl.float32)

        # Online softmax state
        m_val = float("-inf")
        d_val = 0.0
        o_acc = tl.zeros([head_dim], dtype=tl.float32)

        # Iterate over sequence in tiles of BLOCK_SEQ
        for tile_start in range(0, actual_seq_len, BLOCK_SEQ):
            seq_offsets = tl.arange(0, BLOCK_SEQ)
            token_ids = tile_start + seq_offsets
            seq_mask = token_ids < actual_seq_len

            # Paged addressing: compute (page_iter, entry_idx) for each token
            packed_pos = indptr_start * page_size + token_ids
            page_iter = packed_pos // page_size
            entry_idx = packed_pos % page_size

            # Load physical page indices
            page_idx = tl.load(kv_indices + page_iter, mask=seq_mask, other=0)

            # ── Phase 1: K dequant + QK dot product ──
            qk = tl.zeros([BLOCK_SEQ], dtype=tl.float32)

            for chunk in range(dim_chunks):
                # Load per-token norm for this chunk
                n_off = (page_idx * norm_stride_page
                         + kv_head_idx * norm_stride_h
                         + entry_idx * norm_stride_n
                         + chunk)
                k_norm = tl.load(K_norms + n_off, mask=seq_mask, other=0.0).to(tl.float32)
                ns = codebook_scale * k_norm  # [BLOCK_SEQ]

                # Quant base offset for this chunk
                qb = (page_idx * quant_stride_page
                      + kv_head_idx * quant_stride_h
                      + entry_idx * quant_stride_n
                      + chunk * 32)

                # Process 32 packed bytes -> 64 dequantized dims
                for bi in range(32):
                    packed = tl.load(K_quant + qb + bi, mask=seq_mask, other=0)
                    packed_i32 = packed.to(tl.int32)
                    hi_idx = (packed_i32 >> 4) & 0x0F
                    lo_idx = packed_i32 & 0x0F

                    # Codebook gather from global memory
                    k_hi = tl.load(Codebook + hi_idx) * ns
                    k_lo = tl.load(Codebook + lo_idx) * ns

                    # Dot product with Q
                    dim_hi = chunk * 64 + bi * 2
                    dim_lo = dim_hi + 1
                    if dim_hi < head_dim:
                        q_hi = tl.load(Q + q_base + dim_hi).to(tl.float32)
                        qk += q_hi * k_hi
                    if dim_lo < head_dim:
                        q_lo = tl.load(Q + q_base + dim_lo).to(tl.float32)
                        qk += q_lo * k_lo

            # Scale and mask
            qk = qk * sm_scale
            qk = tl.where(seq_mask, qk, float("-inf"))

            # Online softmax update
            tile_max = tl.max(qk, axis=0)
            m_new = tl.maximum(m_val, tile_max)
            # Rescale previous state
            alpha = tl.where(m_val == float("-inf"), 0.0,
                             tl.math.exp2((m_val - m_new) * 1.4426950408889634))
            d_val = d_val * alpha
            o_acc = o_acc * alpha

            # Attention weights for this tile
            p = tl.math.exp2((qk - m_new) * 1.4426950408889634)
            p = tl.where(seq_mask, p, 0.0)
            d_val += tl.sum(p, axis=0)
            m_val = m_new

            # ── Phase 2: V dequant + weighted accumulate ──
            for chunk in range(dim_chunks):
                n_off = (page_idx * norm_stride_page
                         + kv_head_idx * norm_stride_h
                         + entry_idx * norm_stride_n
                         + chunk)
                v_norm = tl.load(V_norms + n_off, mask=seq_mask, other=0.0).to(tl.float32)
                vs = codebook_scale * v_norm

                qb = (page_idx * quant_stride_page
                      + kv_head_idx * quant_stride_h
                      + entry_idx * quant_stride_n
                      + chunk * 32)

                for bi in range(32):
                    packed = tl.load(V_quant + qb + bi, mask=seq_mask, other=0)
                    packed_i32 = packed.to(tl.int32)
                    hi_idx = (packed_i32 >> 4) & 0x0F
                    lo_idx = packed_i32 & 0x0F

                    v_hi = tl.load(Codebook + hi_idx) * vs  # [BLOCK_SEQ]
                    v_lo = tl.load(Codebook + lo_idx) * vs  # [BLOCK_SEQ]

                    dim_hi = chunk * 64 + bi * 2
                    dim_lo = dim_hi + 1

                    # Weighted V: sum(p * v) for each dim
                    if dim_hi < head_dim:
                        contrib_hi = tl.sum(p * v_hi, axis=0)
                        o_acc = tl.where(q_offsets == dim_hi, o_acc + contrib_hi, o_acc)
                    if dim_lo < head_dim:
                        contrib_lo = tl.sum(p * v_lo, axis=0)
                        o_acc = tl.where(q_offsets == dim_lo, o_acc + contrib_lo, o_acc)

        # Normalize
        o_acc = o_acc / d_val

        # Store output
        o_base = batch_idx * num_qo_heads * head_dim + qo_head_idx * head_dim
        tl.store(O + o_base + q_offsets, o_acc.to(tl.float16))


@torch.no_grad()
def turboquant_decode(
    q: torch.Tensor,             # [batch, num_qo_heads, head_dim] fp16
    k_quant: torch.Tensor,       # flat uint8
    v_quant: torch.Tensor,       # flat uint8
    k_norms: torch.Tensor,       # flat fp16
    v_norms: torch.Tensor,       # flat fp16
    kv_indices: torch.Tensor,    # [total_pages] int32
    kv_indptr: torch.Tensor,     # [batch + 1] int32
    last_page_len: torch.Tensor, # [batch] int32
    num_kv_heads: int,
    head_dim: int,
    padded_dim: int,
    page_size: int,
    seq_len: int,
    quant_stride_page: int,
    quant_stride_h: int,
    quant_stride_n: int,
    norm_stride_page: int,
    norm_stride_h: int,
    norm_stride_n: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """Launch TurboQuant decode attention Triton kernel.

    Returns: [batch, num_qo_heads, head_dim] fp16
    """
    batch, num_qo_heads, hd = q.shape
    assert hd == head_dim

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    gqa_ratio = num_qo_heads // num_kv_heads

    # Codebook as a device tensor for gather-based lookup
    codebook = torch.tensor(CODEBOOK_4BIT, dtype=torch.float32, device=q.device)

    output = torch.empty(batch, num_qo_heads, head_dim, dtype=torch.float16, device=q.device)

    grid = (batch * num_kv_heads,)

    _turboquant_decode_kernel[grid](
        q, output,
        k_quant, v_quant,
        k_norms, v_norms,
        codebook,
        kv_indices, kv_indptr, last_page_len,
        num_qo_heads, num_kv_heads,
        head_dim, padded_dim,
        seq_len, page_size,
        quant_stride_page, quant_stride_h, quant_stride_n,
        norm_stride_page, norm_stride_h, norm_stride_n,
        sm_scale,
        GQA_RATIO=gqa_ratio,
    )

    return output
