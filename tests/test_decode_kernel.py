"""Phase 3c: Test the Python wrapper for fused TurboQuant decode kernel.

Verifies end-to-end: quantize KV in Python → call CUDA kernel → compare output.
"""

import math
import torch
import pytest

from turboquant.decode_kernel import TurboQuantDecoder
from turboquant.write_kernel import TurboQuantWriter
from turboquant.tile import (
    TILE_TOKENS, TILE_DIMS, TILE_BYTES,
    HI_DIMS, LO_DIMS, HI_BYTES_PER_ROW, LO_BYTES_PER_ROW,
    pack_4bit, pack_3bit,
)
from turboquant.codebook import _CENTROIDS_N01, _BOUNDARIES_N01

DEVICE = "cuda"


def quantize_kv_for_paged_cache(
    kv: torch.Tensor, page_size: int = 16, head_dim: int = 64, padded_dim: int = 64
):
    """Quantize KV vectors into paged cache format matching paged_kv_turbo_t.

    Args:
        kv: [seq_len, head_dim] float
    Returns:
        quant_data: [total_bytes] uint8 — packed quantized data
        norms_data: [total_norm_entries] fp16
        num_pages: int
    """
    seq_len = kv.shape[0]
    dim_chunks = padded_dim // TILE_DIMS
    quant_bytes_per_chunk = HI_BYTES_PER_ROW + LO_BYTES_PER_ROW  # 28
    quant_bytes_per_token = dim_chunks * quant_bytes_per_chunk
    num_pages = (seq_len + page_size - 1) // page_size

    scale = 1.0 / math.sqrt(padded_dim)
    hi_boundaries = torch.tensor([b * scale for b in _BOUNDARIES_N01[4]], device=kv.device)
    lo_boundaries = torch.tensor([b * scale for b in _BOUNDARIES_N01[3]], device=kv.device)

    # Allocate: [num_pages, 1 head, page_size, quant_bytes_per_token]
    total_tokens = num_pages * page_size
    quant_flat = torch.zeros(total_tokens * quant_bytes_per_token, dtype=torch.uint8, device=kv.device)
    norms_flat = torch.zeros(total_tokens * dim_chunks, dtype=torch.float16, device=kv.device)

    for t in range(seq_len):
        src = kv[t]
        l2 = src.norm()
        norm_fp16 = l2.to(torch.float16)
        norm = norm_fp16.float()
        inv_norm = (1.0 / norm) if norm > 0 else torch.tensor(0.0, device=kv.device)
        normalized = src * inv_norm

        norms_flat[t] = norm_fp16

        # 4-bit hi
        hi_indices = torch.bucketize(normalized[:HI_DIMS], hi_boundaries).to(torch.uint8)
        hi_packed = pack_4bit(hi_indices)

        # 3-bit lo
        lo_indices = torch.bucketize(normalized[HI_DIMS:TILE_DIMS], lo_boundaries).to(torch.uint8)
        lo_packed = pack_3bit(lo_indices)

        offset = t * quant_bytes_per_token
        quant_flat[offset: offset + HI_BYTES_PER_ROW] = hi_packed
        quant_flat[offset + HI_BYTES_PER_ROW: offset + quant_bytes_per_chunk] = lo_packed

    return quant_flat, norms_flat, num_pages


def reference_attention(q, k, v, sm_scale):
    """CPU-style reference attention."""
    scores = torch.matmul(q, k.t()) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class TestTurboQuantDecoder:
    def test_basic(self):
        """Basic decode: 16 tokens, head_dim=64."""
        head_dim = 64
        seq_len = 16
        page_size = 16

        torch.manual_seed(42)
        k = torch.randn(seq_len, head_dim, device=DEVICE)
        v = torch.randn(seq_len, head_dim, device=DEVICE)
        q = torch.randn(1, 1, head_dim, device=DEVICE, dtype=torch.float16)

        # Quantize
        k_quant, k_norms, num_pages = quantize_kv_for_paged_cache(k, page_size, head_dim)
        v_quant, v_norms, _ = quantize_kv_for_paged_cache(v, page_size, head_dim)

        # Page table: identity
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
        kv_last_page_len = torch.tensor(
            [seq_len % page_size if seq_len % page_size != 0 else page_size],
            dtype=torch.int32, device=DEVICE,
        )

        # Dequantize reference
        scale = 1.0 / math.sqrt(head_dim)
        hi_centroids = torch.tensor([c * scale for c in _CENTROIDS_N01[4]], device=DEVICE)
        lo_centroids = torch.tensor([c * scale for c in _CENTROIDS_N01[3]], device=DEVICE)

        from turboquant.tile import unpack_4bit, unpack_3bit
        k_deq = torch.zeros(seq_len, head_dim, device=DEVICE)
        v_deq = torch.zeros(seq_len, head_dim, device=DEVICE)
        quant_bytes_per_token = HI_BYTES_PER_ROW + LO_BYTES_PER_ROW
        for t in range(seq_len):
            for kv_idx, (qdata, ndata, deq) in enumerate([
                (k_quant, k_norms, k_deq), (v_quant, v_norms, v_deq)
            ]):
                offset = t * quant_bytes_per_token
                norm = ndata[t].float()
                hi_packed = qdata[offset: offset + HI_BYTES_PER_ROW]
                hi_idx = unpack_4bit(hi_packed, HI_DIMS)
                deq[t, :HI_DIMS] = hi_centroids[hi_idx.long()] * norm
                lo_packed = qdata[offset + HI_BYTES_PER_ROW: offset + quant_bytes_per_token]
                lo_idx = unpack_3bit(lo_packed, LO_DIMS)
                deq[t, HI_DIMS:] = lo_centroids[lo_idx.long()] * norm

        sm_scale = 1.0 / math.sqrt(head_dim)
        ref_output = reference_attention(q.squeeze(0).float(), k_deq, v_deq, sm_scale)

        # Run fused kernel
        decoder = TurboQuantDecoder(head_dim=head_dim, page_size=page_size)
        output = decoder.decode(
            q, k_quant, v_quant, k_norms, v_norms,
            kv_indices, kv_indptr, kv_last_page_len,
            num_qo_heads=1, sm_scale=sm_scale,
        )

        # Compare
        out_float = output.squeeze(0).float()
        cos = torch.nn.functional.cosine_similarity(
            ref_output.flatten(), out_float.flatten(), dim=0
        )
        assert cos > 0.99, f"cosine={cos:.6f}"
        print(f"  cosine={cos:.6f}, PASS")
