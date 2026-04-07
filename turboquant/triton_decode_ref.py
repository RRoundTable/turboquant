"""PyTorch reference implementation for TurboQuant decode attention.

Provides quantization, dequantization, paged KV cache setup, and reference
decode attention — all in pure PyTorch for correctness testing against the
Triton kernel.

Data layout matches paged_kv_turbo_t (HND, uniform 4-bit):
  k_quant/v_quant: [max_pages, num_kv_heads, page_size, quant_bytes_per_token]
  k_norms/v_norms: [max_pages, num_kv_heads, page_size, dim_chunks]
  Nibble packing: high nibble = even dim index, low nibble = odd dim index
"""

import math
import torch

# Lloyd-Max 4-bit codebook centroids for N(0,1), matching kCodebook4bit in CUDA.
CODEBOOK_4BIT = [
    -2.7326368, -2.0693470, -1.6180345, -1.2562491,
    -0.9423684, -0.6567591, -0.3880823, -0.1284154,
     0.1284154,  0.3880823,  0.6567591,  0.9423684,
     1.2562491,  1.6180345,  2.0693470,  2.7326368,
]

# Boundaries (midpoints between centroids) for bucketize quantization.
BOUNDARIES_4BIT = [
    (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0 for i in range(15)
]


def _padded_dim(head_dim: int) -> int:
    """Next multiple of 64 >= head_dim."""
    return ((head_dim + 63) // 64) * 64


@torch.no_grad()
def quantize_kv(kv_fp16: torch.Tensor, padded_dim: int):
    """Quantize fp16 KV vectors to 4-bit packed format + norms.

    Args:
        kv_fp16: [seq_len, head_dim] float16 or float32
        padded_dim: dimension padded to multiple of 64

    Returns:
        packed: [seq_len, padded_dim // 2] uint8  (nibble-packed codebook indices)
        norms:  [seq_len, dim_chunks] float16  (L2 norm per chunk)
    """
    device = kv_fp16.device
    seq_len, head_dim = kv_fp16.shape
    dim_chunks = padded_dim // 64

    codebook_scale = 1.0 / math.sqrt(padded_dim)
    boundaries = torch.tensor(
        [b * codebook_scale for b in BOUNDARIES_4BIT],
        dtype=torch.float32, device=device,
    )

    # Pad to padded_dim if needed
    if head_dim < padded_dim:
        kv_padded = torch.zeros(seq_len, padded_dim, dtype=torch.float32, device=device)
        kv_padded[:, :head_dim] = kv_fp16.float()
    else:
        kv_padded = kv_fp16.float()

    # Reshape into chunks: [seq_len, dim_chunks, 64]
    kv_chunks = kv_padded.view(seq_len, dim_chunks, 64)

    # Compute per-chunk L2 norm
    norms = kv_chunks.norm(dim=-1).clamp(min=1e-8)  # [seq_len, dim_chunks]

    # Normalize each chunk
    kv_normalized = kv_chunks / norms.unsqueeze(-1)  # [seq_len, dim_chunks, 64]

    # Quantize: find nearest centroid index
    kv_flat = kv_normalized.reshape(-1)
    indices_flat = torch.bucketize(kv_flat, boundaries).to(torch.uint8)
    indices = indices_flat.view(seq_len, dim_chunks, 64)

    # Nibble-pack: pair (even, odd) dims into one byte
    # High nibble = even dim, low nibble = odd dim
    even = indices[:, :, 0::2]  # [seq_len, dim_chunks, 32]
    odd = indices[:, :, 1::2]   # [seq_len, dim_chunks, 32]
    packed = (even << 4) | odd  # [seq_len, dim_chunks, 32]
    packed = packed.view(seq_len, dim_chunks * 32)  # [seq_len, quant_bytes_per_token]

    return packed, norms.to(torch.float16)


@torch.no_grad()
def dequantize_kv(packed: torch.Tensor, norms: torch.Tensor,
                  head_dim: int, padded_dim: int) -> torch.Tensor:
    """Dequantize packed 4-bit KV data back to fp16.

    Args:
        packed: [seq_len, quant_bytes_per_token] uint8
        norms:  [seq_len, dim_chunks] float16
        head_dim: original head dimension
        padded_dim: padded dimension (multiple of 64)

    Returns:
        kv_fp16: [seq_len, head_dim] float32
    """
    device = packed.device
    seq_len = packed.shape[0]
    dim_chunks = padded_dim // 64
    codebook_scale = 1.0 / math.sqrt(padded_dim)

    codebook = torch.tensor(CODEBOOK_4BIT, dtype=torch.float32, device=device)

    # Reshape packed bytes: [seq_len, dim_chunks, 32]
    packed_chunks = packed.view(seq_len, dim_chunks, 32)

    # Unpack nibbles
    hi_nibble = (packed_chunks >> 4) & 0x0F  # even dim indices
    lo_nibble = packed_chunks & 0x0F          # odd dim indices

    # Interleave back to [seq_len, dim_chunks, 64]
    indices = torch.zeros(seq_len, dim_chunks, 64, dtype=torch.long, device=device)
    indices[:, :, 0::2] = hi_nibble.long()
    indices[:, :, 1::2] = lo_nibble.long()

    # Lookup codebook values
    values = codebook[indices]  # [seq_len, dim_chunks, 64]

    # Scale by codebook_scale * norm
    norms_f32 = norms.float()  # [seq_len, dim_chunks]
    scale = codebook_scale * norms_f32  # [seq_len, dim_chunks]
    values = values * scale.unsqueeze(-1)  # broadcast over 64 dims

    # Flatten and trim to head_dim
    values_flat = values.view(seq_len, padded_dim)
    return values_flat[:, :head_dim]


@torch.no_grad()
def setup_paged_kv(batch: int, num_kv_heads: int, head_dim: int,
                   seq_len: int, page_size: int, device: str = "cuda"):
    """Create paged KV cache with random data in TurboQuant format.

    Returns a dict with all tensors needed by both the Triton kernel and
    the reference implementation.

    Layout (HND):
        k_quant: [max_pages, num_kv_heads, page_size, dim_chunks * 32] uint8
        k_norms: [max_pages, num_kv_heads, page_size, dim_chunks] float16
        (same for v_quant, v_norms)

    Page table:
        kv_indices: [total_pages] int32
        kv_indptr:  [batch + 1] int32
        last_page_len: [batch] int32
    """
    padded_dim = _padded_dim(head_dim)
    dim_chunks = padded_dim // 64
    quant_bytes_per_token = dim_chunks * 32

    pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch * pages_per_seq
    last_len = seq_len % page_size
    if last_len == 0:
        last_len = page_size

    torch.manual_seed(42)

    # Generate random K and V in fp32, then quantize
    # Per-sequence K, V: [seq_len, head_dim]
    k_raw_all = []
    v_raw_all = []
    for _ in range(batch * num_kv_heads):
        k_raw_all.append(torch.randn(seq_len, head_dim, device=device))
        v_raw_all.append(torch.randn(seq_len, head_dim, device=device))

    # Allocate paged storage
    k_quant = torch.zeros(total_pages, num_kv_heads, page_size, quant_bytes_per_token,
                          dtype=torch.uint8, device=device)
    v_quant = torch.zeros(total_pages, num_kv_heads, page_size, quant_bytes_per_token,
                          dtype=torch.uint8, device=device)
    k_norms = torch.zeros(total_pages, num_kv_heads, page_size, dim_chunks,
                          dtype=torch.float16, device=device)
    v_norms = torch.zeros(total_pages, num_kv_heads, page_size, dim_chunks,
                          dtype=torch.float16, device=device)

    # Page table
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    for b in range(batch):
        kv_indptr[b + 1] = kv_indptr[b] + pages_per_seq
    last_page_len = torch.full((batch,), last_len, dtype=torch.int32, device=device)

    # Fill pages with quantized data
    # Also collect dequantized K, V for reference
    k_deq_all = []
    v_deq_all = []

    for b in range(batch):
        page_base = b * pages_per_seq
        for h in range(num_kv_heads):
            raw_idx = b * num_kv_heads + h
            k_raw = k_raw_all[raw_idx]
            v_raw = v_raw_all[raw_idx]

            # Quantize
            k_packed, k_ns = quantize_kv(k_raw, padded_dim)
            v_packed, v_ns = quantize_kv(v_raw, padded_dim)

            # Dequantize for reference
            k_deq = dequantize_kv(k_packed, k_ns, head_dim, padded_dim)
            v_deq = dequantize_kv(v_packed, v_ns, head_dim, padded_dim)
            k_deq_all.append(k_deq)
            v_deq_all.append(v_deq)

            # Fill paged buffers
            for t in range(seq_len):
                page_idx = page_base + t // page_size
                entry_idx = t % page_size
                k_quant[page_idx, h, entry_idx, :] = k_packed[t]
                v_quant[page_idx, h, entry_idx, :] = v_packed[t]
                k_norms[page_idx, h, entry_idx, :] = k_ns[t]
                v_norms[page_idx, h, entry_idx, :] = v_ns[t]

    # Flatten for kernel consumption (contiguous byte arrays)
    k_quant_flat = k_quant.contiguous().view(-1)
    v_quant_flat = v_quant.contiguous().view(-1)
    k_norms_flat = k_norms.contiguous().view(-1)
    v_norms_flat = v_norms.contiguous().view(-1)

    # Compute strides matching paged_kv_turbo_t
    quant_stride_n = dim_chunks * 32
    quant_stride_h = page_size * quant_stride_n
    quant_stride_page = num_kv_heads * quant_stride_h
    norm_stride_n = dim_chunks
    norm_stride_h = page_size * norm_stride_n
    norm_stride_page = num_kv_heads * norm_stride_h

    return {
        "batch": batch,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "padded_dim": padded_dim,
        "dim_chunks": dim_chunks,
        "seq_len": seq_len,
        "page_size": page_size,
        "pages_per_seq": pages_per_seq,
        "k_quant": k_quant_flat,
        "v_quant": v_quant_flat,
        "k_norms": k_norms_flat,
        "v_norms": v_norms_flat,
        "kv_indices": kv_indices,
        "kv_indptr": kv_indptr,
        "last_page_len": last_page_len,
        "quant_stride_page": quant_stride_page,
        "quant_stride_h": quant_stride_h,
        "quant_stride_n": quant_stride_n,
        "norm_stride_page": norm_stride_page,
        "norm_stride_h": norm_stride_h,
        "norm_stride_n": norm_stride_n,
        # Dequantized tensors for reference attention
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        "k_deq": torch.stack(k_deq_all).view(batch, num_kv_heads, seq_len, head_dim),
        "v_deq": torch.stack(v_deq_all).view(batch, num_kv_heads, seq_len, head_dim),
    }


@torch.no_grad()
def reference_decode(q: torch.Tensor, k_deq: torch.Tensor, v_deq: torch.Tensor,
                     sm_scale: float) -> torch.Tensor:
    """Reference decode attention on dequantized KV.

    Args:
        q:     [batch, num_qo_heads, head_dim] float32
        k_deq: [batch, num_kv_heads, seq_len, head_dim] float32
        v_deq: [batch, num_kv_heads, seq_len, head_dim] float32
        sm_scale: softmax scaling factor (typically 1/sqrt(head_dim))

    Returns:
        output: [batch, num_qo_heads, head_dim] float32
    """
    batch, num_qo_heads, head_dim = q.shape
    _, num_kv_heads, seq_len, _ = k_deq.shape
    gqa_ratio = num_qo_heads // num_kv_heads

    output = torch.zeros_like(q)

    for b in range(batch):
        for qo_h in range(num_qo_heads):
            kv_h = qo_h // gqa_ratio
            q_vec = q[b, qo_h]  # [head_dim]
            k_mat = k_deq[b, kv_h]  # [seq_len, head_dim]
            v_mat = v_deq[b, kv_h]  # [seq_len, head_dim]

            # QK scores
            scores = (k_mat @ q_vec) * sm_scale  # [seq_len]

            # Softmax
            weights = torch.softmax(scores, dim=0)  # [seq_len]

            # Weighted sum of V
            output[b, qo_h] = weights @ v_mat  # [head_dim]

    return output
