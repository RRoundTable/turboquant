"""HYP-029: byte-equivalence test for `decode_v4_from_cache`.

Verifies that reading quant+norms directly from the interleaved fp8-cache
layout produces output identical to the old path that pre-slices into
tight-packed buffers via Python `.contiguous()` views.

This is the correctness gate for HYP-029 (graph-safe read path).
"""

import math
import torch
import pytest

from turboquant.decode_kernel_v4 import _get_module

TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32  # uniform 4-bit, matches paged_kv_turbo_t
DEVICE = "cuda"


@pytest.fixture(scope="module", autouse=True)
def _load_extension():
    _get_module()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@pytest.mark.parametrize("head_dim,num_kv_heads,bdy", [
    (64, 2, 1),
    (128, 4, 2),
    (128, 8, 4),
])
@pytest.mark.parametrize("batch,seq_len", [(2, 48), (3, 113)])
def test_from_cache_matches_sliced_path(head_dim, num_kv_heads, bdy, batch, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import os
    os.environ["TQ_DEBUG_STRIDES"] = "1"

    torch.manual_seed(0)
    num_qo_heads = num_kv_heads * bdy
    padded_dim = _next_pow2(head_dim)
    dim_chunks = padded_dim // TILE_DIMS
    qbytes = dim_chunks * QUANT_BYTES_PER_CHUNK
    nbytes = dim_chunks * 2

    block_size = 16
    max_pages_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * max_pages_per_seq

    # Interleaved cache (what vLLM allocates for fp8): per-entry layout is
    # [qbytes of quantized data | nbytes of fp16 norms], stride = qbytes+nbytes.
    # Populate: random bytes for quant (valid codebook indices 0-15 packed),
    # small positive fp16 values for norms (random uint8 → fp16 often yields NaN/Inf).
    cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, qbytes + nbytes),
        dtype=torch.uint8, device=DEVICE,
    )
    cache[..., :qbytes] = torch.randint(
        0, 256, cache[..., :qbytes].shape, dtype=torch.uint8, device=DEVICE,
    )
    # Fill norm slots with valid fp16 values in [0.1, 1.0).
    norms_fp16 = (torch.rand(2, num_blocks, block_size, num_kv_heads, dim_chunks,
                             dtype=torch.float16, device=DEVICE) * 0.9 + 0.1)
    cache[..., qbytes:qbytes + nbytes] = norms_fp16.view(torch.uint8).view(
        2, num_blocks, block_size, num_kv_heads, nbytes,
    )

    # Old path: slice + .contiguous() to get tight-packed flat buffers.
    k_q = cache[0][..., :qbytes].contiguous().view(-1)
    v_q = cache[1][..., :qbytes].contiguous().view(-1)
    k_n = cache[0][..., qbytes:qbytes + nbytes].contiguous().view(torch.float16).view(-1)
    v_n = cache[1][..., qbytes:qbytes + nbytes].contiguous().view(torch.float16).view(-1)

    # Page table: strided layout, block 0..max_pages-1 belong to seq 0, etc.
    indptr = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * max_pages_per_seq
    indices = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE)
    num_pages = (seq_len + block_size - 1) // block_size
    last_page_len = torch.full(
        (batch,), seq_len - (num_pages - 1) * block_size,
        dtype=torch.int32, device=DEVICE,
    )

    # Python-side sanity: verify that the tight k_n buffer and the cache's
    # interleaved norm bytes describe the same fp16 values.
    k_n_reshaped = k_n.view(num_blocks, block_size, num_kv_heads, dim_chunks)
    cache_norms_direct = cache[0, ..., qbytes:qbytes + nbytes].contiguous().view(torch.float16)
    assert torch.equal(k_n_reshaped, cache_norms_direct), "norm view mismatch"
    k_q_reshaped = k_q.view(num_blocks, block_size, num_kv_heads, qbytes)
    cache_quant_direct = cache[0, ..., :qbytes]
    assert torch.equal(k_q_reshaped, cache_quant_direct), "quant view mismatch"

    q = torch.randn(batch, num_qo_heads, padded_dim, dtype=torch.float16, device=DEVICE) * 0.1
    signs = torch.sign(torch.randn(padded_dim, device=DEVICE)).to(torch.float32)
    sm_scale = 1.0 / math.sqrt(head_dim)

    out_old = torch.ops.turboquant.decode_v4(
        q, k_q, v_q, k_n, v_n,
        indices, indptr, last_page_len,
        num_kv_heads, block_size,
        head_dim, padded_dim, sm_scale,
        signs, 0, True,  # tight-packed after .contiguous(), NHD layout
    )
    out_new = torch.ops.turboquant.decode_v4_from_cache(
        q, cache,
        indices, indptr, last_page_len,
        num_kv_heads, block_size,
        head_dim, padded_dim, sm_scale,
        signs, qbytes, nbytes,
    )

    # --- Isolation probes: pass partial strided inputs through decode_v4 ---
    # Probe A: strided quant directly from cache[0], tight norms (tests quant stride).
    cache_k_flat = cache[0].reshape(-1)  # [num_blocks*block_size*num_heads*ebs] uint8
    cache_v_flat = cache[1].reshape(-1)
    out_strided_quant = torch.ops.turboquant.decode_v4(
        q, cache_k_flat, cache_v_flat, k_n, v_n,
        indices, indptr, last_page_len,
        num_kv_heads, block_size,
        head_dim, padded_dim, sm_scale,
        signs, qbytes + nbytes, True,  # strided quant ebs, tight norm
    )
    # Probe B: tight quant, norms read from interleaved via explicit offset ptr.
    # Since decode_v4 takes k_norms/v_norms as Tensor, we need a Tensor view
    # starting at cache[0][:, :, :, qbytes:] treated as fp16. That's what
    # decode_v4_from_cache does internally — probe skipped (same path).

    if not torch.equal(out_old, out_new):
        diff = (out_old.float() - out_new.float()).abs()
        diff_sq = (out_old.float() - out_strided_quant.float()).abs()
        lines = [
            f"=== DIAG hd={head_dim} kv={num_kv_heads} bdy={bdy} batch={batch} seq={seq_len} ===",
            f"max_abs_diff(old,new)={diff.max().item():.6f} mean={diff.mean().item():.6f} (expected: 0)",
            f"max_abs_diff(old,strided_quant)={diff_sq.max().item():.6f} mean={diff_sq.mean().item():.6f} "
            f"(nonzero={(diff_sq > 0).sum().item()}/{diff_sq.numel()}) — tests QUANT stride only",
            f"nonzero(old,new)={(diff > 0).sum().item()}/{diff.numel()}",
            f"out_old[0,0,:8]={out_old[0,0,:8].tolist()}",
            f"out_new[0,0,:8]={out_new[0,0,:8].tolist()}",
            f"out_strided_quant[0,0,:8]={out_strided_quant[0,0,:8].tolist()}",
        ]
        text = "\n".join(lines)
        with open("/tmp/tq_hyp029_diag.txt", "w") as f:
            f.write(text)
        assert False, f"mismatch: max_abs_diff={diff.max().item()} shape={out_old.shape}\n{text}"
