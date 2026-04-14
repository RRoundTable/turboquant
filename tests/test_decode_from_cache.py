"""HYP-029: byte-equivalence test for `decode_v4_from_cache`.

Asserts that reading quant+norms directly from the aligned interleaved
cache layout produces output bit-identical to the old path that slices
the cache into tight-packed fp16/uint8 buffers via `.contiguous()`.
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

    torch.manual_seed(0)
    num_qo_heads = num_kv_heads * bdy
    padded_dim = _next_pow2(head_dim)
    dim_chunks = padded_dim // TILE_DIMS
    qbytes = dim_chunks * QUANT_BYTES_PER_CHUNK
    nbytes = dim_chunks * 2
    # cp_async.ca 128-bit loads require 16-byte aligned src.
    bytes_per_head = (qbytes + nbytes + 15) & ~15

    block_size = 16
    max_pages_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * max_pages_per_seq

    cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, bytes_per_head),
        dtype=torch.uint8, device=DEVICE,
    )
    cache[..., :qbytes] = torch.randint(
        0, 256, cache[..., :qbytes].shape, dtype=torch.uint8, device=DEVICE,
    )
    norms_fp16 = (torch.rand(2, num_blocks, block_size, num_kv_heads, dim_chunks,
                             dtype=torch.float16, device=DEVICE) * 0.9 + 0.1)
    cache[..., qbytes:qbytes + nbytes] = norms_fp16.view(torch.uint8).view(
        2, num_blocks, block_size, num_kv_heads, nbytes,
    )

    k_q = cache[0][..., :qbytes].contiguous().view(-1)
    v_q = cache[1][..., :qbytes].contiguous().view(-1)
    k_n = cache[0][..., qbytes:qbytes + nbytes].contiguous().view(torch.float16).view(-1)
    v_n = cache[1][..., qbytes:qbytes + nbytes].contiguous().view(torch.float16).view(-1)

    indptr = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * max_pages_per_seq
    indices = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE)
    num_pages = (seq_len + block_size - 1) // block_size
    last_page_len = torch.full(
        (batch,), seq_len - (num_pages - 1) * block_size,
        dtype=torch.int32, device=DEVICE,
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=DEVICE)

    q = torch.randn(batch, num_qo_heads, padded_dim, dtype=torch.float16, device=DEVICE) * 0.1
    signs = torch.sign(torch.randn(padded_dim, device=DEVICE)).to(torch.float32)
    sm_scale = 1.0 / math.sqrt(head_dim)

    out_old = torch.ops.turboquant.decode_v4(
        q, k_q, v_q, k_n, v_n,
        indices, indptr, last_page_len,
        num_kv_heads, block_size,
        head_dim, padded_dim, sm_scale,
        signs, 0, True,
    )
    out_new = torch.ops.turboquant.decode_v4_from_cache(
        q, cache,
        indices, indptr, last_page_len, seq_lens,
        num_kv_heads, block_size,
        head_dim, padded_dim, sm_scale,
        signs, qbytes, nbytes,
    )
    assert torch.equal(out_old, out_new)
