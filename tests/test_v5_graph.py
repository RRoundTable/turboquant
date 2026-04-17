"""HYP-033: CUDA-graph-safety test for v5 workspace variant.

Captures `torch.ops.turboquant_v5.decode_v5_from_cache_ws` into a
`torch.cuda.CUDAGraph`, replays it multiple times, and asserts:
  1. Capture succeeds (no synchronous ops inside the op).
  2. Replay output matches a direct eager v5 call on the same inputs.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

DEVICE = "cuda"
HEAD_DIM = 128
PADDED_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 32   # bdy=4
BATCH = 1
PAGE_SIZE = 16
TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32


@pytest.fixture(scope="module")
def v5_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from turboquant.decode_kernel_v4 import _CSRC_DIR, _find_flashinfer_include
    from torch.utils.cpp_extension import load
    return load(
        name="turboquant_v5_tc",
        sources=[str(_CSRC_DIR / "src" / "decode_v5_tc_binding.cu")],
        extra_include_paths=[str(_CSRC_DIR / "include"),
                             str(_find_flashinfer_include())],
        extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr",
            "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False)


def _build_inputs(seq_len: int):
    dim_chunks = PADDED_DIM // TILE_DIMS
    qbytes = dim_chunks * QUANT_BYTES_PER_CHUNK
    nbytes = dim_chunks * 2
    ebs = (qbytes + nbytes + 15) & ~15
    max_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = BATCH * max_pages

    torch.manual_seed(seq_len * 7919)
    cache = torch.zeros((2, num_blocks, PAGE_SIZE, NUM_KV_HEADS, ebs),
                        dtype=torch.uint8, device=DEVICE)
    cache[..., :qbytes] = torch.randint(0, 256, cache[..., :qbytes].shape,
                                        dtype=torch.uint8, device=DEVICE)
    norms = (torch.rand(2, num_blocks, PAGE_SIZE, NUM_KV_HEADS, dim_chunks,
                        dtype=torch.float16, device=DEVICE) * 0.9 + 0.1)
    cache[..., qbytes:qbytes + nbytes] = norms.view(torch.uint8).view(
        2, num_blocks, PAGE_SIZE, NUM_KV_HEADS, nbytes)

    indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=DEVICE) * max_pages
    indices = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE)
    num_pages_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = torch.full((BATCH,),
                               seq_len - (num_pages_per_seq - 1) * PAGE_SIZE,
                               dtype=torch.int32, device=DEVICE)
    seq_lens = torch.full((BATCH,), seq_len, dtype=torch.int32, device=DEVICE)
    q = torch.randn(BATCH, NUM_QO_HEADS, PADDED_DIM,
                    dtype=torch.float16, device=DEVICE) * 0.1
    signs = torch.sign(torch.randn(PADDED_DIM, device=DEVICE)).to(torch.float32)

    max_len = max_pages * PAGE_SIZE
    ws = {
        "k_quant": torch.empty((BATCH, NUM_KV_HEADS, max_len, qbytes),
                               dtype=torch.uint8, device=DEVICE),
        "v_quant": torch.empty((BATCH, NUM_KV_HEADS, max_len, qbytes),
                               dtype=torch.uint8, device=DEVICE),
        "k_norms": torch.empty((BATCH, NUM_KV_HEADS, max_len, dim_chunks),
                               dtype=torch.float16, device=DEVICE),
        "v_norms": torch.empty((BATCH, NUM_KV_HEADS, max_len, dim_chunks),
                               dtype=torch.float16, device=DEVICE),
        "o":       torch.empty((BATCH, NUM_QO_HEADS, PADDED_DIM),
                               dtype=torch.float16, device=DEVICE),
    }
    return {
        "cache": cache, "indices": indices, "indptr": indptr,
        "last_page_len": last_page_len, "seq_lens": seq_lens,
        "q": q, "signs": signs, "sm_scale": 1.0 / math.sqrt(HEAD_DIM),
        "qbytes": qbytes, "nbytes": nbytes, "max_len": max_len, "ws": ws,
    }


@pytest.mark.parametrize("seq_len", [64, 256, 1024])
def test_v5_ws_graph_capture_matches_eager(v5_module, seq_len):
    d = _build_inputs(seq_len)

    # Eager reference (call the non-ws op directly).
    ref = v5_module.decode_v5_from_cache(
        d["q"], d["cache"],
        d["indices"], d["indptr"], d["last_page_len"], d["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        d["sm_scale"], d["signs"], d["qbytes"], d["nbytes"],
    )
    torch.cuda.synchronize()

    # Capture the _ws op into a graph. This will throw if any op inside
    # performs a host sync or a non-graph-safe allocation.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            torch.ops.turboquant_v5.decode_v5_from_cache_ws(
                d["q"], d["cache"], d["indices"], d["indptr"],
                d["last_page_len"], d["seq_lens"],
                NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
                d["sm_scale"], d["signs"], d["qbytes"], d["nbytes"],
                d["ws"]["k_quant"], d["ws"]["v_quant"],
                d["ws"]["k_norms"], d["ws"]["v_norms"], d["ws"]["o"],
                d["max_len"],
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.turboquant_v5.decode_v5_from_cache_ws(
            d["q"], d["cache"], d["indices"], d["indptr"],
            d["last_page_len"], d["seq_lens"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
            d["sm_scale"], d["signs"], d["qbytes"], d["nbytes"],
            d["ws"]["k_quant"], d["ws"]["v_quant"],
            d["ws"]["k_norms"], d["ws"]["v_norms"], d["ws"]["o"],
            d["max_len"],
        )

    # Replay multiple times; every replay should write the same output.
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    diff = (ref.float() - d["ws"]["o"].float()).abs()
    max_abs = diff.max().item()
    assert max_abs < 1e-2, (
        f"Graph replay output diverged from eager reference: "
        f"max_abs={max_abs:.6f} at seq_len={seq_len}")
