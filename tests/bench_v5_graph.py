"""HYP-033: Benchmark v5 CUDA-graph-safe decode vs baselines.

Runs four variants under CUDA graph capture at a single seq_len:
  - FP16 SDPA           : torch.nn.functional.scaled_dot_product_attention
  - FlashInfer          : single_decode_with_kv_cache (paged-free, batch=1)
  - TurboQuant v4 graph : torch.ops.turboquant.decode_v4_from_cache
  - TurboQuant v5 graph : torch.ops.turboquant_v5.decode_v5_from_cache_ws (HYP-033)

Emits JSON with per-variant p50 + p99 latencies in μs.

Usage:
  uv run python tests/bench_v5_graph.py --seq-len 1024 --out /tmp/seq-1024.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# flashinfer-python 0.6.x requires PyTorch 2.3+ APIs that NGC pytorch:24.01
# (torch 2.2) lacks. Shim the missing surfaces before import so the bench
# can still run flashinfer for comparison.
for _n, _fb in [("uint16", torch.int16), ("uint32", torch.int32),
                ("uint64", torch.int64)]:
    if not hasattr(torch, _n):
        setattr(torch, _n, _fb)
if not hasattr(torch.library, "custom_op"):
    # `custom_op` is a decorator that registers an op. Stub it as a no-op
    # pass-through; flashinfer only needs it importable, not functional for
    # the BatchDecode path we use.
    def _shim_custom_op(*_a, **_kw):
        def deco(fn): return fn
        return deco
    torch.library.custom_op = _shim_custom_op
if not hasattr(torch.library, "register_fake"):
    def _shim_register_fake(*_a, **_kw):
        def deco(fn): return fn
        return deco
    torch.library.register_fake = _shim_register_fake

DEVICE = "cuda"

# Qwen3-8B-ish config; bdy=4 activates the largest v5 WMMA config.
HEAD_DIM = 128
PADDED_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 32
BATCH = int(os.environ.get("TQ_BENCH_BATCH", "1"))
PAGE_SIZE = 16
TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32


def _make_tq_inputs(seq_len: int):
    """Build a packed TurboQuant kv_cache + page table + Q.

    Mirrors the layout produced by `quantize_write_kv_cache` (see
    `tests/test_decode_from_cache.py`). We fill with random bytes/norms — the
    kernel doesn't care about data content for latency; correctness is checked
    separately by comparing v5-ws output against v5-eager on identical inputs.
    """
    dim_chunks = PADDED_DIM // TILE_DIMS
    qbytes = dim_chunks * QUANT_BYTES_PER_CHUNK
    nbytes = dim_chunks * 2
    ebs = (qbytes + nbytes + 15) & ~15

    max_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = BATCH * max_pages

    torch.manual_seed(seq_len)
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
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    return {
        "cache": cache,
        "indices": indices, "indptr": indptr,
        "last_page_len": last_page_len, "seq_lens": seq_lens,
        "q": q, "signs": signs, "sm_scale": sm_scale,
        "qbytes": qbytes, "nbytes": nbytes,
        "max_pages": max_pages,
        "dim_chunks": dim_chunks,
        "max_len": max_pages * PAGE_SIZE,
    }


def _make_fp16_inputs(seq_len: int):
    """Dense fp16 K, V, Q in the shape SDPA expects."""
    torch.manual_seed(seq_len + 1)
    # SDPA expects [batch, heads, seq, hd]
    q = torch.randn(BATCH, NUM_QO_HEADS, 1, HEAD_DIM,
                    dtype=torch.float16, device=DEVICE)
    k = torch.randn(BATCH, NUM_QO_HEADS, seq_len, HEAD_DIM,
                    dtype=torch.float16, device=DEVICE)
    v = torch.randn(BATCH, NUM_QO_HEADS, seq_len, HEAD_DIM,
                    dtype=torch.float16, device=DEVICE)
    return q, k, v


def _alloc_v5_workspace(max_len: int, qbytes: int, dim_chunks: int):
    """Pre-allocate v5 workspace + output buffer (graph-safe)."""
    return {
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


def _alloc_v5_split_workspace(max_len: int, qbytes: int, dim_chunks: int, num_splits: int):
    """Pre-allocate v5 split-KV workspace (HYP-034).

    All split-index scratch is pre-filled here — pure function of
    (BATCH, num_splits, max_len), safe to do outside capture.
    """
    ws = _alloc_v5_workspace(max_len, qbytes, dim_chunks)
    padded_batch = BATCH * num_splits
    arange = torch.arange(padded_batch, dtype=torch.int32, device=DEVICE)
    ws["partition_o"]    = torch.empty((padded_batch, NUM_QO_HEADS, PADDED_DIM),
                                       dtype=torch.float32, device=DEVICE)
    ws["partition_lse"]  = torch.empty((padded_batch, NUM_QO_HEADS),
                                       dtype=torch.float32, device=DEVICE)
    ws["request_indices"] = (arange // num_splits).to(torch.int32)
    ws["kv_tile_indices"] = (arange % num_splits).to(torch.int32)
    ws["split_indptr"]    = (torch.arange(BATCH + 1, dtype=torch.int32, device=DEVICE)
                             * num_splits)
    ws["kv_chunk_size"]   = torch.tensor(
        [(max_len + num_splits - 1) // num_splits],
        dtype=torch.int32, device=DEVICE)
    return ws


def _choose_num_splits(max_len: int) -> int:
    """Same heuristic as TurboQuantFusedImpl._choose_num_splits (HYP-044):
    cap chunk at 256 tokens, no batch divisor. Snaps to the largest divisor
    of max_len ≤ target so chunk_size * num_splits == max_len (no overshoot)."""
    if max_len < 512:
        return 1
    TARGET_CHUNK = 256
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    splits_by_chunk = max(1, max_len // TARGET_CHUNK)
    splits_by_sm = max(1, (4 * sm_count) // NUM_KV_HEADS)
    target = min(splits_by_chunk, splits_by_sm)
    best = 1
    for d in range(2, target + 1):
        if max_len % d == 0:
            best = d
    return best


def _capture_and_time(fn, warmup_replays=50, timed_replays=200):
    """Capture `fn` into a CUDA graph and return (p50_us, p99_us, mean_us).

    `fn` must take no args and touch only tensors that outlive the capture.
    """
    # Warmup on a side stream to settle the caching allocator before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()

    # Per-replay timing via events. Replays in a tight loop for p-stats.
    times_us = []
    for _ in range(timed_replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)  # ms→μs
    times_us.sort()
    p50 = times_us[len(times_us) // 2]
    p99 = times_us[int(len(times_us) * 0.99)]
    mean = sum(times_us) / len(times_us)
    return {"p50_us": p50, "p99_us": p99, "mean_us": mean}


def _time_aggregate(fn, warmup_replays=50, timed_replays=200):
    """Capture once, replay `timed_replays` times in a single event window.

    Same graph, tighter timing (less event overhead). Returns mean μs only.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / timed_replays  # μs


# --- variant builders --------------------------------------------------------

def _bench_fp16_sdpa(seq_len: int):
    q, k, v = _make_fp16_inputs(seq_len)
    sm = 1.0 / math.sqrt(HEAD_DIM)
    out = torch.empty_like(q)
    def run():
        o = F.scaled_dot_product_attention(q, k, v, scale=sm)
        out.copy_(o)
    return _capture_and_time(run)


def _bench_flashinfer(seq_len: int):
    # flashinfer-python 0.6.8 uses torch.uint32 which was added in PyTorch 2.3.
    # On NGC pytorch:24.01 (torch 2.2) we alias it to int32 for compat.
    for name, fallback in [("uint32", torch.int32), ("uint64", torch.int64),
                            ("uint16", torch.int16)]:
        if not hasattr(torch, name):
            setattr(torch, name, fallback)
    try:
        import flashinfer
    except ImportError:
        return {"error": "flashinfer not installed"}
    torch.manual_seed(seq_len + 2)

    if BATCH == 1:
        # single_decode_with_kv_cache: [qo_heads, hd] × [seq, kv_heads, hd]
        q = torch.randn(NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
        k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
        v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
        def run():
            flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD")
        try:
            return _capture_and_time(run)
        except Exception as e:
            return {"error": f"flashinfer call failed: {e}"}

    # BatchDecodeWithPagedKVCacheWrapper for batch > 1
    page_size = PAGE_SIZE
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = BATCH * num_pages_per_seq
    # Paged KV cache: [total_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(total_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
                          dtype=torch.float16, device=DEVICE)
    # Page table
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=DEVICE) * num_pages_per_seq
    last_page_len = torch.full((BATCH,), seq_len - (num_pages_per_seq - 1) * page_size,
                               dtype=torch.int32, device=DEVICE)
    q = torch.randn(BATCH, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    try:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(kv_indptr, kv_indices, last_page_len,
                     NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
                     q_data_type=torch.float16, kv_data_type=torch.float16)
        def run():
            wrapper.run(q, kv_data)
        return _capture_and_time(run)
    except Exception as e:
        import traceback
        return {"error": f"flashinfer BatchDecode failed: {e}",
                "traceback": traceback.format_exc()[:500]}


def _bench_v4_graph(seq_len: int, data: dict):
    def run():
        torch.ops.turboquant.decode_v4_from_cache(
            data["q"], data["cache"],
            data["indices"], data["indptr"],
            data["last_page_len"], data["seq_lens"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
            data["sm_scale"], data["signs"],
            data["qbytes"], data["nbytes"],
        )
    return _capture_and_time(run)


def _bench_v5_graph(seq_len: int, data: dict, ws: dict):
    def run():
        torch.ops.turboquant_v5.decode_v5_from_cache_ws(
            data["q"], data["cache"],
            data["indices"], data["indptr"],
            data["last_page_len"], data["seq_lens"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
            data["sm_scale"], data["signs"],
            data["qbytes"], data["nbytes"],
            ws["k_quant"], ws["v_quant"], ws["k_norms"], ws["v_norms"], ws["o"],
            data["max_len"],
        )
    return _capture_and_time(run)


def _bench_v5_split_graph(seq_len: int, data: dict, ws_split: dict, num_splits: int):
    def run():
        torch.ops.turboquant_v5.decode_v5_from_cache_splitkv_ws(
            data["q"], data["cache"],
            data["indices"], data["indptr"],
            data["last_page_len"], data["seq_lens"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
            data["sm_scale"], data["signs"],
            data["qbytes"], data["nbytes"],
            ws_split["k_quant"], ws_split["v_quant"],
            ws_split["k_norms"], ws_split["v_norms"], ws_split["o"],
            data["max_len"],
            ws_split["partition_o"], ws_split["partition_lse"],
            ws_split["request_indices"], ws_split["kv_tile_indices"],
            ws_split["split_indptr"], ws_split["kv_chunk_size"],
            num_splits,
        )
    return _capture_and_time(run)


def _bench_v5_paged_split_graph(seq_len: int, data: dict, ws_split: dict, num_splits: int):
    """HYP-035: paged-native split-KV. Kernel walks page table directly —
    ws["k_quant"/"v_quant"/"k_norms"/"v_norms"] are unused."""
    def run():
        torch.ops.turboquant_v5.decode_v5_from_cache_paged_splitkv_ws(
            data["q"], data["cache"],
            data["indices"], data["indptr"],
            data["last_page_len"], data["seq_lens"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
            data["sm_scale"], data["signs"],
            data["qbytes"], data["nbytes"],
            ws_split["o"], data["max_len"],
            ws_split["partition_o"], ws_split["partition_lse"],
            ws_split["request_indices"], ws_split["kv_tile_indices"],
            ws_split["split_indptr"], ws_split["kv_chunk_size"],
            num_splits,
        )
    return _capture_and_time(run)


def _check_correctness(data: dict, ws: dict, v5_module):
    """Assert v5-ws (graph-safe) matches v5-eager (reference) bitwise-ish."""
    out_eager = v5_module.decode_v5_from_cache(
        data["q"], data["cache"],
        data["indices"], data["indptr"],
        data["last_page_len"], data["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        data["sm_scale"], data["signs"],
        data["qbytes"], data["nbytes"],
    )
    torch.cuda.synchronize()
    out_ws = torch.ops.turboquant_v5.decode_v5_from_cache_ws(
        data["q"], data["cache"],
        data["indices"], data["indptr"],
        data["last_page_len"], data["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        data["sm_scale"], data["signs"],
        data["qbytes"], data["nbytes"],
        ws["k_quant"], ws["v_quant"], ws["k_norms"], ws["v_norms"], ws["o"],
        data["max_len"],
    )
    torch.cuda.synchronize()
    diff = (out_eager.float() - out_ws.float()).abs()
    max_abs = diff.max().item()
    cos = F.cosine_similarity(out_eager.flatten().float().unsqueeze(0),
                              out_ws.flatten().float().unsqueeze(0)).item()
    return {"max_abs": max_abs, "cosine": cos}


def _load_modules():
    """JIT-build v4 and v5 bindings (registers torch.ops.turboquant[_v5].*)."""
    from turboquant.decode_kernel_v4 import _get_module, _CSRC_DIR, _find_flashinfer_include
    v4_module = _get_module()  # registers torch.ops.turboquant.*
    from torch.utils.cpp_extension import load
    fi_include = _find_flashinfer_include()
    v5_module = load(
        name="turboquant_v5_tc",
        sources=[str(_CSRC_DIR / "src" / "decode_v5_tc_binding.cu")],
        extra_include_paths=[str(_CSRC_DIR / "include"), str(fi_include)],
        extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr",
            "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False)
    return v4_module, v5_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)

    print(f"[bench_v5_graph] seq_len={args.seq_len}  loading kernels...", flush=True)
    v4_module, v5_module = _load_modules()

    print(f"[bench_v5_graph] building inputs...", flush=True)
    data = _make_tq_inputs(args.seq_len)
    ws = _alloc_v5_workspace(data["max_len"], data["qbytes"], data["dim_chunks"])
    num_splits = _choose_num_splits(data["max_len"])
    ws_split = _alloc_v5_split_workspace(data["max_len"], data["qbytes"],
                                         data["dim_chunks"], num_splits)
    print(f"[bench_v5_graph] num_splits (HYP-034 heuristic) = {num_splits}", flush=True)

    print(f"[bench_v5_graph] correctness check (v5-ws vs v5-eager)...", flush=True)
    corr = _check_correctness(data, ws, v5_module)
    print(f"  max_abs={corr['max_abs']:.6f}  cosine={corr['cosine']:.6f}", flush=True)
    corr["passed"] = bool(corr["max_abs"] < 1e-2)

    # HYP-034 correctness: split-KV output should match non-split within cosine.
    split_out = torch.ops.turboquant_v5.decode_v5_from_cache_splitkv_ws(
        data["q"], data["cache"], data["indices"], data["indptr"],
        data["last_page_len"], data["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        data["sm_scale"], data["signs"], data["qbytes"], data["nbytes"],
        ws_split["k_quant"], ws_split["v_quant"],
        ws_split["k_norms"], ws_split["v_norms"], ws_split["o"],
        data["max_len"],
        ws_split["partition_o"], ws_split["partition_lse"],
        ws_split["request_indices"], ws_split["kv_tile_indices"],
        ws_split["split_indptr"], ws_split["kv_chunk_size"],
        num_splits)
    torch.cuda.synchronize()
    nosplit_out = v5_module.decode_v5_from_cache(
        data["q"], data["cache"], data["indices"], data["indptr"],
        data["last_page_len"], data["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        data["sm_scale"], data["signs"], data["qbytes"], data["nbytes"])
    torch.cuda.synchronize()
    split_corr_cos = F.cosine_similarity(
        split_out.flatten().float().unsqueeze(0),
        nosplit_out.flatten().float().unsqueeze(0)).item()
    split_corr_maxabs = (split_out.float() - nosplit_out.float()).abs().max().item()
    print(f"  split vs nosplit: cosine={split_corr_cos:.6f}  max_abs={split_corr_maxabs:.6f}",
          flush=True)

    results = {
        "seq_len": args.seq_len,
        "num_splits": num_splits,
        "config": {
            "batch": BATCH, "num_kv_heads": NUM_KV_HEADS,
            "num_qo_heads": NUM_QO_HEADS, "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
        },
        "correctness": corr,
        "split_correctness": {
            "cosine": split_corr_cos,
            "max_abs": split_corr_maxabs,
            "passed": bool(split_corr_cos >= 0.9999),
        },
    }

    # HYP-035 correctness: paged-native split output should match non-split eager.
    paged_out = torch.ops.turboquant_v5.decode_v5_from_cache_paged_splitkv_ws(
        data["q"], data["cache"], data["indices"], data["indptr"],
        data["last_page_len"], data["seq_lens"],
        NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, PADDED_DIM,
        data["sm_scale"], data["signs"], data["qbytes"], data["nbytes"],
        ws_split["o"], data["max_len"],
        ws_split["partition_o"], ws_split["partition_lse"],
        ws_split["request_indices"], ws_split["kv_tile_indices"],
        ws_split["split_indptr"], ws_split["kv_chunk_size"],
        num_splits)
    torch.cuda.synchronize()
    paged_corr_cos = F.cosine_similarity(
        paged_out.flatten().float().unsqueeze(0),
        nosplit_out.flatten().float().unsqueeze(0)).item()
    paged_corr_maxabs = (paged_out.float() - nosplit_out.float()).abs().max().item()
    print(f"  paged vs nosplit: cosine={paged_corr_cos:.6f}  max_abs={paged_corr_maxabs:.6f}",
          flush=True)
    results["paged_correctness"] = {
        "cosine": paged_corr_cos,
        "max_abs": paged_corr_maxabs,
        "passed": bool(paged_corr_cos >= 0.9999),
    }

    for name, fn in [
        ("fp16_sdpa",               lambda: _bench_fp16_sdpa(args.seq_len)),
        ("flashinfer",              lambda: _bench_flashinfer(args.seq_len)),
        ("tq_v4_graph",             lambda: _bench_v4_graph(args.seq_len, data)),
        ("tq_v5_graph",             lambda: _bench_v5_graph(args.seq_len, data, ws)),
        ("tq_v5_split_graph",       lambda: _bench_v5_split_graph(args.seq_len, data,
                                                                  ws_split, num_splits)),
        ("tq_v5_paged_split_graph", lambda: _bench_v5_paged_split_graph(args.seq_len, data,
                                                                        ws_split, num_splits)),
    ]:
        print(f"[bench_v5_graph] timing {name}...", flush=True)
        try:
            results[name] = fn()
            if "p50_us" in results[name]:
                print(f"  {name}: p50={results[name]['p50_us']:.1f}μs"
                      f"  p99={results[name]['p99_us']:.1f}μs", flush=True)
            else:
                print(f"  {name}: {results[name]}", flush=True)
        except Exception as e:
            import traceback
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()}
            print(f"  {name}: FAILED ({e})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench_v5_graph] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
