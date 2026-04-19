"""HYP-044: batch sweep of v5_paged decode kernel under two split-K heuristics.

Standalone (no vLLM). Times the kernel under CUDA-graph capture so per-call
overhead is amortized. Reports old vs proposed heuristic for batch ∈ {1..32}.

Designed to run on Forge A100. Uses the bench_v5_graph harness for builds.
"""

import argparse
import csv
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override BATCH at runtime — bench_v5_graph reads it as a module-level constant.
def _set_batch(b: int):
    os.environ["TQ_BENCH_BATCH"] = str(b)
    # Reload by re-importing fresh (or just import once and patch the symbol).


def _build_helpers(batch: int):
    """Import bench_v5_graph after BATCH env is set so its module-level constant picks up.
    JIT-compiled kernels register torch.ops globally, so reloading is safe — the
    cost we pay on each reload is just a fresh module-import (no recompile).
    """
    os.environ["TQ_BENCH_BATCH"] = str(batch)
    import importlib
    import bench_v5_graph
    importlib.reload(bench_v5_graph)
    return bench_v5_graph


def _choose_num_splits_old(max_len, batch, num_kv_heads, sm_count):
    """Current heuristic (vllm_backend_fused.py:147)."""
    if max_len < 512:
        return 1
    splits_by_sm = max(1, (4 * sm_count) // (batch * num_kv_heads))
    splits_by_work = max(1, max_len // 32)
    target = min(splits_by_sm, splits_by_work)
    best = 1
    for d in range(2, target + 1):
        if max_len % d == 0:
            best = d
    return best


def _choose_num_splits_new(max_len, batch, num_kv_heads, sm_count,
                           target_chunk: int = 256):
    """Proposed: cap chunk_size, no batch divisor."""
    if max_len < 512:
        return 1
    splits_by_chunk = max(1, max_len // target_chunk)
    splits_by_sm = max(1, (4 * sm_count) // num_kv_heads)
    target = min(splits_by_chunk, splits_by_sm)
    best = 1
    for d in range(2, target + 1):
        if max_len % d == 0:
            best = d
    return best


_MODULES_LOADED = False


def time_kernel(seq_len: int, batch: int, num_splits: int,
                warmup: int = 20, iters: int = 200) -> float:
    """Build inputs at this (batch, seq_len), capture into CUDA graph, time."""
    global _MODULES_LOADED
    bench = _build_helpers(batch)
    if not _MODULES_LOADED:
        bench._load_modules()  # registers torch.ops.turboquant_v5.*
        _MODULES_LOADED = True
    data = bench._make_tq_inputs(seq_len)
    qbytes = data["qbytes"]
    dim_chunks = data["dim_chunks"]
    max_len = data["max_len"]
    ws = bench._alloc_v5_split_workspace(max_len, qbytes, dim_chunks, num_splits)

    # Use the helper that already binds to the correct schema.
    timing = bench._bench_v5_paged_split_graph(seq_len, data, ws, num_splits)
    return timing["mean_us"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--out", default="/workspace/shared/hyp044/results.csv")
    args = ap.parse_args()

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    bench = _build_helpers(1)  # arbitrary; we use it just for NUM_KV_HEADS
    num_kv_heads = bench.NUM_KV_HEADS
    print(f"sm_count={sm_count}  num_kv_heads={num_kv_heads}  "
          f"seq_len={args.seq_len}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for batch in [int(b) for b in args.batches.split(",")]:
        for tag, picker in [("old", _choose_num_splits_old),
                            ("new", _choose_num_splits_new)]:
            ns = picker(args.seq_len, batch, num_kv_heads, sm_count)
            chunk = (args.seq_len + ns - 1) // ns
            grid = batch * ns * num_kv_heads
            print(f"\n[{tag}] batch={batch}  num_splits={ns}  "
                  f"chunk={chunk}  grid_blocks={grid}", flush=True)
            try:
                t0 = time.perf_counter()
                kernel_us = time_kernel(args.seq_len, batch, ns)
                t1 = time.perf_counter()
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM: {e}", flush=True)
                rows.append({
                    "heuristic": tag, "batch": batch, "num_splits": ns,
                    "chunk_size": chunk, "grid_blocks": grid,
                    "kernel_us": None, "wall_s": None, "note": "OOM"
                })
                torch.cuda.empty_cache()
                continue
            print(f"  kernel mean = {kernel_us:.2f} us "
                  f"(setup wall {t1-t0:.1f} s)", flush=True)
            rows.append({
                "heuristic": tag, "batch": batch, "num_splits": ns,
                "chunk_size": chunk, "grid_blocks": grid,
                "kernel_us": round(kernel_us, 2),
                "wall_s": round(t1 - t0, 1), "note": ""
            })
            torch.cuda.empty_cache()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
