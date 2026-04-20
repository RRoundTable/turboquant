"""HYP-047: raw GPU<->CPU transfer cost for TQ-compressed KV cache.

No vLLM, no models, no quant. Just allocate a TQ-shaped uint8 slab,
time the round trip to host pinned memory, compare to measured prefill
from results/v5_split/.
"""
import argparse
import csv
import json
import os
import statistics
import time

import torch

# Qwen3-8B shape (matches HYP-041 / HYP-045 sweep)
NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
QBYTES = 64       # 4-bit nibble + per-tile norms = 64 bytes per (head, token)


def slab_bytes(seq: int, batch: int) -> int:
    # K + V for every layer, one slab per request.
    return NUM_LAYERS * 2 * batch * NUM_KV_HEADS * seq * QBYTES


def time_round_trip(seq: int, batch: int, trials: int = 5):
    nbytes = slab_bytes(seq, batch)
    free, total = torch.cuda.mem_get_info()
    if nbytes > free * 0.9:
        return {"error": f"too large: {nbytes/1e9:.1f} GB > "
                f"{free/1e9:.1f} GB free"}

    g = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    g.random_(0, 256)  # touch every byte

    c = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    # warmup: ensure DMA path is hot
    c.copy_(g, non_blocking=False); torch.cuda.synchronize()
    g.copy_(c, non_blocking=False); torch.cuda.synchronize()

    g2c, c2g = [], []
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        c.copy_(g, non_blocking=False); torch.cuda.synchronize()
        g2c.append(time.perf_counter() - t0)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        g.copy_(c, non_blocking=False); torch.cuda.synchronize()
        c2g.append(time.perf_counter() - t0)

    del g, c
    torch.cuda.empty_cache()

    g2c_med = statistics.median(g2c)
    c2g_med = statistics.median(c2g)
    return {
        "nbytes": nbytes,
        "g2c_med_s": g2c_med,
        "c2g_med_s": c2g_med,
        "g2c_GBs": nbytes / g2c_med / 1e9,
        "c2g_GBs": nbytes / c2g_med / 1e9,
        "g2c_trials_s": g2c,
        "c2g_trials_s": c2g,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="1024,4096,8192,16384")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--out", default="/workspace/shared/hyp047/results.csv")
    args = ap.parse_args()

    seqs = [int(s) for s in args.seqs.split(",")]
    batches = [int(b) for b in args.batches.split(",")]

    rows = []
    for s in seqs:
        for b in batches:
            print(f"\n=== seq={s} batch={b} ({slab_bytes(s,b)/1e9:.2f} GB) ===",
                  flush=True)
            r = time_round_trip(s, b)
            if "error" in r:
                print(f"  SKIP: {r['error']}")
                rows.append({"seq": s, "batch": b, "note": r["error"]})
                continue
            print(f"  GPU->CPU: {r['g2c_med_s']*1000:>7.2f} ms  "
                  f"({r['g2c_GBs']:>5.1f} GB/s)")
            print(f"  CPU->GPU: {r['c2g_med_s']*1000:>7.2f} ms  "
                  f"({r['c2g_GBs']:>5.1f} GB/s)")
            rows.append({"seq": s, "batch": b, **{k: v for k, v in r.items()
                                                  if not k.endswith("trials_s")}})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        keys = ["seq", "batch", "nbytes", "g2c_med_s", "c2g_med_s",
                "g2c_GBs", "c2g_GBs", "note"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
