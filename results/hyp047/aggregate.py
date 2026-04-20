"""HYP-047: cross transfer cost vs TQ prefill cost.

Reads transfer measurements from this dir's results.csv and prefill
measurements from results/v5_split/ (HYP-041 grid) and
results/v5_split_x/ (extension: 16384x4, 32k×{1,4,8}).
"""
import csv
import json
import os

D = os.path.dirname(__file__)
SPLIT = os.path.normpath(os.path.join(D, "..", "v5_split"))
SPLIT_X = os.path.normpath(os.path.join(D, "..", "v5_split_x"))


def load_split(seq, batch):
    for d in (SPLIT, SPLIT_X):
        p = os.path.join(d, f"tq-s{seq}-b{batch}.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


# Read both transfer csvs
rows = []
for csv_path in [os.path.join(D, "results.csv"),
                 os.path.join(D, "results_x.csv")]:
    if os.path.exists(csv_path):
        rows.extend(list(csv.DictReader(open(csv_path))))


def f(x): return float(x) if x else None


print("# HYP-047 — TQ KV offload + reuse: PCIe transfer cost vs measured prefill")
print()
print("A100-40GB, PCIe Gen4 x16. Pinned host memory, blocking copies.")
print("Cache size = 36 layers × 2 (K+V) × batch × 8 KV-heads × seq × 64 qbytes.")
print()
print("| seq × b | KV size | g2c (ms) | c2g (ms) | bw (GB/s) | TQ prefill (ms) | "
      "**c2g / prefill** | overlap (decodes hidden) |")
print("|---------|--------:|---------:|---------:|----------:|----------------:|"
      "------------------:|-------------------------:|")
seen = set()
for r in sorted(rows, key=lambda r: (int(r["seq"]), int(r["batch"]))):
    s, b = int(r["seq"]), int(r["batch"])
    if (s, b) in seen: continue
    seen.add((s, b))
    sp = load_split(s, b)
    pf_ms = sp["prefill_s"] * 1000 if sp else None
    dec_ms = sp["decode_per_step_s"] * 1000 if sp else None
    g2c = f(r.get("g2c_med_s")); c2g = f(r.get("c2g_med_s"))
    if g2c is None or c2g is None:
        print(f"| {s:>5} × {b:>2} | — | — | — | — | — | — | — |"); continue
    g2c_ms = g2c * 1000; c2g_ms = c2g * 1000
    nbytes = float(r["nbytes"])
    ratio = (c2g_ms / pf_ms) if pf_ms else None
    n_steps = (c2g_ms / dec_ms) if dec_ms else None
    flag = "✓" if (ratio is not None and ratio < 1.0) else ("≈" if ratio and ratio < 1.5 else "✗")
    bw = nbytes / c2g / 1e9
    print(f"| {s:>5} × {b:>2} | {nbytes/1e9:>5.2f} GB | {g2c_ms:>7.1f} | {c2g_ms:>7.1f} | "
          f"{bw:>5.1f} | {(f'{pf_ms:.1f}' if pf_ms else '—'):>13} | "
          f"{(f'{ratio:.2f}x' if ratio else '—')} {flag} | "
          f"{(f'{n_steps:.1f} × {dec_ms:.1f} ms' if n_steps else '—')} |")

print()
print("## Read")
print()
print("- **PCIe Gen4 effective bandwidth ≈ 26 GB/s** (vs 32 GB/s peak — 81 %).")
print("  Symmetric in both directions.")
print()
print("### Extended grid (HYP-047b: 16384×4, 32k×{1,4,8})")
print()
print("- **TQ prefill is fast at long-ctx** (32768×1 = 41 ms vs FA 68 ms; 32768×8 =")
print("  259 ms vs FA 361 ms). Side-effect: synchronous restore loses to re-prefill")
print("  at every 32k config (c2g/prefill = 1.13–1.43×). Async overlap still hides it")
print("  (restore = 0.6–4.9 decode steps).")
print("- **The trade flips at 32k**: at the original HYP-041 grid restore wins")
print("  outright; at the extended grid TQ prefill is cheap enough that re-prefill")
print("  is competitive with restore. Async overlap is the deciding factor.")
print("- **One regime is clear**: TQ + offload only beats raw fp16 + offload, never")
print("  itself + re-prefill at 32k+. Past 32k the value is *capacity* (fitting the")
print("  context at all) rather than *prefill amortization*.")
