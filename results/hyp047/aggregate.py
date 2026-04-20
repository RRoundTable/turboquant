"""HYP-047: cross transfer cost vs TQ prefill cost from results/v5_split/."""
import csv
import json
import os

D = os.path.dirname(__file__)
SPLIT = os.path.normpath(os.path.join(D, "..", "v5_split"))


def load_split(seq, batch):
    p = os.path.join(SPLIT, f"tq-s{seq}-b{batch}.json")
    if not os.path.exists(p): return None
    return json.load(open(p))


rows = list(csv.DictReader(open(os.path.join(D, "results.csv"))))


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
for r in rows:
    s, b = int(r["seq"]), int(r["batch"])
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
          f"{bw:>5.1f} | {pf_ms:>13.1f} | "
          f"{(f'{ratio:.2f}x' if ratio else '—')} {flag} | "
          f"{(f'{n_steps:.1f} × {dec_ms:.1f} ms' if n_steps else '—')} |")

print()
print("## Read")
print()
print("- **PCIe Gen4 effective bandwidth ≈ 26 GB/s** (vs 32 GB/s peak — 81 %).")
print("  Symmetric in both directions.")
print("- **One-way restore (cache hit on warm spill)** vs re-prefill:")
print("  - WIN at small/medium configs (≤ 8192 × 8): restore is 0.34–0.97× prefill")
print("  - LOSS at large × large: 16384 × 32 restore is 740 ms vs 350 ms re-prefill")
print("- **Async overlap with decode** changes the picture: at decode/step ≈ 30–130 ms,")
print("  the ~92 ms restore at 8192×8 is hidden behind 3 decode steps; only the largest")
print("  config (16384×32) needs ~5 steps to cover transfer.")
print("- **fp16 KV would be 3.2× larger** → almost no config wins for fp16 offload.")
print("  TQ compression makes the offload-reuse trade viable in a regime where")
print("  raw fp16 cannot.")
