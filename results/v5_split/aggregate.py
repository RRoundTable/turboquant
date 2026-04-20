"""Clean prefill-vs-decode split: FA vs FI vs TQ across 12 configs.

Each JSON has:
  prefill_s, decode_per_step_s, decode_tokens_per_s (clean), total_tokens_per_s (legacy)
"""
import glob
import json
import os

D = os.path.dirname(__file__)


def load(prefix):
    rows = {}
    for p in sorted(glob.glob(os.path.join(D, f"{prefix}*.json"))):
        x = json.load(open(p))
        rows[(x["input_len"], x["batch"])] = x
    return rows


fa = load("baseline-")
fi = load("flashinfer-")
tq = load("tq-")
keys = sorted(set(fa) | set(fi) | set(tq))


def ms(d, key): return f"{d[key]*1000:.1f}" if d else "—"
def tok(d): return f"{d['decode_tokens_per_s']:.0f}" if d else "—"
def ratio(a, b, key):
    if not (a and b): return "—"
    return f"{a[key]/b[key]:.2f}x"


print("# Prefill vs decode (split metric): FA vs FI vs TQ")
print()
print("Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True")
print()
print("Method: T_short = generate(output_len=1) ⇒ prefill + 1 decode.")
print("        T_long  = generate(output_len=128) ⇒ prefill + 128 decode.")
print("        decode_per_step = (T_long - T_short) / 127.")
print("        prefill         = T_short - decode_per_step.")
print()
print("## Prefill latency (ms) — TQ matches or beats baseline at long ctx")
print()
print("| seq × b | FA prefill | FI prefill | TQ prefill | TQ/FA | TQ/FI |")
print("|---------|----------:|----------:|----------:|------:|------:|")
for s, b in keys:
    f, i, t = fa.get((s, b)), fi.get((s, b)), tq.get((s, b))
    print(f"| {s:>5} × {b:>2} | {ms(f, 'prefill_s'):>10} | "
          f"{ms(i, 'prefill_s'):>10} | {ms(t, 'prefill_s'):>10} | "
          f"{ratio(t, f, 'prefill_s'):>5} | {ratio(t, i, 'prefill_s'):>5} |")

print()
print("## Decode-only latency (ms/step) and throughput (tok/s)")
print()
print("| seq × b | FA ms | FI ms | TQ ms | FA tok/s | FI tok/s | TQ tok/s | TQ/FA | TQ/FI |")
print("|---------|------:|------:|------:|---------:|---------:|---------:|------:|------:|")
for s, b in keys:
    f, i, t = fa.get((s, b)), fi.get((s, b)), tq.get((s, b))
    print(f"| {s:>5} × {b:>2} | "
          f"{ms(f, 'decode_per_step_s'):>5} | {ms(i, 'decode_per_step_s'):>5} | {ms(t, 'decode_per_step_s'):>5} | "
          f"{tok(f):>8} | {tok(i):>8} | {tok(t):>8} | "
          f"{ratio(t, f, 'decode_tokens_per_s'):>5} | {ratio(t, i, 'decode_tokens_per_s'):>5} |")

print()
print("## Notes")
print()
print("- **Prefill is at or below baseline for TQ across the grid.** At long ctx"
      " × large batch (16384 × 32), TQ prefill is 350 ms vs FA 537 ms / FI 475 ms"
      " — 1.5× faster than FA. Likely a side-effect of PR #39868: less KV bytes"
      " to write during prefill.")
print("- **Decode is uniformly slower for TQ.** Ratio worsens with (seq × batch):")
print("  - 1024×1   TQ/FI = 0.74×")
print("  - 8192×8   TQ/FI = 0.56×")
print("  - 16384×32 TQ/FI = 0.29×")
print("- The previous 'tok/s' columns in HYP-041/044/045 reports used"
      " total_wall = prefill + decode, which understated the decode gap at"
      " long ctx (where prefill dominates) and overstated TQ at short ctx"
      " (where prefill is negligible). Use this report's separated numbers.")
