"""Serving-mode comparison: vllm bench serve metrics, FA / FI / TQ."""
import glob
import json
import os

D = os.path.dirname(__file__)


def load(prefix):
    rows = {}
    for p in sorted(glob.glob(os.path.join(D, f"{prefix}-s*.json"))):
        x = json.load(open(p))
        # tag = "<backend>-s<seq>-c<conc>"
        tag = os.path.basename(p).replace(".json", "")
        # parse seq, conc out of tag
        parts = tag.split("-")
        seq = int([p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()][0])
        conc = int([p[1:] for p in parts if p.startswith("c") and p[1:].isdigit()][0])
        rows[(seq, conc)] = x
    return rows


fa = load("fa")
fi = load("fi")
tq = load("tq")

# Drop the s2048-c8 validation run; not part of the sweep
for d in (fa, fi, tq):
    d.pop((2048, 8), None)

keys = sorted(set(fa) | set(fi) | set(tq))


def f(x, k): return f"{x[k]:.0f}" if x else "—"
def fr(x, k): return f"{x[k]:.2f}" if x else "—"
def ratio(a, b, k):
    if not (a and b): return "—"
    return f"{a[k]/b[k]:.2f}x"


print("# Serving-mode comparison: deployed image (TQ 16c6e93) vs FA vs FI")
print()
print("Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True")
print("`vllm bench serve --dataset-name random` against `vllm serve` HTTP endpoint.")
print()

print("## Time-to-first-token (TTFT) — prefill cost in serving")
print()
print("| seq × conc | metric | FA (ms) | FI (ms) | **TQ (ms)** | TQ/FA | TQ/FI |")
print("|------------|--------|--------:|--------:|------------:|------:|------:|")
for s, c in keys:
    rfa, rfi, rtq = fa.get((s, c)), fi.get((s, c)), tq.get((s, c))
    for k, label in [("median_ttft_ms", "median"), ("mean_ttft_ms", "mean"),
                     ("p99_ttft_ms", "p99")]:
        print(f"| {s:>5} × {c:>2} | {label:6} | {f(rfa, k):>7} | {f(rfi, k):>7} | "
              f"**{f(rtq, k):>7}** | {ratio(rtq, rfa, k):>5} | {ratio(rtq, rfi, k):>5} |")

print()
print("## Time-per-output-token (TPOT) — decode latency")
print()
print("| seq × conc | metric | FA (ms) | FI (ms) | **TQ (ms)** | TQ/FA | TQ/FI |")
print("|------------|--------|--------:|--------:|------------:|------:|------:|")
for s, c in keys:
    rfa, rfi, rtq = fa.get((s, c)), fi.get((s, c)), tq.get((s, c))
    for k, label in [("median_tpot_ms", "median"), ("mean_tpot_ms", "mean"),
                     ("p99_tpot_ms", "p99")]:
        print(f"| {s:>5} × {c:>2} | {label:6} | {f(rfa, k):>7} | {f(rfi, k):>7} | "
              f"**{f(rtq, k):>7}** | {ratio(rtq, rfa, k):>5} | {ratio(rtq, rfi, k):>5} |")

print()
print("## Throughput (req/s, output tok/s, total tok/s)")
print()
print("| seq × conc | metric        |    FA |    FI | **TQ** | TQ/FA | TQ/FI |")
print("|------------|---------------|------:|------:|-------:|------:|------:|")
for s, c in keys:
    rfa, rfi, rtq = fa.get((s, c)), fi.get((s, c)), tq.get((s, c))
    for k, label in [("request_throughput", "req/s"),
                     ("output_throughput", "out tok/s"),
                     ("total_token_throughput", "total tok/s")]:
        print(f"| {s:>5} × {c:>2} | {label:13} | {fr(rfa, k):>5} | {fr(rfi, k):>5} | "
              f"**{fr(rtq, k):>5}** | {ratio(rtq, rfa, k):>5} | {ratio(rtq, rfi, k):>5} |")

print()
print("## Read")
print()
print("- **TTFT (prefill in serving) wins for TQ at long ctx**:")
print("  s8192×8 median TTFT: TQ 1315 ms vs FI 1544 ms (0.85×), FA 1788 ms (0.74×).")
print("  Matches offline split-metric finding (HYP-045 split-metric report).")
print("- **TPOT (decode in serving) is 1.19–1.34× slower for TQ**:")
print("  s2048×32 median TPOT: TQ 62 ms vs FI 50 ms (1.26×) — same A100 kernel-")
print("  ceiling story as HYP-042b.")
print("- **Output throughput is 0.85–0.91× FI for TQ**: smaller gap than the offline")
print("  bench because continuous-batching amortizes some of the per-step overhead.")
print("- **No regressions vs the deployed image** (TQ 16c6e93). All 3 configs ran")
print("  under `vllm serve` with the production patch overlay (PR #39868 + HYP-044 +")
print("  HYP-045) and produced clean serving-mode metrics.")
