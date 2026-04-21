"""Long-ctx 4-way comparison: FA / FI / ours-TQ (CUDA) / upstream-TQ (Triton).

Configs: s16384 × c8, s32768 × c4, s32768 × c8.
"""
import glob
import json
import os

HERE = os.path.dirname(__file__)


def load(prefix):
    rows = {}
    for p in sorted(glob.glob(os.path.join(HERE, f"{prefix}-s*.json"))):
        x = json.load(open(p))
        tag = os.path.basename(p).replace(".json", "")
        parts = tag.split("-")
        seq = int([p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()][0])
        conc = int([p[1:] for p in parts if p.startswith("c") and p[1:].isdigit()][0])
        rows[(seq, conc)] = x
    return rows


fa = load("fa")
fi = load("fi")
tq_ours = load("tq")
tq_up = load("upstream")
keys = sorted(set(fa) | set(fi) | set(tq_ours) | set(tq_up))


def f(x, k, digits=1): return f"{x[k]:.{digits}f}" if x else "—"
def ratio(a, b, k):
    if not (a and b): return "—"
    return f"{a[k]/b[k]:.2f}x"


print("# Long-context serving: FA / FI / ours vs upstream turboquant")
print()
print("Qwen/Qwen3-8B fp16 on A100-40GB. output_len=128, num_prompts=64.")
print("- FA / FI / ours: stock vllm v0.19.0 + docker/vllm_patches (PR #39868).")
print("- upstream: vllm nightly 0.19.2rc1.dev21+g893611813, `--kv-cache-dtype turboquant_4bit_nc`.")
print()
print("## Median TTFT (ms) — lower is better")
print()
print("| seq × c |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|---------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {f(rfa,'median_ttft_ms',0):>4} | {f(rfi,'median_ttft_ms',0):>4} | "
          f"**{f(ro,'median_ttft_ms',0):>5}** | **{f(ru,'median_ttft_ms',0):>5}** | "
          f"{ratio(ro,rfa,'median_ttft_ms'):>6} | {ratio(ru,rfa,'median_ttft_ms'):>5} | {ratio(ru,ro,'median_ttft_ms'):>6} |")

print()
print("## Median TPOT (ms) — lower is better")
print()
print("| seq × c |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|---------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {f(rfa,'median_tpot_ms'):>4} | {f(rfi,'median_tpot_ms'):>4} | "
          f"**{f(ro,'median_tpot_ms'):>5}** | **{f(ru,'median_tpot_ms'):>5}** | "
          f"{ratio(ro,rfa,'median_tpot_ms'):>6} | {ratio(ru,rfa,'median_tpot_ms'):>5} | {ratio(ru,ro,'median_tpot_ms'):>6} |")

print()
print("## Output throughput (tok/s) — higher is better")
print()
print("| seq × c |    FA |    FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|---------|------:|------:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {f(rfa,'output_throughput'):>5} | {f(rfi,'output_throughput'):>5} | "
          f"**{f(ro,'output_throughput'):>6}** | **{f(ru,'output_throughput'):>6}** | "
          f"{ratio(ro,rfa,'output_throughput'):>6} | {ratio(ru,rfa,'output_throughput'):>5} | {ratio(ru,ro,'output_throughput'):>6} |")
