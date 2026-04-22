"""CUDA-graph comparison: FA / FI / ours, all with graphs enabled.

Configs: s1024×c8, s2048×c32, s8192×c8, s16384×c8, s32768×c4, s32768×c8.
Upstream is not re-run (already compiled+graphs in prior results).

Compares against the eager-mode numbers from results/v5_serve/ and
results/v5_serve_long/ to quantify the speedup CUDA graphs provide.
"""
import glob
import json
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)


def load(dirname, prefix):
    """Load `{prefix}-s*.json` from results/{dirname}/, keyed by (seq, conc)."""
    rows = {}
    base = os.path.join(ROOT, dirname)
    for p in sorted(glob.glob(os.path.join(base, f"{prefix}-s*.json"))):
        x = json.load(open(p))
        tag = os.path.basename(p).replace(".json", "").replace("-g", "")
        parts = tag.split("-")
        seq = int([p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()][0])
        conc = int([p[1:] for p in parts if p.startswith("c") and p[1:].isdigit()][0])
        rows[(seq, conc)] = x
    return rows


fa_g = load("v5_serve_graphs", "fa")
fi_g = load("v5_serve_graphs", "fi")
tq_g = load("v5_serve_graphs", "tq")

fa_e = {**load("v5_serve", "fa"), **load("v5_serve_long", "fa")}
fi_e = {**load("v5_serve", "fi"), **load("v5_serve_long", "fi")}
tq_e = {**load("v5_serve", "tq"), **load("v5_serve_long", "tq")}

keys = sorted(set(fa_g) | set(fi_g) | set(tq_g))


def f(x, k, digits=1):
    return f"{x[k]:.{digits}f}" if x else "—"


def ratio(a, b, k):
    if not (a and b):
        return "—"
    return f"{a[k]/b[k]:.2f}x"


def speedup(e, g, k):
    """eager / graphs — >1 means graphs are faster (for latency metrics)."""
    if not (e and g):
        return "—"
    return f"{e[k]/g[k]:.2f}x"


print("# CUDA graphs: FA / FI / ours with full graphs enabled")
print()
print("Qwen/Qwen3-8B fp16 on A100-40GB. output_len=128, num_prompts=64.")
print("- All backends: --compilation-config cudagraph_mode:FULL (ours also mode:0")
print("  because A100 SM80 cannot torch.compile fp8e4nv; inductor stays off while")
print("  graph capture stays on).")
print("- Upstream column omitted: already compiled+graphs in prior eager tables.")
print()
print("## Median TTFT (ms) — lower is better")
print()
print("| seq × c |   FA |   FI | **ours** | ours/FA | FA g→e | FI g→e | ours g→e |")
print("|---------|-----:|-----:|---------:|--------:|-------:|-------:|---------:|")
for s, c in keys:
    rfa, rfi, ro = fa_g.get((s, c)), fi_g.get((s, c)), tq_g.get((s, c))
    rfae, rfie, roe = fa_e.get((s, c)), fi_e.get((s, c)), tq_e.get((s, c))
    k = "median_ttft_ms"
    print(
        f"| {s:>5} × {c:>2} | {f(rfa, k, 0):>4} | {f(rfi, k, 0):>4} | "
        f"**{f(ro, k, 0):>5}** | {ratio(ro, rfa, k):>6} | "
        f"{speedup(rfae, rfa, k):>5} | {speedup(rfie, rfi, k):>5} | {speedup(roe, ro, k):>6} |"
    )

print()
print("## Median TPOT (ms) — lower is better")
print()
print("| seq × c |   FA |   FI | **ours** | ours/FA | FA g→e | FI g→e | ours g→e |")
print("|---------|-----:|-----:|---------:|--------:|-------:|-------:|---------:|")
for s, c in keys:
    rfa, rfi, ro = fa_g.get((s, c)), fi_g.get((s, c)), tq_g.get((s, c))
    rfae, rfie, roe = fa_e.get((s, c)), fi_e.get((s, c)), tq_e.get((s, c))
    k = "median_tpot_ms"
    print(
        f"| {s:>5} × {c:>2} | {f(rfa, k):>4} | {f(rfi, k):>4} | "
        f"**{f(ro, k):>5}** | {ratio(ro, rfa, k):>6} | "
        f"{speedup(rfae, rfa, k):>5} | {speedup(rfie, rfi, k):>5} | {speedup(roe, ro, k):>6} |"
    )

print()
print("## Output throughput (tok/s) — higher is better")
print()
print("| seq × c |    FA |    FI | **ours** | ours/FA | FA e→g | FI e→g | ours e→g |")
print("|---------|------:|------:|---------:|--------:|-------:|-------:|---------:|")
for s, c in keys:
    rfa, rfi, ro = fa_g.get((s, c)), fi_g.get((s, c)), tq_g.get((s, c))
    rfae, rfie, roe = fa_e.get((s, c)), fi_e.get((s, c)), tq_e.get((s, c))
    k = "output_throughput"
    # For throughput, graphs / eager > 1 means graphs are faster.
    def tput(e, g):
        if not (e and g):
            return "—"
        return f"{g[k]/e[k]:.2f}x"
    print(
        f"| {s:>5} × {c:>2} | {f(rfa, k):>5} | {f(rfi, k):>5} | "
        f"**{f(ro, k):>6}** | {ratio(ro, rfa, k):>6} | "
        f"{tput(rfae, rfa):>5} | {tput(rfie, rfi):>5} | {tput(roe, ro):>6} |"
    )
