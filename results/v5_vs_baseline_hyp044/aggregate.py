"""Compare HYP-041 v0 (pre-HYP-044) vs HYP-041 sweep with HYP-044 chunk-cap heuristic."""
import glob
import json
import os

V0_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "v5_vs_baseline"))
V1_DIR = os.path.dirname(__file__)


def load(d):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        x = json.load(open(p))
        rows.setdefault((x["input_len"], x["batch"]), {})[x["backend"]] = x
    return rows


old, new = load(V0_DIR), load(V1_DIR)
keys = sorted(set(old) | set(new))

print("# HYP-044 end-to-end: HYP-041 sweep with chunk-cap heuristic")
print()
print("Model: Qwen/Qwen3-8B (fp16) | output_len=128 | A100-40GB | enforce_eager=True")
print()
print("v0 = HYP-041 (old _choose_num_splits). v1 = HYP-044 patched.")
print()
print("| seq  | b  | base v0 | base v1 | tq v0  | tq v1  | Δ tq   | tq/base v0 | tq/base v1 | Δ ratio |")
print("|-----:|---:|--------:|--------:|-------:|-------:|-------:|-----------:|-----------:|--------:|")


def fmt(d, k="decode_tokens_per_s"):
    if d is None: return "—"
    return f"{d[k]:.1f}"


for s, b in keys:
    o, n = old.get((s, b), {}), new.get((s, b), {})
    bo = o.get("baseline"); to = o.get("tq")
    bn = n.get("baseline"); tn = n.get("tq")
    dtq = (f"{(tn['decode_tokens_per_s']/to['decode_tokens_per_s']-1)*100:+.0f}%"
           if (to and tn) else "—")
    r0 = (f"{to['decode_tokens_per_s']/bo['decode_tokens_per_s']:.2f}x"
          if (bo and to) else "—")
    r1 = (f"{tn['decode_tokens_per_s']/bn['decode_tokens_per_s']:.2f}x"
          if (bn and tn) else "—")
    drat = ("—" if "—" in (r0, r1) else
            f"{(float(r1[:-1])-float(r0[:-1]))*100:+.0f}pp")
    print(f"| {s:>4} | {b:>2} | {fmt(bo):>7} | {fmt(bn):>7} | {fmt(to):>6} | "
          f"{fmt(tn):>6} | {dtq:>6} | {r0:>10} | {r1:>10} | {drat:>7} |")

print()
print("Notes:")
print("- tq v1/v0 = kernel speedup from HYP-044 (chunk_size cap at 256 tokens,")
print("  no batch divisor in split count); microbench predicted 0.79–0.84× kernel latency")
print("  at batch≥8, translating to ~15% end-to-end because attention is ~91% of per-step Δ.")
print("- Configs marked — are HYP-041 OOMs that persist (workspace allocation, not split-K);")
print("  HYP-045 (pre-allocated workspace) is the fix.")
