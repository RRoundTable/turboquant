"""4-way: FA / FI / ours-TQ (CUDA) / upstream-TQ (Triton, nightly vllm)."""
import glob
import json
import os

HERE = os.path.dirname(__file__)
V5 = os.path.normpath(os.path.join(HERE, "..", "v5_serve"))


def load(d, prefix):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, f"{prefix}*.json"))):
        x = json.load(open(p))
        tag = os.path.basename(p).replace(".json", "")
        parts = tag.split("-")
        seq = int([p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()][0])
        conc = int([p[1:] for p in parts if p.startswith("c") and p[1:].isdigit()][0])
        rows[(seq, conc)] = x
    return rows


fa = load(V5, "fa")
fi = load(V5, "fi")
tq_ours = load(V5, "tq")
tq_up = load(HERE, "upstream")
keys = sorted(set(fa) & set(fi) & set(tq_ours) & set(tq_up))


def fmt(x, k, digits=1): return f"{x[k]:.{digits}f}" if x else "—"
def ratio(a, b, k):
    if not (a and b): return "—"
    return f"{a[k]/b[k]:.2f}x"


print("# Upstream vllm turboquant vs our TurboQuant-CUDA vs FA / FI")
print()
print("- Upstream turboquant: vllm nightly (0.19.2rc1.dev21+g893611813), `--kv-cache-dtype turboquant_4bit_nc`, Triton kernels.")
print("- Ours: vllm v0.19.0 + docker/vllm_patches (PR #39868), `--attention-backend CUSTOM --kv-cache-dtype fp8`, hand-written CUDA kernel.")
print("- FA / FI: vllm v0.19.0 baselines from results/v5_serve/.")
print("- Qwen/Qwen3-8B fp16, A100-40GB. `vllm bench serve --dataset-name random`.")
print()
print("## Median TTFT (ms) — lower is better")
print()
print("| seq × conc |  FA | FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|------------|----:|---:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {fmt(rfa,'median_ttft_ms',0):>3} | {fmt(rfi,'median_ttft_ms',0):>3} | "
          f"**{fmt(ro,'median_ttft_ms',0):>4}** | **{fmt(ru,'median_ttft_ms',0):>4}** | "
          f"{ratio(ro,rfa,'median_ttft_ms'):>6} | {ratio(ru,rfa,'median_ttft_ms'):>5} | {ratio(ru,ro,'median_ttft_ms'):>6} |")

print()
print("## Median TPOT (ms) — lower is better")
print()
print("| seq × conc |   FA |   FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|------------|-----:|-----:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {fmt(rfa,'median_tpot_ms'):>4} | {fmt(rfi,'median_tpot_ms'):>4} | "
          f"**{fmt(ro,'median_tpot_ms'):>5}** | **{fmt(ru,'median_tpot_ms'):>5}** | "
          f"{ratio(ro,rfa,'median_tpot_ms'):>6} | {ratio(ru,rfa,'median_tpot_ms'):>5} | {ratio(ru,ro,'median_tpot_ms'):>6} |")

print()
print("## Output throughput (tok/s) — higher is better")
print()
print("| seq × conc |     FA |     FI | **ours** | **upstream** | ours/FA | up/FA | up/ours |")
print("|------------|-------:|-------:|---------:|-------------:|--------:|------:|--------:|")
for s, c in keys:
    rfa, rfi, ro, ru = fa.get((s, c)), fi.get((s, c)), tq_ours.get((s, c)), tq_up.get((s, c))
    print(f"| {s:>5} × {c:>2} | {fmt(rfa,'output_throughput'):>6} | {fmt(rfi,'output_throughput'):>6} | "
          f"**{fmt(ro,'output_throughput'):>6}** | **{fmt(ru,'output_throughput'):>6}** | "
          f"{ratio(ro,rfa,'output_throughput'):>6} | {ratio(ru,rfa,'output_throughput'):>5} | {ratio(ru,ro,'output_throughput'):>6} |")

print()
print("## Read")
print()
print("Two different performance envelopes:")
print()
print("- **Upstream wins at short-ctx low-load** (s1024×8): TPOT 19.2 ms (vs our 29.5,")
print("  FA 22.7) and 253 tok/s (vs our 198, FA 235). Its Triton dequant matches FA on")
print("  lightly-loaded decode.")
print("- **Ours wins at long-ctx high-load** (s8192×8, s2048×32): upstream falls to")
print("  ~30 % slower than ours and ~25 % slower than FA. Our fused dequant+attention")
print("  CUDA kernel beats upstream's separate-dequant Triton path once the decode")
print("  hot loop dominates.")
print("- **TTFT** at long ctx: ours beats FA (1315 vs 1788 at s8192×8); upstream is")
print("  *worse* than FA (1993) — its Triton write path adds prefill overhead the")
print("  CUDA path doesn't.")
print("- **Neither reliably beats FA at throughput** except upstream at s1024×8.")
print()
print("Upstream and ours solve different points on the curve: portability + short-ctx")
print("vs A100-specialized + long-ctx / high-batch.")
