"""Build comparison table: TurboQuant v5_paged vs baseline vLLM."""
import glob
import json
import os

D = os.path.dirname(__file__)
rows = {}
for path in sorted(glob.glob(os.path.join(D, "*.json"))):
    d = json.load(open(path))
    key = (d["input_len"], d["batch"])
    rows.setdefault(key, {})[d["backend"]] = d


def fmt_tps(d):
    if d is None:
        return "OOM"
    return f"{d['decode_tokens_per_s']:.1f}"


def fmt_lat(d):
    if d is None:
        return "OOM"
    return f"{d['median_s']*1000:.0f}"


def fmt_mem(d):
    if d is None:
        return "OOM"
    m = d.get("nvsmi_gpu_mem_gb")
    return f"{m:.2f}" if m else "?"


print("# TurboQuant v5_paged vs baseline vLLM")
print()
print(f"Model: Qwen/Qwen3-8B (fp16) | output_len=128 | A100-40GB | enforce_eager=True")
print()
print("| seq  | batch | base tok/s | tq tok/s | tq/base | base lat (ms) | tq lat (ms) | base mem (GB) | tq mem (GB) |")
print("|-----:|------:|-----------:|---------:|--------:|--------------:|------------:|--------------:|------------:|")
for (seq, batch), row in sorted(rows.items()):
    b = row.get("baseline")
    t = row.get("tq")
    ratio = (
        f"{t['decode_tokens_per_s']/b['decode_tokens_per_s']:.2f}×"
        if (b and t) else "—"
    )
    print(f"| {seq:>4} | {batch:>5} | {fmt_tps(b):>10} | {fmt_tps(t):>8} | {ratio:>7} "
          f"| {fmt_lat(b):>13} | {fmt_lat(t):>11} | {fmt_mem(b):>13} | {fmt_mem(t):>11} |")

print()
print("Notes:")
print("- enforce_eager=True for both (A100 SM80 cannot torch.compile fp8e4nv ops; ")
print("  TQ requires kv_cache_dtype='fp8' which lowers to fp8e4nv otherwise).")
print("- baseline = stock vLLM, kv_cache_dtype=auto, FLASHINFER backend.")
print("- tq = vLLM + TurboQuant patches, kv_cache_dtype='fp8', CUSTOM backend.")
print("- OOM = TurboQuant workspace allocation exceeds 40 GB at this batch×seq.")
print("- per-trial median over 3 trials, 1 warmup, output_len=128.")
