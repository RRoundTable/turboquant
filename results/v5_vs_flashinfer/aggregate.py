"""3-way decode-tok/s comparison: FLASHINFER vs FLASH_ATTN (vLLM auto) vs TurboQuant.

- FLASHINFER: explicit attention_backend='FLASHINFER' (this dir)
- FLASH_ATTN: vLLM auto-pick on v0.19/A100 = FLASH_ATTN (results/v5_vs_baseline_hyp044/baseline-*.json)
- TurboQuant: HYP-044 patched (results/v5_vs_baseline_hyp044/tq-*.json)
"""
import glob
import json
import os

D_FI = os.path.dirname(__file__)
D_HYP044 = os.path.normpath(os.path.join(D_FI, "..", "v5_vs_baseline_hyp044"))


def load_dir(d, prefix=None):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if prefix and not os.path.basename(p).startswith(prefix):
            continue
        x = json.load(open(p))
        rows[(x["input_len"], x["batch"])] = x
    return rows


fi = load_dir(D_FI, prefix="flashinfer-")
fa = load_dir(D_HYP044, prefix="baseline-")  # vLLM auto = FLASH_ATTN
tq = load_dir(D_HYP044, prefix="tq-")

keys = sorted(set(fi) | set(fa) | set(tq))


def f(x): return f"{x['decode_tokens_per_s']:.1f}" if x else "—"


print("# TurboQuant (HYP-044) vs FlashInfer vs FlashAttention (vLLM auto)")
print()
print("Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True | tok/s")
print()
print("| seq  | b  | FlashInfer | FlashAttn | TurboQuant | tq/FI | tq/FA | FI/FA |")
print("|-----:|---:|-----------:|----------:|-----------:|------:|------:|------:|")
for s, b in keys:
    rfi, rfa, rtq = fi.get((s, b)), fa.get((s, b)), tq.get((s, b))
    rfm = lambda a, c: f"{a['decode_tokens_per_s']/c['decode_tokens_per_s']:.2f}x" if (a and c) else "—"
    print(f"| {s:>4} | {b:>2} | {f(rfi):>10} | {f(rfa):>9} | {f(rtq):>10} | "
          f"{rfm(rtq, rfi):>5} | {rfm(rtq, rfa):>5} | {rfm(rfi, rfa):>5} |")

print()
print("Notes:")
print("- FlashInfer = explicit attention_backend='FLASHINFER'.")
print("- FlashAttn = vLLM v0.19 auto-selects FLASH_ATTN at this seq/dtype on A100.")
print("- TurboQuant = HYP-044 patched (chunk_size cap = 256).")
print("- 'tq/FI' and 'tq/FA' are TurboQuant÷baseline tok/s ratios (>1 = TQ faster).")
print("- All eager — A100 SM80 cannot torch.compile fp8e4nv (HYP-041 constraint).")
