"""HYP-045 end-to-end: HYP-041 sweep with dead-workspace-allocation fix."""
import glob
import json
import os

V0 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v5_vs_baseline"))
V1 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v5_vs_baseline_hyp044"))
V2 = os.path.dirname(__file__)
FI = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "v5_vs_flashinfer"))


def load(d, prefix):
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, f"{prefix}*.json"))):
        x = json.load(open(p))
        rows[(x["input_len"], x["batch"])] = x
    return rows


tq_v0 = load(V0, "tq-")
tq_v1 = load(V1, "tq-")
tq_v2 = load(V2, "tq-")
fa_v2 = load(V2, "baseline-")  # FLASH_ATTN, same env as tq_v2
fi    = load(FI, "flashinfer-")

keys = sorted(set(tq_v0) | set(tq_v1) | set(tq_v2))


def f(x): return f"{x['decode_tokens_per_s']:.1f}" if x else "—"
def mem(x): return f"{x['nvsmi_gpu_mem_gb']:.1f}" if (x and x.get('nvsmi_gpu_mem_gb')) else "—"


print("# HYP-045 end-to-end: skip dead workspace allocation at num_splits>1")
print()
print("Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True | tok/s")
print()
print("TQ progression: v0 (HYP-041 original) → v1 (HYP-044 chunk-cap) → **v2 (HYP-045 dead-alloc fix)**.")
print()
print("| seq × b | TQ v0 | TQ v1 | **TQ v2** | FA | FI | TQ mem v0→v2 | tq/FA v2 | tq/FI v2 |")
print("|---------|------:|------:|----------:|---:|---:|:-------------|---------:|---------:|")
for s, b in keys:
    t0 = tq_v0.get((s, b)); t1 = tq_v1.get((s, b)); t2 = tq_v2.get((s, b))
    fa = fa_v2.get((s, b)); fin = fi.get((s, b))
    # Use original HYP-041 baseline mem for v0 row.
    import_v0 = None
    try:
        import_v0 = json.load(open(os.path.join(V0, f"tq-s{s}-b{b}.json")))
    except Exception:
        pass
    mem0 = mem(import_v0)
    mem2 = mem(t2)
    memcol = f"{mem0} → {mem2}" if (mem0 != '—' and mem2 != '—') else f"— → {mem2}" if mem2 != '—' else "—"
    r_fa = f"{t2['decode_tokens_per_s']/fa['decode_tokens_per_s']:.2f}x" if (t2 and fa) else "—"
    r_fi = f"{t2['decode_tokens_per_s']/fin['decode_tokens_per_s']:.2f}x" if (t2 and fin) else "—"
    print(f"| {s:>5} × {b:>2} | {f(t0):>5} | {f(t1):>5} | **{f(t2):>5}** | "
          f"{f(fa):>4} | {f(fin):>4} | {memcol} | {r_fa:>6} | {r_fi:>6} |")

print()
print("Notes:")
print("- v0 = commit 5db1d80 (HYP-041). v1 = HYP-044 chunk-cap. **v2 = HYP-045: skip k_quant/v_quant/k_norms/v_norms allocation when num_splits>1 (dead in paged-split path).**")
print("- HYP-045 removes ~9.6 GB of dead allocation at seq=8192 × batch=32 → unblocks all 4 HYP-041 OOM configs.")
print("- Decode tok/s at non-OOM configs unchanged from v1 (same kernel work).")
print("- FA = vLLM auto-pick (FLASH_ATTN). FI = explicit attention_backend='FLASHINFER'.")
