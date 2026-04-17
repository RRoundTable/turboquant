"""Aggregate per-seq_len JSONs from the HYP-033 Forge sweep and print a
comparison table. Reads every `seq-*.json` under the given directory.

Usage:
  uv run python scripts/aggregate_bench.py /workspace/shared/bench-v5-graph/
  uv run python scripts/aggregate_bench.py ./results/
"""

import json
import sys
from pathlib import Path


def _fmt_us(result: dict) -> str:
    if "error" in result:
        return f"ERR({result['error'][:20]})"
    return f"{result.get('p50_us', float('nan')):.1f}"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    files = sorted(root.glob("seq-*.json"),
                   key=lambda p: int(p.stem.split("-")[1]))
    if not files:
        print(f"No seq-*.json under {root}", file=sys.stderr)
        sys.exit(1)

    # Header
    cols = ["seq", "fp16_sdpa", "flashinfer", "tq_v4", "tq_v5",
            "v5/v4", "v5/fp16", "v5/fi", "corr_max_abs"]
    widths = [6, 10, 11, 9, 9, 7, 7, 7, 12]
    hdr = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))

    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        sdpa = d.get("fp16_sdpa", {})
        flash = d.get("flashinfer", {})
        v4 = d.get("tq_v4_graph", {})
        v5 = d.get("tq_v5_graph", {})
        corr = d.get("correctness", {}).get("max_abs", float("nan"))

        def p50(x):
            return x.get("p50_us", float("nan")) if isinstance(x, dict) else float("nan")

        v5_us = p50(v5)
        v4_us = p50(v4)
        sdpa_us = p50(sdpa)
        fi_us = p50(flash)

        def ratio(num, den):
            if num != num or den != den or den == 0:
                return "n/a"
            return f"{num/den:.2f}x"

        row = [
            f"{seq:>6}",
            _fmt_us(sdpa).rjust(10),
            _fmt_us(flash).rjust(11),
            _fmt_us(v4).rjust(9),
            _fmt_us(v5).rjust(9),
            ratio(v5_us, v4_us).rjust(7),
            ratio(v5_us, sdpa_us).rjust(7),
            ratio(v5_us, fi_us).rjust(7),
            f"{corr:.2e}".rjust(12),
        ]
        print("  ".join(row))

    print()
    # HYP-033 verdict
    verdicts = []
    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        v5 = d.get("tq_v5_graph", {}).get("p50_us", float("nan"))
        v4 = d.get("tq_v4_graph", {}).get("p50_us", float("nan"))
        sdpa = d.get("fp16_sdpa", {}).get("p50_us", float("nan"))
        fi = d.get("flashinfer", {}).get("p50_us", float("nan"))
        if v5 == v5 and v4 == v4 and seq >= 1024:
            verdicts.append(("v5/v4 @ seq=" + str(seq), v5/v4, 0.5, "≤ 0.5"))
        if v5 == v5 and sdpa == sdpa and seq == 4096:
            verdicts.append(("v5/fp16 @ seq=4096", v5/sdpa, 1.5, "≤ 1.5"))
        if v5 == v5 and fi == fi and seq == 4096:
            verdicts.append(("v5/flashinfer @ seq=4096", v5/fi, 2.0, "≤ 2.0"))

    if verdicts:
        print("HYP-033 prediction checks:")
        for name, actual, threshold, desc in verdicts:
            status = "PASS" if actual <= threshold else "FAIL"
            print(f"  [{status}] {name} = {actual:.2f}x (target: {desc})")


if __name__ == "__main__":
    main()
