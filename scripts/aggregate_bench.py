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
    cols = ["seq", "splits", "fi", "v5_ns", "v5_gather", "v5_paged",
            "paged/gather", "paged/fi"]
    widths = [6, 7, 8, 9, 10, 10, 13, 10]
    hdr = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))

    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        splits = d.get("num_splits", "?")
        flash = d.get("flashinfer", {})
        v5 = d.get("tq_v5_graph", {})
        v5s = d.get("tq_v5_split_graph", {})   # HYP-034 gather+split
        v5p = d.get("tq_v5_paged_split_graph", {})  # HYP-035 paged

        def p50(x):
            return x.get("p50_us", float("nan")) if isinstance(x, dict) else float("nan")

        v5s_us = p50(v5s)
        v5p_us = p50(v5p)
        fi_us = p50(flash)

        def ratio(num, den):
            if num != num or den != den or den == 0:
                return "n/a"
            return f"{num/den:.2f}x"

        row = [
            f"{seq:>6}",
            str(splits).rjust(7),
            _fmt_us(flash).rjust(8),
            _fmt_us(v5).rjust(9),
            _fmt_us(v5s).rjust(10),
            _fmt_us(v5p).rjust(10),
            ratio(v5p_us, v5s_us).rjust(13),
            ratio(v5p_us, fi_us).rjust(10),
        ]
        print("  ".join(row))

    print()
    # HYP-035 verdicts (paged-native vs gather-based split)
    verdicts = []
    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        v5s = d.get("tq_v5_split_graph", {}).get("p50_us", float("nan"))
        v5p = d.get("tq_v5_paged_split_graph", {}).get("p50_us", float("nan"))
        fi = d.get("flashinfer", {}).get("p50_us", float("nan"))
        paged_cos = d.get("paged_correctness", {}).get("cosine", float("nan"))

        if v5s == v5s and v5p == v5p and seq >= 1024:
            verdicts.append((f"paged/gather @ seq={seq}", v5p/v5s, 0.8, "≤ 0.80"))
        if v5p == v5p and fi == fi and seq == 4096:
            verdicts.append(("paged/flashinfer @ seq=4096", v5p/fi, 3.0, "≤ 3.0"))
        if paged_cos == paged_cos and seq >= 256:
            verdicts.append((f"paged correctness cos @ seq={seq}", paged_cos, 0.9999,
                             "≥ 0.9999"))

    if verdicts:
        print("HYP-035 prediction checks:")
        for name, actual, threshold, desc in verdicts:
            if "correctness" in name:
                passed = actual >= threshold
            else:
                passed = actual <= threshold
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name} = {actual:.4f} (target: {desc})")


if __name__ == "__main__":
    main()
