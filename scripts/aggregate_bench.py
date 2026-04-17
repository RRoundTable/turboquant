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
    cols = ["seq", "splits", "fp16_sdpa", "flashinfer", "tq_v4", "tq_v5", "tq_v5_split",
            "split/v5", "split/fi", "corr"]
    widths = [6, 7, 10, 11, 9, 9, 11, 9, 9, 10]
    hdr = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))

    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        splits = d.get("num_splits", "?")
        sdpa = d.get("fp16_sdpa", {})
        flash = d.get("flashinfer", {})
        v4 = d.get("tq_v4_graph", {})
        v5 = d.get("tq_v5_graph", {})
        v5s = d.get("tq_v5_split_graph", {})
        corr = d.get("correctness", {}).get("max_abs", float("nan"))

        def p50(x):
            return x.get("p50_us", float("nan")) if isinstance(x, dict) else float("nan")

        v5_us = p50(v5)
        v5s_us = p50(v5s)
        fi_us = p50(flash)

        def ratio(num, den):
            if num != num or den != den or den == 0:
                return "n/a"
            return f"{num/den:.2f}x"

        row = [
            f"{seq:>6}",
            str(splits).rjust(7),
            _fmt_us(sdpa).rjust(10),
            _fmt_us(flash).rjust(11),
            _fmt_us(v4).rjust(9),
            _fmt_us(v5).rjust(9),
            _fmt_us(v5s).rjust(11),
            ratio(v5s_us, v5_us).rjust(9),
            ratio(v5s_us, fi_us).rjust(9),
            f"{corr:.1e}".rjust(10),
        ]
        print("  ".join(row))

    print()
    # HYP-034 verdicts (focus on split-KV wins over non-split + FI gap)
    verdicts = []
    for f in files:
        d = json.loads(f.read_text())
        seq = d["seq_len"]
        v5 = d.get("tq_v5_graph", {}).get("p50_us", float("nan"))
        v5s = d.get("tq_v5_split_graph", {}).get("p50_us", float("nan"))
        fi = d.get("flashinfer", {}).get("p50_us", float("nan"))
        split_cos = d.get("split_correctness", {}).get("cosine", float("nan"))

        if v5 == v5 and v5s == v5s and seq >= 1024:
            verdicts.append((f"v5_split/v5_nosplit @ seq={seq}", v5s/v5, 0.15,
                             "≤ 0.15"))
        if v5s == v5s and fi == fi and seq == 4096:
            verdicts.append(("v5_split/flashinfer @ seq=4096", v5s/fi, 2.5,
                             "≤ 2.5"))
        if split_cos == split_cos and seq >= 256:
            verdicts.append((f"split correctness cos @ seq={seq}", split_cos, 0.9999,
                             "≥ 0.9999"))

    if verdicts:
        print("HYP-034 prediction checks:")
        for name, actual, threshold, desc in verdicts:
            if "correctness" in name:
                passed = actual >= threshold
            else:
                passed = actual <= threshold
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name} = {actual:.4f} (target: {desc})")


if __name__ == "__main__":
    main()
