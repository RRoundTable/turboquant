#!/bin/bash
# HYP-044 microbench entrypoint: batch sweep of v5_paged kernel under two
# split-K heuristics. Standalone (no vLLM stack).
set -e

OUT_DIR=/workspace/shared/hyp044
mkdir -p "$OUT_DIR"

REPO=/workspace/shared/turboquant-bench
cd "$REPO"

pip install --quiet --no-deps -e "$REPO"
export TURBOQUANT_CSRC="$REPO/csrc"

python tests/bench_hyp044.py --seq-len 8192 --out "$OUT_DIR/results.csv" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "=== DONE ==="
ls -la "$OUT_DIR"
