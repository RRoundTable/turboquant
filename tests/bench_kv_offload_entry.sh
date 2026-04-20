#!/bin/bash
set -e
OUT_DIR=/workspace/shared/hyp047
mkdir -p "$OUT_DIR"
REPO=/workspace/shared/turboquant-bench
cd "$REPO"
python tests/bench_kv_offload.py --out "$OUT_DIR/results.csv" \
  2>&1 | tee "$OUT_DIR/run.log"
echo "=== DONE ==="
ls -la "$OUT_DIR"
