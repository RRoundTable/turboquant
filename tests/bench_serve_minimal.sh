#!/bin/bash
# Minimal entry: skip the runtime patch-overlay and -e pip install that
# bench_serve_entry.sh does. Uses whatever turboquant + vllm the image
# baked in. Lets us isolate image perf vs harness-induced divergence.
set -e

SEQ="${SEQ:?SEQ}"; CONC="${CONCURRENCY:?CONC}"; OUT_LEN="${OUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"; MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU_MEM="${GPU_MEM:-0.85}"; OUT_DIR="${OUT_DIR:?OUT_DIR}"; PORT=${PORT:-8000}
HOST=127.0.0.1; TAG="${BACKEND:-tq}-s${SEQ}-c${CONC}"
mkdir -p "$OUT_DIR"

export HF_HOME=/workspace/shared/hf-cache

MAX_LEN=$(( SEQ + OUT_LEN + 16 ))
COMMON=("$MODEL" --dtype float16 --gpu-memory-utilization "$GPU_MEM"
        --max-model-len "$MAX_LEN" --enforce-eager --disable-log-stats --port "$PORT")
case "${BACKEND:-tq}" in
  tq) EXTRA=(--attention-backend CUSTOM --kv-cache-dtype fp8) ;;
  fa) EXTRA=() ;;
esac

RESULT="$OUT_DIR/${TAG}.json"
[ -f "$RESULT" ] && { echo "=== $TAG already complete, skipping ==="; exit 0; }

echo "=== launching: vllm serve ${COMMON[@]} ${EXTRA[@]} ==="
vllm serve "${COMMON[@]}" "${EXTRA[@]}" > "$OUT_DIR/${TAG}.server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 3; kill -KILL "$SERVER_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 900); do
  curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1 && { echo "up after ${i}s"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED"; tail -80 "$OUT_DIR/${TAG}.server.log"; exit 1; }
  sleep 1
done
curl -sf "http://${HOST}:${PORT}/health" > /dev/null || { echo "NOT READY"; exit 1; }

vllm bench serve --backend vllm --model "$MODEL" --host "$HOST" --port "$PORT" \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len "$SEQ" --random-output-len "$OUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" --ignore-eos \
  --save-result --result-dir "$OUT_DIR" --result-filename "${TAG}.json" \
  --label "$TAG" --trust-remote-code
echo "=== DONE: $RESULT ==="
