#!/bin/bash
# Serving-mode bench: starts vllm serve + runs `vllm bench serve` against it.
# Env: SEQ, OUT_LEN, CONCURRENCY, NUM_PROMPTS, BACKEND (fa|fi|tq)
set -e

SEQ="${SEQ:?SEQ env required}"
OUT_LEN="${OUT_LEN:-128}"
CONCURRENCY="${CONCURRENCY:?CONCURRENCY env required}"
NUM_PROMPTS="${NUM_PROMPTS:?NUM_PROMPTS env required}"
BACKEND="${BACKEND:?BACKEND env required}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU_MEM="${GPU_MEM:-0.85}"
OUT_DIR="${OUT_DIR:-/workspace/shared/bench_serve}"
PORT=${PORT:-8000}
HOST=127.0.0.1

mkdir -p "$OUT_DIR"

REPO=/workspace/shared/turboquant-bench
cd "$REPO"

export HF_HOME=/workspace/shared/hf-cache

SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cp "$REPO/docker/vllm_patches/v1/attention/backend.py"            "$SITE/v1/attention/backend.py"
cp "$REPO/docker/vllm_patches/v1/kv_cache_interface.py"           "$SITE/v1/kv_cache_interface.py"
cp "$REPO/docker/vllm_patches/v1/worker/gpu_model_runner.py"      "$SITE/v1/worker/gpu_model_runner.py"
cp "$REPO/docker/vllm_patches/model_executor/layers/attention/attention.py" \
   "$SITE/model_executor/layers/attention/attention.py"

pip install --quiet --no-deps -e "$REPO"
export TURBOQUANT_CSRC="$REPO/csrc"

MAX_LEN=$(( SEQ + OUT_LEN + 16 ))
COMMON=("$MODEL" --dtype float16
        --gpu-memory-utilization "$GPU_MEM"
        --max-model-len "$MAX_LEN"
        --enforce-eager
        --disable-log-stats
        --port "$PORT")

case "$BACKEND" in
  fa)  EXTRA=() ;;                                         # vLLM auto -> FA
  fi)  EXTRA=(--attention-backend FLASHINFER) ;;
  tq)  EXTRA=(--attention-backend CUSTOM --kv-cache-dtype fp8) ;;
  *)   echo "bad BACKEND=$BACKEND"; exit 2 ;;
esac

TAG="${BACKEND}-s${SEQ}-c${CONCURRENCY}"
LOG_FILE="$OUT_DIR/${TAG}.server.log"
RESULT_FILE="$OUT_DIR/${TAG}.json"

if [ -f "$RESULT_FILE" ]; then
  echo "=== $TAG already complete, skipping ==="
  exit 0
fi

echo "=== launching server: vllm serve ${COMMON[@]} ${EXTRA[@]} ==="
vllm serve "${COMMON[@]}" "${EXTRA[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

cleanup() {
  echo "=== cleanup: kill $SERVER_PID ==="
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 3
  kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== waiting for /health (up to 900s) ==="
for i in $(seq 1 900); do
  if curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo "server up after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER DIED during startup. Last 80 lines:"
    tail -80 "$LOG_FILE"
    exit 1
  fi
  sleep 1
done

if ! curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
  echo "SERVER DID NOT BECOME READY within 900s"
  tail -80 "$LOG_FILE"
  exit 1
fi

echo "=== running vllm bench serve ==="
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len "$SEQ" \
  --random-output-len "$OUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --ignore-eos \
  --save-result --result-dir "$OUT_DIR" \
  --result-filename "${TAG}.json" \
  --label "$TAG" \
  --trust-remote-code \
  2>&1 | tee "$OUT_DIR/${TAG}.bench.log"

echo "=== DONE: $RESULT_FILE ==="
