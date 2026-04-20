#!/bin/bash
# Serving-mode bench using UPSTREAM vLLM turboquant (--kv-cache-dtype turboquant_*).
# Replaces the stock vllm 0.19.0 in the image with v0.19.2rc0 which is the first
# tagged release that carries upstream's vllm.model_executor.layers.quantization.turboquant.
set -e

SEQ="${SEQ:?SEQ env required}"
OUT_LEN="${OUT_LEN:-128}"
CONCURRENCY="${CONCURRENCY:?CONCURRENCY env required}"
NUM_PROMPTS="${NUM_PROMPTS:?NUM_PROMPTS env required}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
GPU_MEM="${GPU_MEM:-0.85}"
OUT_DIR="${OUT_DIR:-/workspace/shared/bench_serve_upstream}"
PORT=${PORT:-8000}
HOST=127.0.0.1
KV_DTYPE="${KV_DTYPE:-turboquant_4bit_nc}"

mkdir -p "$OUT_DIR"

# The tq-upstream-nightly:v2 Forge image already has vllm nightly + turboquant.
# Just verify it's importable before serving.
python3 -c "
import vllm
print('vllm version:', vllm.__version__)
from vllm.model_executor.layers.quantization.turboquant import TurboQuantConfig
print('upstream TurboQuantConfig present:', TurboQuantConfig)
"

export HF_HOME=/workspace/shared/hf-cache

MAX_LEN=$(( SEQ + OUT_LEN + 16 ))
# No attention-backend override: let vllm auto-pick (usually FLASHINFER or FA2).
# No --enforce-eager: upstream path uses Triton kernels, runs on tensor cores.
COMMON=("$MODEL" --dtype float16
        --gpu-memory-utilization "$GPU_MEM"
        --max-model-len "$MAX_LEN"
        --disable-log-stats
        --port "$PORT"
        --kv-cache-dtype "$KV_DTYPE")

TAG="upstream-s${SEQ}-c${CONCURRENCY}"
LOG_FILE="$OUT_DIR/${TAG}.server.log"
RESULT_FILE="$OUT_DIR/${TAG}.json"

if [ -f "$RESULT_FILE" ]; then
  echo "=== $TAG already complete, skipping ==="
  exit 0
fi

echo "=== launching: vllm serve ${COMMON[@]} ==="
vllm serve "${COMMON[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

cleanup() {
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
  echo "SERVER NOT READY within 900s"
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
