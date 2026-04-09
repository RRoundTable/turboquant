#!/bin/bash
# Start vLLM serving with TurboQuant 4-bit KV cache.
#
# Usage:
#   ./scripts/serve.sh                          # Default: Qwen3-1.7B
#   MODEL=Qwen/Qwen3-8B ./scripts/serve.sh     # Custom model
#   MODEL=Qwen/Qwen3-8B PORT=8080 ./scripts/serve.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-4096}"

echo "=== TurboQuant vLLM Server ==="
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "KV cache: 4-bit TurboQuant (3.76× compression)"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --dtype float16 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --kv-cache-dtype turboquant \
    --port "${PORT}" \
    "$@"
