#!/bin/bash
# Install TurboQuant into an existing vLLM environment.
# Requirements: vLLM v0.19+ already installed, python on PATH, CUDA toolchain
# present (for kernel JIT compile on first inference).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== TurboQuant plugin install ==="
echo "Source: $HERE"

# 1. Apply vLLM source patches (carries vllm-project/vllm#39868: custom KV
#    cache page size hook). Without these the plugin cannot register a
#    page size that differs from 2 * block_size * num_kv_heads * head_size *
#    dtype_size, which caps TQ's effective compression at 2x instead of 3.2x.
SITE="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "Patching vLLM site at: $SITE"
cp "$HERE/vllm_patches/v1/attention/backend.py"            "$SITE/v1/attention/backend.py"
cp "$HERE/vllm_patches/v1/kv_cache_interface.py"           "$SITE/v1/kv_cache_interface.py"
cp "$HERE/vllm_patches/v1/worker/gpu_model_runner.py"      "$SITE/v1/worker/gpu_model_runner.py"
cp "$HERE/vllm_patches/model_executor/layers/attention/attention.py" \
   "$SITE/model_executor/layers/attention/attention.py"

# 2. Install the turboquant wheel. Auto-registers via the
#    `vllm.general_plugins` entry point on next vLLM import.
WHEEL="$(ls "$HERE"/turboquant-*.whl | head -1)"
echo "Installing: $WHEEL"
pip install --no-deps --force-reinstall "$WHEEL"

echo
echo "=== Done. To use: ==="
echo "  vllm serve <MODEL> --dtype float16 --enforce-eager \\"
echo "      --attention-backend CUSTOM --kv-cache-dtype fp8 \\"
echo "      --gpu-memory-utilization 0.85"
echo
echo "Constraints (A100 SM80): enforce_eager required; fp8 cache via TQ"
echo "internals, not vLLM's fp8e4nv path. Read /opt/tq-plugin/STATUS.md for the"
echo "full benchmark table and known limitations."
