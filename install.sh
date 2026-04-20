#!/bin/bash
# Install TurboQuant into an existing vLLM environment.
#
# Two modes, autodetected:
#   1. Repo checkout (cwd contains setup.py + docker/vllm_patches/):
#        ./install.sh
#   2. Plugin OCI artifact (extracted from vllm-turboquant:plugin-* image to
#      a directory containing turboquant-*.whl + vllm_patches/):
#        TQ_PATCH_DIR=$EXTRACT_DIR/vllm_patches ./install.sh
#
# Requirements: vLLM v0.19 already installed, python on PATH,
# CUDA toolchain present (kernel JITs on first inference).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== TurboQuant plugin install ==="
echo "Source: $HERE"

# 1. Apply vLLM source patches. Required until vllm-project/vllm#39868 lands
#    upstream — without these the per-block page size is fixed at
#    2 * block_size * num_kv_heads * head_size * dtype_size, which caps TQ's
#    effective compression at ~2x instead of the published 3.2x.
PATCHES="${TQ_PATCH_DIR:-$HERE/docker/vllm_patches}"
if [ ! -d "$PATCHES" ]; then
    echo "ERROR: patch directory $PATCHES not found. Set TQ_PATCH_DIR=<path>."
    exit 1
fi
SITE="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "Patching vLLM site at: $SITE"
echo "Source patches:        $PATCHES"
cp "$PATCHES/v1/attention/backend.py"            "$SITE/v1/attention/backend.py"
cp "$PATCHES/v1/kv_cache_interface.py"           "$SITE/v1/kv_cache_interface.py"
cp "$PATCHES/v1/worker/gpu_model_runner.py"      "$SITE/v1/worker/gpu_model_runner.py"
cp "$PATCHES/model_executor/layers/attention/attention.py" \
   "$SITE/model_executor/layers/attention/attention.py"

# 2. Install the turboquant package. Auto-registers via the
#    `vllm.general_plugins` entry point on next vLLM import.
if ls "$HERE"/dist/turboquant-*.whl >/dev/null 2>&1; then
    WHEEL="$(ls "$HERE"/dist/turboquant-*.whl | head -1)"
    echo "Installing wheel: $WHEEL"
    pip install --no-deps --force-reinstall "$WHEEL"
elif ls "$HERE"/turboquant-*.whl >/dev/null 2>&1; then
    # Plugin OCI layout (wheel sits next to install.sh)
    WHEEL="$(ls "$HERE"/turboquant-*.whl | head -1)"
    echo "Installing wheel: $WHEEL"
    pip install --no-deps --force-reinstall "$WHEEL"
elif [ -f "$HERE/setup.py" ]; then
    echo "Installing from source: $HERE"
    pip install --no-deps --force-reinstall "$HERE"
else
    echo "ERROR: no wheel under $HERE/dist/ or $HERE/, no setup.py at $HERE"
    exit 1
fi

echo
echo "=== Done. To use: ==="
echo "  vllm serve <MODEL> --dtype float16 --enforce-eager \\"
echo "      --attention-backend CUSTOM --kv-cache-dtype fp8 \\"
echo "      --gpu-memory-utilization 0.85"
echo
echo "Constraints (A100 SM80): enforce_eager required; fp8 cache via TQ"
echo "internals, not vLLM's fp8e4nv path. See STATUS.md for the full"
echo "benchmark table and known limitations."
