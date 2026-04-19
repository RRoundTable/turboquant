#!/bin/bash
# Runs baseline + TurboQuant vLLM benchmarks for one (seq, batch) config.
# Expects: SEQ, BATCH env vars. Outputs JSON to /workspace/shared/bench_v5_vs_baseline/.
set -e

SEQ="${SEQ:?SEQ env required}"
BATCH="${BATCH:?BATCH env required}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT_LEN="${OUT_LEN:-128}"
GPU_MEM="${GPU_MEM:-0.85}"

OUT_DIR="${OUT_DIR:-/workspace/shared/bench_v5_vs_baseline}"
mkdir -p "$OUT_DIR"

REPO=/workspace/shared/turboquant-bench
cd "$REPO"

export HF_HOME=/workspace/shared/hf-cache
mkdir -p "$HF_HOME"

# Refresh vLLM patches to match current code (image's patches may be stale).
SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "Applying patches to $SITE"
cp -v "$REPO/docker/vllm_patches/v1/attention/backend.py" "$SITE/v1/attention/backend.py"
cp -v "$REPO/docker/vllm_patches/v1/kv_cache_interface.py" "$SITE/v1/kv_cache_interface.py"
cp -v "$REPO/docker/vllm_patches/v1/worker/gpu_model_runner.py" "$SITE/v1/worker/gpu_model_runner.py"
cp -v "$REPO/docker/vllm_patches/model_executor/layers/attention/attention.py" \
      "$SITE/model_executor/layers/attention/attention.py"

# Install turboquant so its vllm plugin registers.
pip install --quiet --no-deps -e "$REPO"
export TURBOQUANT_CSRC="$REPO/csrc"

run_bench() {
    local backend="$1"
    local out="$OUT_DIR/${backend}-s${SEQ}-b${BATCH}.json"
    if [ -f "$out" ]; then
        echo "=== $backend already complete, skipping ==="
        return 0
    fi
    echo "=== $backend (seq=$SEQ batch=$BATCH) ==="
    python tests/bench_vllm_serve.py \
      --model "$MODEL" --backend "$backend" \
      --input-len "$SEQ" --batch "$BATCH" --output-len "$OUT_LEN" \
      --gpu-mem "$GPU_MEM" \
      --out "$out"
}

run_bench baseline
run_bench tq

echo "=== DONE ==="
