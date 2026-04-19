#!/bin/bash
# FlashInfer-only baseline run for HYP-044 comparison.
# Forces attention_backend=FLASHINFER; same env as bench_entry.sh otherwise.
set -e

SEQ="${SEQ:?SEQ env required}"
BATCH="${BATCH:?BATCH env required}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT_LEN="${OUT_LEN:-128}"
GPU_MEM="${GPU_MEM:-0.85}"

OUT_DIR="${OUT_DIR:-/workspace/shared/bench_v5_vs_flashinfer}"
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

out="$OUT_DIR/flashinfer-s${SEQ}-b${BATCH}.json"
if [ -f "$out" ]; then
    echo "=== already complete: $out ==="
    exit 0
fi

echo "=== flashinfer (seq=$SEQ batch=$BATCH) ==="
python tests/bench_vllm_serve.py \
  --model "$MODEL" --backend flashinfer \
  --input-len "$SEQ" --batch "$BATCH" --output-len "$OUT_LEN" \
  --gpu-mem "$GPU_MEM" \
  --out "$out"

echo "=== DONE ==="
