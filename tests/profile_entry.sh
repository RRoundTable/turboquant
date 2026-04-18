#!/bin/bash
# HYP-042: torch.profiler attribution of one TQ vs baseline decode step.
# nsys/ncu unavailable in tq-hyp029:pr image; profiling-debug security profile
# rejects custom images. torch.profiler covers what we need (per-kernel CUDA
# time, RecordFunction ranges, chrome trace).
set -e

OUT_DIR=/workspace/shared/hyp042
mkdir -p "$OUT_DIR"

REPO=/workspace/shared/turboquant-bench
cd "$REPO"

export HF_HOME=/workspace/shared/hf-cache

# Apply current vLLM patches (carries vllm-project/vllm#39868).
SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cp "$REPO/docker/vllm_patches/v1/attention/backend.py"            "$SITE/v1/attention/backend.py"
cp "$REPO/docker/vllm_patches/v1/kv_cache_interface.py"           "$SITE/v1/kv_cache_interface.py"
cp "$REPO/docker/vllm_patches/v1/worker/gpu_model_runner.py"      "$SITE/v1/worker/gpu_model_runner.py"
cp "$REPO/docker/vllm_patches/model_executor/layers/attention/attention.py" \
   "$SITE/model_executor/layers/attention/attention.py"

pip install --quiet --no-deps -e "$REPO"
export TURBOQUANT_CSRC="$REPO/csrc"

run_profile() {
    local backend="$1"
    local trace_dir="$OUT_DIR/$backend"
    mkdir -p "$trace_dir"
    echo "=== profile $backend (trace -> $trace_dir) ==="
    VLLM_TORCH_PROFILER_DIR="$trace_dir" \
    python tests/profile_decode_step.py --backend "$backend" \
      2>&1 | tee "$OUT_DIR/${backend}.log"
}

run_profile baseline
run_profile tq

echo "=== DONE === artifacts under $OUT_DIR"
ls -la "$OUT_DIR"
