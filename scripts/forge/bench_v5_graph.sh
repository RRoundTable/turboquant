#!/usr/bin/env bash
# Forge job entrypoint for HYP-033 v5 graph-safe benchmark.
# Requires env vars: SEQ_LEN, REPO_URL (or a staged checkout at /workspace/shared/turboquant).
set -euo pipefail

SEQ_LEN="${SEQ_LEN:?SEQ_LEN must be set}"
RESULTS_DIR="/workspace/shared/bench-v5-graph"
REPO_DIR="/workspace/turboquant"

mkdir -p "${RESULTS_DIR}"

# Pull repo if not already staged. Jobs get a fresh container every time.
if [ ! -d "${REPO_DIR}" ]; then
    if [ -d "/workspace/shared/turboquant" ]; then
        cp -r /workspace/shared/turboquant "${REPO_DIR}"
    elif [ -n "${REPO_URL:-}" ]; then
        git clone "${REPO_URL}" "${REPO_DIR}"
    else
        echo "No repo source: stage /workspace/shared/turboquant or set REPO_URL" >&2
        exit 1
    fi
fi

cd "${REPO_DIR}"

# Install (idempotent). Use uv if available, else pip.
if command -v uv >/dev/null 2>&1; then
    uv pip install -e '.[dev]' flashinfer 2>&1 | tail -5
    RUN=(uv run python)
else
    pip install -e '.[dev]' flashinfer 2>&1 | tail -5
    RUN=(python)
fi

OUT="${RESULTS_DIR}/seq-${SEQ_LEN}.json"
echo "[entrypoint] running bench_v5_graph.py --seq-len ${SEQ_LEN} --out ${OUT}"
"${RUN[@]}" -m tests.bench_v5_graph --seq-len "${SEQ_LEN}" --out "${OUT}"

echo "[entrypoint] done. Artifact:"
ls -la "${OUT}"
