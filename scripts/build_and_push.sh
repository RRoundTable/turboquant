#!/bin/bash
# Build TurboQuant + vLLM Docker image and push to AWS ECR.
#
# Usage:
#   ECR_URI=123456789.dkr.ecr.ap-northeast-2.amazonaws.com/turboquant ./scripts/build_and_push.sh
#   ECR_URI=... TAG=v0.1.0 ./scripts/build_and_push.sh

set -euo pipefail

ECR_URI="${ECR_URI:-847366387031.dkr.ecr.us-east-1.amazonaws.com/vllm-turboquant}"
TAG="${TAG:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "=== Building turboquant-vllm:${TAG} ==="
docker build -t turboquant-vllm:"${TAG}" .

echo "=== Tagging for ECR ==="
docker tag turboquant-vllm:"${TAG}" "${ECR_URI}:${TAG}"

echo "=== Logging in to ECR ==="
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI%%/*}"

echo "=== Pushing to ${ECR_URI}:${TAG} ==="
docker push "${ECR_URI}:${TAG}"

echo "=== Done ==="
echo "Pull with: docker pull ${ECR_URI}:${TAG}"
echo "Run with:  docker run --gpus all -p 8000:8000 ${ECR_URI}:${TAG} --model Qwen/Qwen3-8B --kv-cache-dtype turboquant"
