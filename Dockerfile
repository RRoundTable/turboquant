FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install vLLM + FlashInfer (pulls compatible PyTorch)
RUN pip install --no-cache-dir vllm flashinfer-python

# Copy TurboQuant source
COPY . /opt/turboquant
WORKDIR /opt/turboquant

# Ensure build tools are up to date
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install TurboQuant as plugin (registers entry_points with vLLM)
RUN pip install --no-cache-dir .

# Note: CUDA kernel JIT compilation happens at first inference request.
# Cannot pre-compile during docker build (no GPU available).
# First request will take ~30s extra for JIT compilation.

EXPOSE 8000

# Default: vLLM OpenAI-compatible API server
# Override MODEL and flags via docker run args
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "Qwen/Qwen3-1.7B", "--dtype", "float16", "--gpu-memory-utilization", "0.9"]
