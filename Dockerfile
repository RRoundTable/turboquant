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

# Install TurboQuant as plugin (registers entry_points with vLLM)
RUN pip install --no-cache-dir .

# Pre-compile CUDA kernels to avoid cold-start JIT latency
# This compiles for the build GPU — will recompile at runtime if target GPU differs
RUN python -c "\
try:\
    from turboquant.decode_kernel_v4 import _get_module;\
    _get_module();\
    print('Decode kernel compiled');\
except Exception as e:\
    print(f'Decode kernel JIT skipped (no GPU during build): {e}');\
"

EXPOSE 8000

# Default: vLLM OpenAI-compatible API server
# Override MODEL and flags via docker run args
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "Qwen/Qwen3-1.7B", "--dtype", "float16", "--gpu-memory-utilization", "0.9"]
