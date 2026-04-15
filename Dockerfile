FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir vllm flashinfer-python pytest

# Apply vLLM patches — adds the custom_page_size seam proposed in
# vllm-project/vllm PR (see docker/vllm_patches/README.md).
COPY docker/vllm_patches /opt/vllm_patches
RUN set -e; \
    SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"); \
    cp /opt/vllm_patches/v1/attention/backend.py       $SITE/v1/attention/backend.py; \
    cp /opt/vllm_patches/v1/kv_cache_interface.py      $SITE/v1/kv_cache_interface.py; \
    cp /opt/vllm_patches/v1/worker/gpu_model_runner.py $SITE/v1/worker/gpu_model_runner.py; \
    cp /opt/vllm_patches/v1/worker/gpu/attn_utils.py   $SITE/v1/worker/gpu/attn_utils.py; \
    cp /opt/vllm_patches/model_executor/layers/attention/attention.py \
       $SITE/model_executor/layers/attention/attention.py

COPY . /opt/turboquant
WORKDIR /opt/turboquant
RUN pip install --no-cache-dir .

# csrc/ is not a Python module; point the JIT compiler at the copy
# kept at /opt/turboquant/csrc (data_files install is site-layout dependent).
ENV TURBOQUANT_CSRC=/opt/turboquant/csrc

# Kernels JIT-compile on first inference (~30s), since no GPU at build time.

EXPOSE 8000
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "Qwen/Qwen3-8B", "--dtype", "float16", \
     "--attention-backend", "CUSTOM", "--kv-cache-dtype", "fp8", \
     "--gpu-memory-utilization", "0.9"]
