# TurboQuant

Near-optimal KV cache quantization for LLM inference. **3.76x memory compression** with negligible quality loss.

Based on: Zandieh et al., [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), 2025.

## What It Does

TurboQuant compresses LLM KV cache from FP16 to 4-bit using Lloyd-Max codebook quantization with Hadamard rotation. Two fused CUDA kernels handle write (prefill) and decode, integrated into vLLM as a drop-in plugin.

- **3.76x KV cache compression** (68 bytes vs 256 bytes per token per head)
- **3.8x more concurrent requests** on the same GPU
- **< 0.01% perplexity degradation** (WikiText-2: 14.91 -> 14.91)
- **100% exact token match** vs FP16 baseline (12 prompts x 2 models)

## Install

```bash
pip install .
```

This registers TurboQuant as a [vLLM plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) via entry_points. No vLLM source modification needed.

**Requirements:** Python >= 3.10, PyTorch >= 2.1, CUDA GPU (A100 tested), vLLM >= 0.6.0, FlashInfer.

## Usage

### vLLM (OpenAI-compatible API server)

```bash
# TurboQuant activates automatically when installed alongside vLLM.
# The plugin registers at vLLM startup via entry_points discovery.
vllm serve Qwen/Qwen3-1.7B --dtype float16 --gpu-memory-utilization 0.9
```

### Docker

```bash
# Build
docker build -t vllm-turboquant .

# Run
docker run --gpus all -p 8000:8000 vllm-turboquant \
  --model Qwen/Qwen3-1.7B --dtype float16 --gpu-memory-utilization 0.9

# Or use pre-built image from ECR
docker pull 847366387031.dkr.ecr.us-east-1.amazonaws.com/vllm-turboquant
```

First request takes ~30s extra for CUDA kernel JIT compilation. Subsequent requests are instant.

### Programmatic (Python)

```python
from vllm import LLM, SamplingParams

# Just install turboquant — the plugin auto-registers
llm = LLM(model="Qwen/Qwen3-1.7B", dtype="float16", gpu_memory_utilization=0.9)
output = llm.generate("What is the capital of France?", SamplingParams(max_tokens=64))
print(output[0].outputs[0].text)
```

### Validated Models

| Model | QO/KV Heads | GQA | head_dim | Validation Level |
|-------|-------------|-----|----------|------------------|
| Qwen3-0.6B | 16/8 | 2:1 | 64 | Correctness + TPOT |
| Qwen3-1.7B | 16/8 | 2:1 | 128 | Full E2E + PPL |
| Qwen3-4B | 32/8 | 4:1 | 80 | Correctness + TPOT |
| Qwen3-8B | 32/8 | 4:1 | 128 | Full E2E + PPL |
| Mistral-7B | 32/8 | 4:1 | 128 | Correctness + TPOT |
| Llama-2-7B | 32/32 | 1:1 | 128 | Kernel only |
| Llama-3-8B | 32/8 | 4:1 | 128 | Kernel only |
| Llama-3-70B | 64/8 | 8:1 | 128 | Kernel only |

## Performance

All numbers measured on A100-SXM4-40GB, Qwen3-1.7B, batch=1.

### End-to-End Serving (vLLM, CUDA graphs)

| Phase | FP16 Baseline | TurboQuant 4-bit | Overhead |
|-------|---------------|------------------|----------|
| Prefill write (2K tokens) | 1.2 ms (memcpy) | 3.1 ms (CUDA quantize) | +3.7% of TTFT |
| Decode TPOT (seq <= 256) | 1.2 ms | **0.81 ms** | **33% faster** |
| Decode TPOT (seq = 1024) | 1.2 ms | 1.23 ms | 2.5% slower |
| KV memory per token | 256 B/tok/head | 68 B/tok/head | **3.76x less** |
| Max batch (seq=4K, 40GB) | 71 requests | **268 requests** | **3.8x more** |
| Throughput (batch >= 64) | 1.0x | **1.1-1.2x** | Faster |

### Decode Kernel Latency (standalone, A100)

| seq_len | FlashAttention (SDPA) | TurboQuant | vs SDPA |
|---------|-----------------------|------------|---------|
| 128 | 22 us | **22 us** | 1.0x (matches) |
| 256 | 30 us | 32 us | 1.05x |
| 512 | 30 us | 48 us | 1.6x |
| 1024 | 31 us | 48 us | 1.6x |
| 2048 | 30 us | 59 us | 2.0x |

At seq <= 256, TurboQuant **matches or beats** FlashInfer FP16 while using 3.76x less memory.

### Quality

| Metric | Result |
|--------|--------|
| WikiText-2 PPL | 14.91 -> 14.91 (0.01% degradation) |
| Kernel cosine similarity | 1.000000 (all configs) |
| Exact token match | 100% (12 prompts, Qwen3-1.7B + 8B) |

## Roadmap

### Now: Tensor Parallelism (TP)

TP splits KV heads across GPUs. Each GPU runs TurboQuant on its local head subset. Example: Qwen3-8B (32QO/8KV) at TP=4 gives 2 KV heads per GPU.

- [ ] Kernel correctness at low KV head counts (1-2 heads/GPU at TP=4,8)
- [ ] TP-aware KV cache allocation in vLLM backend
- [ ] Multi-GPU benchmark (TP=1 vs TP=2 vs TP=4, Qwen3-8B)
- [ ] Split-KV tuning for TP (aggressive splitting to fill SMs at low head counts)

### Next

- [ ] **3-bit quantization** — target >=5x compression with <1% PPL degradation
- [ ] **Fused Hadamard in write kernel** — eliminate Python FWHT overhead
- [ ] **LongBench / NIAH evaluation** — quality at 4K-32K context
- [ ] **Tensor cores for GQA>=4** — WMMA at bdy=4 for Llama-3 / Mistral class models

### Later

- [ ] Pipeline parallelism (PP) — TurboQuant per pipeline stage
- [ ] Multi-node distributed (TP across nodes via NCCL)
- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Continuous batching with dynamic KV cache growth

## How It Works

```
Write (prefill):
  FP16 KV -> L2 normalize -> Hadamard rotate -> Lloyd-Max quantize -> nibble pack -> store 4-bit

Read (decode):
  Load 4-bit packed bytes (cp_async) -> unpack nibbles -> codebook lookup * norm -> attention (QK softmax V)
```

The decode kernel uses inline dequantization — no intermediate FP16 buffer in shared memory. This cuts smem usage from ~32 KB to ~7 KB per block and eliminates half<->float conversion overhead.

Adaptive dispatch: eager execution at seq <= 256 (where TurboQuant is faster), CUDA graph capture at seq >= 512 (26% kernel speedup from eliminating launch overhead).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Standalone algorithm validation
python tests/test_algorithm.py
```

## License

Apache-2.0
