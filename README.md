# TurboQuant

A CUDA-specialized vLLM attention backend for long-context serving. Hadamard
rotation + 4-bit Lloyd-Max KV-cache quantization with a hand-written CUDA
decode kernel that fuses dequant into attention.

Paper: Zandieh et al., [TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), 2026.

## Related: `vllm` already has a TurboQuant module

Upstream vLLM merged a TurboQuant implementation on 2026-04-15
([module docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/)),
available from `v0.19.2rc0` onward. It uses **Triton kernels** and plugs in as a
`QuantizationConfig` (`--kv-cache-dtype turboquant_4bit_nc`). It covers the
short-to-medium-context case very well.

**This repo is a different point on the design curve**: hand-written
**CUDA** kernels that win the long-context / high-batch regime where the
decode-kernel hot loop and the scheduler's cache budget start to matter.
See the [benchmarks](docs/BENCHMARKS.md) for when to pick which.

## Headline numbers (Qwen3-8B on A100-40GB, `vllm bench serve`)

| regime | winner | result |
|---|---|---|
| short ctx (≤ 2k), chat-style | **upstream TurboQuant** | TPOT 19 ms, matches FA |
| medium ctx (4–8k), batch ≥ 8 | stock **FA / FI** | TurboQuant pays decode tax |
| **long ctx (≥ 16k), batch ≥ 4** | **ours** | **1.89× FA throughput @ s32768×c8** |

At s32768 × c8: **3.2 s TTFT vs FA's 31.8 s** (10× faster first-token,
driven by avoided preemption; see
[BENCHMARKS.md](docs/BENCHMARKS.md#why-ours-wins-ttft-at-long-ctx)).

| config | FA tok/s | FI tok/s | ours tok/s | upstream tok/s |
|---|---:|---:|---:|---:|
| s1024 × c8 | 235 | 244 | 198 | **253** |
| s8192 × c8 | **112** | **116** | 106 | 82 |
| s16384 × c8 | 60 | 64 | **69** | 44 |
| s32768 × c4 | 24 | 26 | **36** | 19 |
| s32768 × c8 | 24 | 26 | **45** | 20 |

Full tables + methodology + raw JSON: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

TurboQuant requires both a Python package (the plugin + CUDA kernels) **and**
four small vLLM source patches that wire up the per-backend KV-cache page-size
hook from [vllm-project/vllm#39868](https://github.com/vllm-project/vllm/pull/39868).
Until that PR merges upstream, the patches must be applied locally; once it
merges, plain `pip install turboquant` will be enough.

### Option 1 — turn-key Docker image (recommended)

```bash
docker pull 847366387031.dkr.ecr.ap-northeast-2.amazonaws.com/vllm-turboquant:latest

docker run --gpus all -p 8000:8000 \
  847366387031.dkr.ecr.ap-northeast-2.amazonaws.com/vllm-turboquant:latest \
  --model Qwen/Qwen3-8B --dtype float16 --enforce-eager \
  --attention-backend CUSTOM --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85
```

Bundles vLLM v0.19, the patches, and the plugin. Nothing to apply.

### Option 2 — plugin overlay onto your own vLLM image

For users who already maintain their own vLLM image. Pull the 504 KB plugin
artifact and run `install.sh` against the existing vLLM install:

```bash
# Pull the plugin artifact (just wheel + patches + install.sh, no vllm)
docker create --name tq 847366387031.dkr.ecr.ap-northeast-2.amazonaws.com/vllm-turboquant:plugin-latest
docker cp tq:/opt/tq-plugin ./tq-plugin && docker rm tq

# Inside your container (or wherever vLLM is installed):
TQ_PATCH_DIR=./tq-plugin/vllm_patches ./tq-plugin/install.sh
```

Or as a multi-stage Dockerfile:

```dockerfile
FROM 847366387031.dkr.ecr.ap-northeast-2.amazonaws.com/vllm-turboquant:plugin-latest AS tq
FROM your-vllm-image
COPY --from=tq /opt/tq-plugin /opt/tq-plugin
RUN TQ_PATCH_DIR=/opt/tq-plugin/vllm_patches /opt/tq-plugin/install.sh
```

### Option 3 — install from source

```bash
git clone https://github.com/RRoundTable/turboquant.git
cd turboquant
./install.sh    # applies patches from docker/vllm_patches + pip-installs the package
```

`install.sh` patches your vLLM site-packages (run-once) and installs the
turboquant package, which auto-registers via vLLM's `vllm.general_plugins`
entry point. To run without applying the patches, see "Without patches" below.

### Without patches (for evaluation only)

`pip install turboquant` alone *registers* the backend (the
`vllm.general_plugins` entry-point machinery is stock), but the page-size
override hook isn't wired in stock vLLM. The plugin will run with effective
**~2× compression** instead of the published 3.2×, and may hit a
layout-mismatch depending on vLLM version. The patches are required for the
full 3.2× story until PR #39868 lands.

**Requirements:** Python >= 3.10, PyTorch >= 2.1, CUDA GPU (A100 tested),
vLLM v0.19, FlashInfer.

## Usage

```bash
vllm serve Qwen/Qwen3-8B --dtype float16 --enforce-eager \
  --attention-backend CUSTOM --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85
```

`--attention-backend CUSTOM --kv-cache-dtype fp8` are required to activate
TurboQuant. `--enforce-eager` is required on A100 (SM80) — vLLM lowers the
fp8 path to `fp8e4nv` which doesn't compile on SM80; H100 and later don't
need it. First request takes ~30s extra for CUDA kernel JIT compilation;
subsequent requests are instant.

### Programmatic (Python)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    dtype="float16",
    enforce_eager=True,
    attention_backend="CUSTOM",
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.85,
)
out = llm.generate("What is the capital of France?", SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
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

## When to use this vs upstream vllm.turboquant

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full 4-way comparison
(FlashAttention, FlashInfer, this repo, upstream vLLM TurboQuant) across
6 configs. Decision in one table:

| your workload | use |
|---|---|
| chat, prompts mostly ≤ 2k tokens | **upstream** (`--kv-cache-dtype turboquant_4bit_nc` in vLLM ≥ 0.19.2rc0) |
| general serving, 4–8k context, no memory pressure | **stock FA / FI** — simpler, faster |
| **long-doc / RAG / agentic, ≥ 16k context or batch ≥ 4 at 32k** | **this repo** (`--attention-backend CUSTOM --kv-cache-dtype fp8`) |

The long-ctx win is driven primarily by avoided scheduler preemption once
the fp16 KV cache exceeds the memory budget: ours keeps 8 concurrent 32k
contexts on-GPU; FA preempts 2–3 of them and inflates median TTFT ~10×.

### Quality

| Metric | Result |
|--------|--------|
| WikiText-2 PPL | 14.91 → 14.91 (0.01% degradation) |
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
