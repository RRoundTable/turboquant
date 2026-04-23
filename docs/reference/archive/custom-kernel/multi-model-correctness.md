# Multi-Model Correctness: FP16 vs TQ 4-bit

4 prompts per model, greedy decoding, 20 tokens, A100.

## Results

| Model | Config | Exact Match | Factual | Status |
|-------|--------|------------|---------|--------|
| Qwen3-0.6B | 28L, 16QO/8KV, hd=64, GQA=2:1 | **4/4** | 4/4 = 4/4 | PASS |
| Qwen3-1.7B | 28L, 16QO/8KV, hd=128, GQA=2:1 | **4/4** | 4/4 = 4/4 | PASS |
| Qwen3-4B | 36L, 32QO/8KV, hd=80, GQA=4:1 | **4/4** | 4/4 = 4/4 | PASS |
| Qwen3-8B | 36L, 32QO/8KV, hd=128, GQA=4:1 | **4/4** | 4/4 = 4/4 | PASS |
| Mistral-7B | 32L, 32QO/8KV, hd=128, GQA=4:1 | **4/4** | 2/4 = 2/4 | PASS |

**5/5 models: 100% exact token match.** Factual accuracy identical between FP16 and TQ.

## Skipped / Failed

| Model | Reason |
|-------|--------|
| Llama-3.2-1B/3B | Gated repo (needs HF_TOKEN) |
| Phi-3-mini | DynamicCache `.seen_tokens` API incompatibility |
| Gemma-2-2B | Not reached (script crashed on Phi-3) |

## Tested architectures

| Feature | Values tested |
|---------|--------------|
| head_dim | 64, 80, 128 |
| GQA ratio | 2:1, 4:1 |
| Layers | 28, 32, 36 |
| Model size | 0.6B — 8B |
| KV heads | 8 |
