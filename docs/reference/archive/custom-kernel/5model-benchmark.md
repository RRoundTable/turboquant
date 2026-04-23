# 5-Model Full Benchmark: FP16 vs TQ 4-bit

Hardware: A100-SXM4-40GB, eager mode, batch=1, 4 prompts × 30 tokens.

## TTFT and TPOT

| Model | Config | FP16 TTFT | TQ TTFT | FP16 TPOT | TQ TPOT | TPOT ratio | KV compress |
|-------|--------|-----------|---------|-----------|---------|-----------|------------|
| Qwen3-0.6B | 28L, GQA=2:1, hd=64 | 32.6ms | 32.1ms | 30.3ms | 29.7ms | **0.98×** | 3.8× |
| Qwen3-1.7B | 28L, GQA=2:1, hd=128 | 33.7ms | 33.5ms | 30.2ms | 30.1ms | **1.00×** | 3.8× |
| Qwen3-4B | 36L, GQA=4:1, hd=80 | 42.2ms | 42.0ms | 39.3ms | 39.3ms | **1.00×** | 3.8× |
| Qwen3-8B | 36L, GQA=4:1, hd=128 | 42.0ms | 42.1ms | 39.7ms | 39.8ms | **1.00×** | 3.8× |
| Mistral-7B | 32L, GQA=4:1, hd=128 | 30.8ms | 30.2ms | 27.6ms | 27.3ms | **0.99×** | 3.8× |

## Throughput

| Model | FP16 tok/s | TQ tok/s | Ratio |
|-------|-----------|---------|-------|
| Qwen3-0.6B | 31.7 | 32.4 | 1.02× |
| Qwen3-1.7B | 31.8 | 31.9 | 1.00× |
| Qwen3-4B | 24.5 | 24.5 | 1.00× |
| Qwen3-8B | 24.2 | 24.2 | 1.00× |
| Mistral-7B | 34.8 | 35.2 | 1.01× |

## Key findings

1. **TPOT overhead: 0-2% across all models** — TQ is essentially free in eager mode
2. **TTFT overhead: 0-2%** — the Python quantize-dequant hook adds negligible cost
   (would be even less with the CUDA write kernel)
3. **Some models are faster with TQ** (Qwen3-0.6B, Mistral-7B) — within noise
4. **3.8× KV compression** is constant across all architectures
5. **Tested: hd={64,80,128}, GQA={2:1,4:1}, 28-36 layers, 0.6B-8B params**

## Projected throughput at max batch

| Model | FP16 max batch (seq=2K) | TQ max batch | Throughput gain |
|-------|------------------------|-------------|----------------|
| Qwen3-0.6B | ~300 | ~1100 | ~3.7× |
| Qwen3-1.7B | ~72 | ~270 | ~3.6× |
| Qwen3-4B | ~30 | ~110 | ~3.6× |
| Qwen3-8B | ~15 | ~55 | ~3.6× |
| Mistral-7B | ~15 | ~55 | ~3.6× |
