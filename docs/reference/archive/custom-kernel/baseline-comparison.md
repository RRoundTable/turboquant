# TurboQuant v4 vs FP16 SDPA Baseline — A100

All measurements: batch=1, page_size=16, A100-SXM4-40GB.
SDPA = PyTorch `scaled_dot_product_attention` with contiguous fp16 KV.

## Per-Model Results

### Qwen3-0.6B (16 QO, 8 KV, head_dim=64, GQA=2:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 22 μs | 36 μs | 1.6× | 3.8× less |
| 512 | 32 μs | 43 μs | 1.4× | 3.8× less |
| 1024 | 30 μs | 68 μs | 2.3× | 3.8× less |
| 2048 | 31 μs | 118 μs | 3.8× | 3.8× less |

### Qwen3-1.7B (16 QO, 8 KV, head_dim=128, GQA=2:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 22 μs | 37 μs | 1.6× | 3.8× less |
| 512 | 31 μs | 60 μs | 2.0× | 3.8× less |
| 1024 | 33 μs | 103 μs | 3.1× | 3.8× less |
| 2048 | 31 μs | 188 μs | 6.1× | 3.8× less |

### Llama-2-7B (32 QO, 32 KV, head_dim=128, MHA=1:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 23 μs | 38 μs | 1.7× | 3.8× less |
| 512 | 30 μs | 41 μs | 1.4× | 3.8× less |
| 1024 | 31 μs | 71 μs | 2.3× | 3.8× less |
| 2048 | 36 μs | 125 μs | 3.5× | 3.8× less |

### Llama-3-8B (32 QO, 8 KV, head_dim=128, GQA=4:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 22 μs | 42 μs | 1.9× | 3.8× less |
| 512 | 31 μs | 93 μs | 3.0× | 3.8× less |
| 1024 | 30 μs | 168 μs | 5.6× | 3.8× less |
| 2048 | 33 μs | 297 μs | 8.9× | 3.8× less |

### Mistral-7B (32 QO, 8 KV, head_dim=128, GQA=4:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 22 μs | 39 μs | 1.8× | 3.8× less |
| 512 | 30 μs | 76 μs | 2.5× | 3.8× less |
| 1024 | 30 μs | 138 μs | 4.6× | 3.8× less |
| 2048 | 30 μs | 250 μs | 8.2× | 3.8× less |

### Llama-3-70B (64 QO, 8 KV, head_dim=128, GQA=8:1)

| seq | SDPA | TQ v4 | Ratio | Memory |
|-----|------|-------|-------|--------|
| 128 | 23 μs | 41 μs | 1.8× | 3.8× less |
| 512 | 30 μs | 124 μs | 4.1× | 3.8× less |
| 1024 | 31 μs | 235 μs | 7.7× | 3.8× less |
| 2048 | 63 μs | 457 μs | 7.3× | 3.8× less |

## Summary

| Model | Best ratio (short seq) | Worst ratio (long seq) | Memory saving |
|-------|----------------------|----------------------|---------------|
| Qwen3-0.6B | 1.4× slower | 3.8× slower | 3.8× |
| Qwen3-1.7B | 1.6× slower | 6.1× slower | 3.8× |
| Llama-2-7B | 1.4× slower | 3.5× slower | 3.8× |
| Llama-3-8B | 1.9× slower | 8.9× slower | 3.8× |
| Mistral-7B | 1.8× slower | 8.2× slower | 3.8× |
| Llama-3-70B | 1.8× slower | 7.3× slower | 3.8× |

## Analysis

1. **MHA models (Llama-2-7B) have the best latency ratio** — 32 KV heads means
   more blocks, better GPU utilization, and bdz can be high.

2. **High-GQA models (Llama-3-8B/70B) have the worst ratio** — fewer KV heads
   means fewer blocks, and each block does more work (bdy QO heads per KV head).

3. **SDPA is nearly constant at 22-33 μs** for seq 128-2048 — it uses FlashAttention
   which is compute-bound and well-pipelined. TQ v4 scales linearly with seq_len.

4. **The trade-off is clear: 1.4-9× slower kernel for 3.8× memory savings.**
   At serving time, the memory savings allow 3.8× more concurrent requests,
   which more than compensates for the per-request latency increase in
   throughput-bound scenarios.

5. **At short sequences (128-512), the overhead is modest (1.4-3×).**
   For chatbot workloads with many short requests, TQ is practical.
