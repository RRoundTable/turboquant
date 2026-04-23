# TQ v4 + Split-KV vs FP16 SDPA Baseline — A100

Adaptive: use non-split at short seq, split-8 at long seq. Best of both shown.

## Qwen3-1.7B (16 QO, 8 KV, GQA=2:1)

| seq | SDPA | TQ best | vs SDPA | Memory |
|-----|------|---------|---------|--------|
| 256 | 31 μs | 43 μs (nosplit) | 1.4× | 3.8× less |
| 512 | 31 μs | 64 μs (nosplit) | 2.1× | 3.8× less |
| 1024 | 30 μs | 106 μs (nosplit) | 3.6× | 3.8× less |
| 2048 | 31 μs | **121 μs (split-8)** | 3.9× | 3.8× less |
| 4096 | 41 μs | **144 μs (split-8)** | 3.5× | 3.8× less |

Split-KV reduces 4096 from 364μs → 144μs (**2.5× speedup**).

## Llama-3-8B (32 QO, 8 KV, GQA=4:1)

| seq | SDPA | TQ best | vs SDPA | Memory |
|-----|------|---------|---------|--------|
| 256 | 31 μs | 60 μs (nosplit) | 2.0× | 3.8× less |
| 512 | 31 μs | 88 μs (nosplit) | 2.8× | 3.8× less |
| 1024 | 31 μs | **117 μs (split-8)** | 3.7× | 3.8× less |
| 2048 | 35 μs | **134 μs (split-8)** | 3.9× | 3.8× less |
| 4096 | 68 μs | **177 μs (split-8)** | 2.6× | 3.8× less |

Split-KV reduces 4096 from 631μs → 177μs (**3.6× speedup**).

## Llama-2-7B (32 QO, 32 KV, MHA=1:1)

| seq | SDPA | TQ best | vs SDPA | Memory |
|-----|------|---------|---------|--------|
| 256 | 31 μs | 37 μs (nosplit) | **1.2×** | 3.8× less |
| 512 | 31 μs | 41 μs (nosplit) | **1.3×** | 3.8× less |
| 1024 | 33 μs | 79 μs (nosplit) | 2.4× | 3.8× less |
| 2048 | 36 μs | 132 μs (nosplit) | 3.7× | 3.8× less |
| 4096 | 68 μs | **169 μs (split-8)** | 2.5× | 3.8× less |

MHA models (32 KV heads = 32 blocks already) benefit least from split-KV.
At short seq: only **1.2-1.3× overhead** with 3.8× memory savings.

## Summary: best TQ vs SDPA ratio (adaptive split)

| Model | seq=256 | seq=512 | seq=1024 | seq=2048 | seq=4096 |
|-------|---------|---------|----------|----------|----------|
| Qwen3-1.7B | 1.4× | 2.1× | 3.6× | 3.9× | **3.5×** |
| Llama-3-8B | 2.0× | 2.8× | 3.7× | 3.9× | **2.6×** |
| Llama-2-7B | **1.2×** | **1.3×** | 2.4× | 3.7× | **2.5×** |

**Key insight:** Split-KV flattens the latency curve at long sequences.
Without split-KV, TQ latency grows linearly. With split-KV, it caps at
~120-180μs regardless of seq length (work distributed across SMs).

At seq=4096, the TQ/SDPA ratio **improves** compared to seq=2048 because
SDPA also starts scaling while TQ's split-KV absorbs the extra work.
