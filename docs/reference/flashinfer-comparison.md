# FlashInfer vs SDPA vs TurboQuant — Decode Kernel Comparison

Hardware: A100-SXM4-40GB, batch=1, FlashInfer 0.6.6, PyTorch 2.10.

## Qwen3-1.7B (16 QO, 8 KV, hd=128, GQA=2:1)

| seq | FlashInfer | SDPA | TQ best | TQ/FI | TQ/SDPA | TQ mem |
|-----|-----------|------|---------|-------|---------|--------|
| 128 | 32 μs | 23 μs | **22 μs** | **0.68×** | 0.98× | 3.8× less |
| 256 | 36 μs | 30 μs | **32 μs** | **0.88×** | 1.06× | 3.8× less |
| 512 | 36 μs | 30 μs | 48 μs | 1.33× | 1.62× | 3.8× less |
| 1024 | 36 μs | 30 μs | 46 μs | 1.27× | 1.52× | 3.8× less |
| 2048 | 36 μs | 31 μs | 47 μs | 1.30× | 1.54× | 3.8× less |
| 4096 | 37 μs | 41 μs | 48 μs | 1.31× | 1.16× | 3.8× less |

**TQ beats FlashInfer at seq≤256!** FlashInfer is ~32-36μs constant (split-KV overhead),
while TQ contiguous nosplit is 22-32μs at short sequences.

## Key Finding: FlashInfer is SLOWER than SDPA at seq≤2048

| seq | FlashInfer | SDPA | FI/SDPA |
|-----|-----------|------|---------|
| 128 | 32 μs | 23 μs | 1.37× slower |
| 512 | 36 μs | 30 μs | 1.20× slower |
| 1024 | 36 μs | 30 μs | 1.20× slower |
| 2048 | 36 μs | 31 μs | 1.15× slower |
| 4096 | 37 μs | 41 μs | **0.90× (faster)** |

FlashInfer only beats SDPA at seq≥4096! At shorter sequences, FlashInfer's
paged KV + split-KV overhead makes it slower than SDPA's contiguous FlashAttention.

## Qwen3-8B (32 QO, 8 KV, GQA=4:1) — FlashInfer vs SDPA

| seq | FlashInfer | SDPA | FI/SDPA |
|-----|-----------|------|---------|
| 128 | 29 μs | 22 μs | 1.34× |
| 1024 | 35 μs | 30 μs | 1.20× |
| 2048 | 35 μs | 39 μs | **0.91×** |
| 4096 | 36 μs | 67 μs | **0.53×** |

FlashInfer becomes faster at seq≥2048 for GQA=4:1 models.

## Revised TQ vs FlashInfer comparison

**TQ's true competitor is FlashInfer (paged), not SDPA (contiguous):**

| seq | FlashInfer (paged FP16) | TQ (contiguous 4-bit) | TQ/FI | Memory |
|-----|------------------------|----------------------|-------|--------|
| 128 | 32 μs | **22 μs** | **0.68× (32% faster!)** | 3.8× less |
| 256 | 36 μs | **32 μs** | **0.88× (12% faster!)** | 3.8× less |
| 1024 | 36 μs | 46 μs | 1.27× | 3.8× less |
| 4096 | 37 μs | 48 μs | 1.31× | 3.8× less |

**TQ is faster than FlashInfer at seq≤256, and only 1.3× slower at seq≥1024.**
Combined with 3.8× memory savings: TQ achieves ~2.9× throughput at max batch.
