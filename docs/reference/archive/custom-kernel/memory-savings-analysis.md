# Phase 8b+8e: Memory Savings and Max Batch Size Analysis

## KV Cache Memory per Token per Layer

| Format | Bytes per token per head | Formula |
|--------|------------------------|---------|
| FP16 | 2 × head_dim × 2 (K+V) = 512 B | hd=128 |
| TQ 4-bit | (32 + 2) × 2 chunks × 2 (K+V) = 136 B | 4-bit packed + fp16 norm |
| **Compression ratio** | **3.76×** | 512 / 136 |

## Total KV Cache Memory (Qwen3-1.7B: 28 layers, 8 KV heads, hd=128)

| Seq len | FP16 KV cache | TQ 4-bit KV cache | Savings |
|---------|--------------|-------------------|---------|
| 1K | 28 × 8 × 1024 × 512 B = **112 MB** | 28 × 8 × 1024 × 136 B = **30 MB** | 82 MB |
| 4K | **448 MB** | **119 MB** | 329 MB |
| 8K | **896 MB** | **238 MB** | 658 MB |
| 32K | **3.5 GB** | **952 MB** | 2.6 GB |
| 128K | **14.0 GB** | **3.7 GB** | 10.3 GB |

## Max Batch Size (Phase 8e)

Assuming A100-40GB with 32GB available for KV cache (rest for model weights + activations):

### Qwen3-1.7B (28L, 8KV, hd=128)

| Seq len | FP16 max batch | TQ 4-bit max batch | Improvement |
|---------|---------------|-------------------|-------------|
| 1K | 32000/112 = **285** | 32000/30 = **1066** | **3.7×** |
| 4K | 32000/448 = **71** | 32000/119 = **268** | **3.8×** |
| 8K | 32000/896 = **35** | 32000/238 = **134** | **3.8×** |
| 32K | 32000/3584 = **8** | 32000/952 = **33** | **4.1×** |

### Llama-3-8B (32L, 8KV, hd=128)

KV cache per token: FP16 = 32 × 8 × 512 = 128 KB, TQ = 32 × 8 × 136 = 34 KB

| Seq len | FP16 max batch | TQ 4-bit max batch | Improvement |
|---------|---------------|-------------------|-------------|
| 4K | 32000/512 = **62** | 32000/136 = **235** | **3.8×** |
| 8K | 32000/1024 = **31** | 32000/272 = **117** | **3.8×** |
| 32K | 32000/4096 = **7** | 32000/1088 = **29** | **4.1×** |

### Llama-3-70B (80L, 8KV, hd=128) — per GPU with TP=8

KV per token per GPU: FP16 = 80/8 × 8 × 512 = 40 KB, TQ = 80/8 × 8 × 136 = 10.6 KB

| Seq len | FP16 max batch | TQ 4-bit max batch | Improvement |
|---------|---------------|-------------------|-------------|
| 4K | 32000/160 = **200** | 32000/42.4 = **754** | **3.8×** |
| 32K | 32000/1280 = **25** | 32000/340 = **94** | **3.8×** |

## Key Takeaway

**3.76× more concurrent requests** across all model sizes and sequence lengths.
At long contexts (32K+), this is the difference between serving 8 vs 33 concurrent
users on a single GPU — a direct throughput multiplier.

The savings scale linearly: longer sequences and larger models benefit more in
absolute terms, but the ratio stays constant at 3.76×.
