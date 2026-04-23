# Correctness Test: FP16 vs TurboQuant 4-bit

12 prompts, greedy decoding, 30 tokens each, A100.

## Results

### Qwen3-1.7B (28L, 16QO/8KV, GQA=2:1)

**Exact token match: 12/12 (100%)**

| # | Prompt | Match | Factual |
|---|--------|-------|---------|
| 0 | The capital of France is | EXACT | OK (Paris) |
| 1 | The capital of Japan is | EXACT | OK (Tokyo) |
| 2 | Water boils at | EXACT | OK (100°C) |
| 3 | The speed of light is approximately | EXACT | OK (3×10⁸) |
| 4 | Einstein is famous for his theory of | EXACT | OK (relativity) |
| 5 | Machine learning is a subset of | EXACT | OK (AI) |
| 6 | The largest planet in our solar system is | EXACT | - |
| 7 | Python is a | EXACT | OK (programming) |
| 8 | DNA stands for | EXACT | OK (deoxyribonucleic) |
| 9 | The chemical formula for water is | EXACT | OK (H2O) |
| 10 | 1 + 1 = | EXACT | OK (2) |
| 11 | The first president of the United States was | EXACT | OK (Washington) |

### Qwen3-8B (36L, 32QO/8KV, GQA=4:1)

**Exact token match: 12/12 (100%)**

All 12 prompts produce identical output tokens between FP16 and TQ 4-bit.
Factual accuracy: 12/12 for both FP16 and TQ.

## Method

Quantize-dequant hook on attention forward: after each layer's attention,
the KV cache entries are quantized to 4-bit Lloyd-Max codebook indices
and immediately dequantized back to fp16. This simulates the TurboQuant
pipeline where KV is stored compressed and restored on read.

The 100% exact match means the 4-bit quantization error is below the
fp16 rounding threshold for these prompts — the model produces the same
greedy-decoded tokens regardless of quantization.
