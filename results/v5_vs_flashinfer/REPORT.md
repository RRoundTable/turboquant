# TurboQuant (HYP-044) vs FlashInfer vs FlashAttention (vLLM auto)

Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True | tok/s

| seq  | b  | FlashInfer | FlashAttn | TurboQuant | tq/FI | tq/FA | FI/FA |
|-----:|---:|-----------:|----------:|-----------:|------:|------:|------:|
| 1024 |  1 |       50.7 |      48.0 |       38.8 | 0.77x | 0.81x | 1.06x |
| 1024 |  8 |      392.9 |     389.1 |      300.4 | 0.76x | 0.77x | 1.01x |
| 1024 | 32 |      873.4 |    1471.2 |     1138.3 | 1.30x | 0.77x | 0.59x |
| 4096 |  1 |       51.8 |      47.7 |       38.6 | 0.75x | 0.81x | 1.09x |
| 4096 |  8 |      223.3 |     383.4 |      296.3 | 1.33x | 0.77x | 0.58x |
| 4096 | 32 |     1438.9 |    1271.3 |          — |     — |     — | 1.13x |
| 8192 |  1 |       49.8 |      48.3 |       37.9 | 0.76x | 0.79x | 1.03x |
| 8192 |  8 |      406.6 |     376.2 |      227.7 | 0.56x | 0.61x | 1.08x |
| 8192 | 32 |     1101.5 |     921.5 |          — |     — |     — | 1.20x |
| 16384 |  1 |       50.3 |      47.2 |       37.9 | 0.75x | 0.80x | 1.07x |
| 16384 |  8 |      386.5 |     351.6 |          — |     — |     — | 1.10x |

Notes:
- FlashInfer = explicit attention_backend='FLASHINFER'.
- FlashAttn = vLLM v0.19 auto-selects FLASH_ATTN at this seq/dtype on A100.
- TurboQuant = HYP-044 patched (chunk_size cap = 256).
- 'tq/FI' and 'tq/FA' are TurboQuant÷baseline tok/s ratios (>1 = TQ faster).
- All eager — A100 SM80 cannot torch.compile fp8e4nv (HYP-041 constraint).
