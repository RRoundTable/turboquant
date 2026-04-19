# HYP-045 end-to-end: skip dead workspace allocation at num_splits>1

Qwen/Qwen3-8B fp16 | output_len=128 | A100-40GB | enforce_eager=True | tok/s

TQ progression: v0 (HYP-041 original) → v1 (HYP-044 chunk-cap) → **v2 (HYP-045 dead-alloc fix)**.

| seq × b | TQ v0 | TQ v1 | **TQ v2** | FA | FI | TQ mem v0→v2 | tq/FA v2 | tq/FI v2 |
|---------|------:|------:|----------:|---:|---:|:-------------|---------:|---------:|
|  1024 ×  1 |  38.4 |  38.8 | ** 37.4** | 49.1 | 50.7 | 34.3 → 34.3 |  0.76x |  0.74x |
|  1024 ×  8 | 291.6 | 300.4 | **297.0** | 374.2 | 392.9 | 34.6 → 34.3 |  0.79x |  0.76x |
|  1024 × 32 | 961.2 | 1138.3 | **1154.4** | 1446.8 | 873.4 | 37.9 → 34.3 |  0.80x |  1.32x |
|  4096 ×  1 |  38.1 |  38.6 | ** 37.5** | 48.5 | 51.8 | 34.5 → 34.5 |  0.77x |  0.72x |
|  4096 ×  8 | 291.3 | 296.3 | **296.4** | 381.2 | 223.3 | 37.0 → 34.5 |  0.78x |  1.33x |
|  4096 × 32 |     — |     — | **605.1** | 1266.6 | 1438.9 | — → 34.7 |  0.48x |  0.42x |
|  8192 ×  1 |  37.3 |  37.9 | ** 38.0** | 47.9 | 49.8 | 34.9 → 34.9 |  0.79x |  0.76x |
|  8192 ×  8 | 215.1 | 227.7 | **228.2** | 370.9 | 406.6 | 38.5 → 34.9 |  0.62x |  0.56x |
|  8192 × 32 |     — |     — | **394.5** | 918.6 | 1101.5 | — → 34.9 |  0.43x |  0.36x |
| 16384 ×  1 |  38.2 |  37.9 | ** 37.5** | 48.3 | 50.3 | 34.9 → 34.9 |  0.78x |  0.75x |
| 16384 ×  8 |     — |     — | **156.1** | 352.1 | 386.5 | — → 34.9 |  0.44x |  0.40x |
| 16384 × 32 |     — |     — | **236.2** | 598.1 |    — | — → 34.9 |  0.39x |      — |

Notes:
- v0 = commit 5db1d80 (HYP-041). v1 = HYP-044 chunk-cap. **v2 = HYP-045: skip k_quant/v_quant/k_norms/v_norms allocation when num_splits>1 (dead in paged-split path).**
- HYP-045 removes ~9.6 GB of dead allocation at seq=8192 × batch=32 → unblocks all 4 HYP-041 OOM configs.
- Decode tok/s at non-OOM configs unchanged from v1 (same kernel work).
- FA = vLLM auto-pick (FLASH_ATTN). FI = explicit attention_backend='FLASHINFER'.
