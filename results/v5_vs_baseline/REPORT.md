# TurboQuant v5_paged vs baseline vLLM

Model: Qwen/Qwen3-8B (fp16) | output_len=128 | A100-40GB | enforce_eager=True

| seq  | batch | base tok/s | tq tok/s | tq/base | base lat (ms) | tq lat (ms) | base mem (GB) | tq mem (GB) |
|-----:|------:|-----------:|---------:|--------:|--------------:|------------:|--------------:|------------:|
| 1024 |     1 |       48.4 |     38.4 |   0.79× |          2643 |        3332 |         34.26 |       34.31 |
| 1024 |     8 |      376.8 |    291.6 |   0.77× |          2718 |        3512 |         34.26 |       34.62 |
| 1024 |    32 |     1415.6 |    961.2 |   0.68× |          2894 |        4261 |         34.26 |       37.88 |
| 4096 |     1 |       47.2 |     38.1 |   0.81× |          2710 |        3360 |         34.45 |       34.49 |
| 4096 |     8 |      373.9 |    291.3 |   0.78× |          2739 |        3515 |         34.45 |       37.00 |
| 4096 |    32 |     1268.9 |      OOM |       — |          3228 |         OOM |         34.45 |         OOM |
| 8192 |     1 |       47.6 |     37.3 |   0.78× |          2690 |        3434 |         34.83 |       34.92 |
| 8192 |     8 |      380.3 |    215.1 |   0.57× |          2693 |        4760 |         34.83 |       38.51 |
| 8192 |    32 |      922.0 |      OOM |       — |          4443 |         OOM |         34.83 |         OOM |
| 16384 |     1 |       48.2 |     38.2 |   0.79× |          2656 |        3352 |         34.83 |       34.94 |
| 16384 |     8 |      351.3 |      OOM |       — |          2915 |         OOM |         34.83 |         OOM |
| 16384 |    32 |      600.5 |      OOM |       — |          6821 |         OOM |         34.83 |         OOM |

Notes:
- enforce_eager=True for both (A100 SM80 cannot torch.compile fp8e4nv ops; 
  TQ requires kv_cache_dtype='fp8' which lowers to fp8e4nv otherwise).
- baseline = stock vLLM, kv_cache_dtype=auto, FLASHINFER backend.
- tq = vLLM + TurboQuant patches, kv_cache_dtype='fp8', CUSTOM backend.
- OOM = TurboQuant workspace allocation exceeds 40 GB at this batch×seq.
- per-trial median over 3 trials, 1 warmup, output_len=128.
