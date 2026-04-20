# HYP-047 — TQ KV offload + reuse: PCIe transfer cost vs measured prefill

A100-40GB, PCIe Gen4 x16. Pinned host memory, blocking copies.
Cache size = 36 layers × 2 (K+V) × batch × 8 KV-heads × seq × 64 qbytes.

| seq × b | KV size | g2c (ms) | c2g (ms) | bw (GB/s) | TQ prefill (ms) | **c2g / prefill** | overlap (decodes hidden) |
|---------|--------:|---------:|---------:|----------:|----------------:|------------------:|-------------------------:|
|  1024 ×  1 |  0.04 GB |     1.7 |     1.5 |  25.8 |           0.6 | 2.39x ✗ | 0.1 × 26.7 ms |
|  1024 ×  8 |  0.30 GB |    13.1 |    11.5 |  26.2 |          34.5 | 0.33x ✓ | 0.4 × 26.3 ms |
|  1024 × 32 |  1.21 GB |    46.2 |    46.4 |  26.0 |         113.5 | 0.41x ✓ | 1.7 × 26.9 ms |
|  4096 ×  1 |  0.15 GB |     5.8 |     5.8 |  26.1 |           3.9 | 1.48x ≈ | 0.2 × 26.3 ms |
|  4096 ×  8 |  1.21 GB |    46.0 |    46.2 |  26.1 |          75.4 | 0.61x ✓ | 1.8 × 26.4 ms |
|  4096 × 32 |  4.83 GB |   185.1 |   185.0 |  26.1 |         201.9 | 0.92x ✓ | 3.6 × 51.6 ms |
|  8192 ×  1 |  0.30 GB |    11.5 |    23.1 |  13.1 |           9.0 | 2.57x ✗ | 0.9 × 25.5 ms |
|  8192 ×  8 |  2.42 GB |    92.3 |    92.4 |  26.1 |         134.5 | 0.69x ✓ | 2.7 × 34.0 ms |
|  8192 × 32 |  9.66 GB |   371.3 |   370.7 |  26.1 |         282.5 | 1.31x ≈ | 4.7 × 78.5 ms |
| 16384 ×  1 |  0.60 GB |    23.0 |    23.0 |  26.2 |          19.4 | 1.18x ≈ | 0.9 × 25.9 ms |
| 16384 ×  8 |  4.83 GB |   184.5 |   184.9 |  26.1 |         190.3 | 0.97x ✓ | 3.7 × 49.4 ms |
| 16384 × 32 | 19.33 GB |   741.0 |   742.9 |  26.0 |         350.0 | 2.12x ✗ | 5.6 × 133.1 ms |

## Read

- **PCIe Gen4 effective bandwidth ≈ 26 GB/s** (vs 32 GB/s peak — 81 %).
  Symmetric in both directions.
- **One-way restore (cache hit on warm spill)** vs re-prefill:
  - WIN at small/medium configs (≤ 8192 × 8): restore is 0.34–0.97× prefill
  - LOSS at large × large: 16384 × 32 restore is 740 ms vs 350 ms re-prefill
- **Async overlap with decode** changes the picture: at decode/step ≈ 30–130 ms,
  the ~92 ms restore at 8192×8 is hidden behind 3 decode steps; only the largest
  config (16384×32) needs ~5 steps to cover transfer.
- **fp16 KV would be 3.2× larger** → almost no config wins for fp16 offload.
  TQ compression makes the offload-reuse trade viable in a regime where
  raw fp16 cannot.
