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
| 16384 ×  4 |  2.42 GB |    92.5 |    92.5 |  26.1 |          90.1 | 1.03x ≈ | 2.5 × 36.6 ms |
| 16384 ×  8 |  4.83 GB |   184.5 |   184.9 |  26.1 |         190.3 | 0.97x ✓ | 3.7 × 49.4 ms |
| 16384 × 32 | 19.33 GB |   741.0 |   742.9 |  26.0 |         350.0 | 2.12x ✗ | 5.6 × 133.1 ms |
| 32768 ×  1 |  1.21 GB |    46.2 |    46.4 |  26.0 |          40.6 | 1.14x ≈ | 1.7 × 26.6 ms |
| 32768 ×  4 |  4.83 GB |   184.6 |   185.0 |  26.1 |         140.6 | 1.32x ≈ | 3.8 × 49.0 ms |
| 32768 ×  8 |  9.66 GB |   369.6 |   370.6 |  26.1 |         259.2 | 1.43x ≈ | 4.9 × 75.7 ms |

## Read

- **PCIe Gen4 effective bandwidth ≈ 26 GB/s** (vs 32 GB/s peak — 81 %).
  Symmetric in both directions.

### Extended grid (HYP-047b: 16384×4, 32k×{1,4,8})

- **TQ prefill is fast at long-ctx** (32768×1 = 41 ms vs FA 68 ms; 32768×8 =
  259 ms vs FA 361 ms). Side-effect: synchronous restore loses to re-prefill
  at every 32k config (c2g/prefill = 1.13–1.43×). Async overlap still hides it
  (restore = 0.6–4.9 decode steps).
- **The trade flips at 32k**: at the original HYP-041 grid restore wins
  outright; at the extended grid TQ prefill is cheap enough that re-prefill
  is competitive with restore. Async overlap is the deciding factor.
- **One regime is clear**: TQ + offload only beats raw fp16 + offload, never
  itself + re-prefill at 32k+. Past 32k the value is *capacity* (fitting the
  context at all) rather than *prefill amortization*.
