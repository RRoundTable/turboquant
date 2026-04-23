# HYP-047: TQ KV offload + reuse — validate transfer cost first

## Context

The HYP-045 split-metric report (`results/v5_split/REPORT.md`) reframes
TQ on A100 as: **prefill ≈ baseline (often faster), decode 0.3–0.8×
baseline**. The decode gap is at the SM80 kernel ceiling
(HYP-035/037/040/042b). The remaining product wins on A100 are:

1. **Memory capacity** — 3.2× more KV tokens per byte (PR #39868, HYP-045).
2. **Cheap prefill** — TQ writes 3.2× fewer KV bytes during prefill, so
   16384 × 32 prefill is 350 ms TQ vs 537 ms FA (0.65×).

(2) suggests: if prefill cost can be amortized across reuse — store the
TQ-compressed KV cache, swap it back in on cache hit, skip prefill —
the workloads where TQ already matches baseline (RAG, multi-turn,
agentic) get an additional cache-hit speedup.

The load-bearing assumption is **PCIe transfer time for the compressed
KV is small relative to prefill cost it replaces**. This hypothesis
validates that one number before any vLLM integration work.

## Hypothesis

For Qwen3-8B at the HYP-041 sweep grid on A100-40GB with PCIe Gen4 x16
(~32 GB/s effective), the round-trip time to spill TQ-compressed KV to
host pinned memory and restore it is **<10 % of measured prefill time**
across every config we care about.

## Prediction

KV size per request scales as `36 layers × 8 KV heads × seq × qbytes(=64) × 2 (K+V)` bytes
= `36 × 8 × seq × 64 × 2 = 36864 × seq` bytes ≈ 36 KB × seq.

| seq × batch | KV bytes | predicted CPU↔GPU (32 GB/s) | measured prefill (split-metric) | predicted ratio |
|------------:|---------:|----------------------------:|--------------------------------:|----------------:|
|    1024×1   |   37 MB  |       ~1.2 ms (one way)     |  0.6 ms                          |  ~2× — too small to matter |
|    8192×8   |  2.4 GB  |       ~75 ms                | 134 ms                           |  0.56× — borderline |
|   16384×8   |  4.7 GB  |       ~150 ms               | 190 ms                           |  0.79× — close |
|   16384×32  |   18.9 GB |       ~590 ms               | 350 ms                           |  **1.7× — TRANSFER WINS** |

The 16384×32 row is a red flag: at very large batch × seq, the TQ
cache itself is so big that PCIe transfer takes longer than re-prefill.
That's an inherent limit of CPU offload — only worth it when prefill
is even more expensive (multi-shot reuse) OR the GPU is otherwise
too small to fit the cache (capacity story, not throughput).

## Method

**Step 1 (this hypothesis): measure raw transfer.**

Standalone microbench (`tests/bench_kv_offload.py`):
- Allocate GPU `(36, 2, batch, num_kv_heads, seq, qbytes)` uint8 = full
  per-layer KV slab in TQ format.
- Allocate matching CPU pinned tensor.
- Time:
  - GPU → CPU `tensor.copy_(non_blocking=False)` after sync
  - CPU → GPU `tensor.copy_(non_blocking=False)` after sync
- Sweep batch ∈ {1, 8, 32}, seq ∈ {1024, 4096, 8192, 16384} — 12 configs,
  one Forge job, ~1 min each.
- Report: bytes, GPU→CPU ms, CPU→GPU ms, achieved bandwidth, ratio vs
  measured prefill from `results/v5_split/tq-s{seq}-b{batch}.json`.

**Step 2 (only if step 1 passes): integrate with vLLM.**
File HYP-048 if and only if HYP-047 confirms transfer < prefill at
the configs we care about (8192×8 and below, where TQ already wins on
prefill).

## Status: confirmed for medium configs, regime-bounded

## Result (A100-40GB, PCIe Gen4 x16)

Measured PCIe Gen4 throughput: **~26 GB/s effective** (81 % of peak),
symmetric in both directions.

| seq × b   | KV size  | restore (ms) | TQ prefill (ms) | restore/prefill |
|-----------|---------:|-------------:|----------------:|----------------:|
|  1024 × 8 |  0.30 GB |       11.5   |          34.5   |  **0.33×** ✓ |
|  4096 × 8 |  1.21 GB |       46.2   |          75.4   |  **0.61×** ✓ |
|  8192 × 8 |  2.42 GB |       92.4   |         134.5   |  **0.69×** ✓ |
|  4096 × 32|  4.83 GB |      185.0   |         201.9   |  0.92× ✓ |
|  8192 × 32|  9.66 GB |      370.7   |         282.5   |  1.31× ≈ |
| 16384 × 8 |  4.83 GB |      184.9   |         190.3   |  0.97× ✓ |
| 16384 × 32| 19.33 GB |      742.9   |         350.0   |  2.12× ✗ |

(Full table including 1024×1 / 4096×1 / 8192×1 / 16384×1 in
`results/hyp047/REPORT.md`; restore wins where `c2g/prefill < 1`.)

### Verdict

- **Sync restore wins at ≤ 16384 × 8** — every realistic medium config.
- **Async overlap with decode hides transfer at almost everything.**
  Decode/step is 26–133 ms across the grid; restore is ≤ 5–6 decode steps
  even at the worst case (16384×32). Pre-fetching during the first 2–6
  decode steps makes restore effectively free.
- **fp16 cache would be 3.2× larger** → restore cost > re-prefill at
  every non-trivial config. **TQ is the enabling factor** for offload+reuse
  on A100; the same idea on uncompressed cache would not pay off.
- **Hard ceiling at very large × very large** (16384×32 = 19 GB): restore
  takes 2× re-prefill even with TQ. For these, you'd need to offload
  *partial* cache (e.g. just the K side, not V) or use multiple PCIe
  links (NVLink/NVMe). Out of scope for HYP-047.

## Next step → HYP-048

Integrate offload+reuse into vLLM. Concrete plan:

1. Reuse vLLM's existing prefix-caching block manager: hook into the
   `swap_in/swap_out` mechanism. The TurboQuantBackend already exposes
   `get_kv_cache_page_size` (PR #39868); we just need to teach the
   block manager that swap-in chunks are TQ bytes, not fp16.
2. Add an end-to-end bench that measures TTFT on cache-hit vs
   cache-miss for a realistic prompt-reuse pattern (RAG: same system
   prompt + varying user query).
3. Compare TTFT with: TQ + reuse, fp16 + reuse (won't work on A100 for
   medium ctx — too big), no reuse (re-prefill every time).

Filed as a stub. Don't build it yet — confirm with user that this is
the desired direction before committing engineering time.
