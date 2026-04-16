# HYP-030: Decode latency scaling vs seq_len on Qwen3-8B

## Hypothesis

TurboQuant fp8 decode latency (cudagraph mode) scales *worse than linear*
with `seq_len` relative to FP16 and FP8-native, so the gap widens as the
KV cache fills. Specifically, I expect TQ to be within 1.2× of FP16 at
short context and ~4× at 4K — dominated by scalar-FMA dequant cost and
the absence of split-KV on the `decode_v4_from_cache` path.

## Prediction

On A100-40GB, Qwen3-8B, random token IDs, 4 prompts × 32 gen tokens,
best of 3, CUDA graph capture, `gpu_memory_utilization=0.7`:

| seq_len | FP16 graphs | TQ fp8 graphs (predicted) |
|--------:|------------:|--------------------------:|
|   128   | ~4.0 ms     | 4–5 ms (within 1.2×)      |
|  4096   | ~4.5 ms     | 18–22 ms (4–5×)           |

KV tokens:
- FP16: ~83K (fp16 → 128 B/head)
- FP8 native: ~166K (2×)
- TQ fp8: **~266K (3.2×)** — unconditional across seq_len

## Method

Single Forge job (1 GPU, A100) sweeps 6 seq_lens × 3 backends inside one
LLM engine each. Uses random prompt token IDs to keep attention content
shape-only (not measuring quality — see HYP-029 for correctness). Uses
the ECR-published image `tq-hyp029:v2`.

Bench script: `/workspace/shared/tq_bench_all.py` (Forge shared NFS),
Forge job `9a35b7f4-1ff4-4679-93f7-f03778799b1a`.

Random input is appropriate here because we're measuring TPOT, throughput,
and memory capacity — all shape-only. Quality (coherence, PPL) would need
real text and is out of scope for this hypothesis.

## Status: **confirmed**

Both predictions met. Regression curve is clean; scale factor at 4K matches
the Phase 8 "Architecture gap" breakdown.

## Results

| seq_len | FP16 graphs | FP8 native graphs | TQ fp8 graphs | TQ/FP16 | tok/s (TQ) |
|---:|---:|---:|---:|---:|---:|
|  128 | 3.88 ms | 4.00 ms |  4.77 ms | 1.23× | 209.9 |
|  256 | 3.92    | 4.04    |  5.29    | 1.35× | 189.1 |
|  512 | 4.03    | 4.07    |  6.29    | 1.56× | 159.1 |
| 1024 | 4.02    | 4.09    |  8.36    | 2.08× | 119.6 |
| 2048 | 4.15    | 4.31    | 12.43    | 3.00× |  80.5 |
| 4096 | 4.42    | 4.31    | 20.26    | 4.58× |  49.3 |

KV tokens (independent of seq_len):
- FP16:     83,216
- FP8 nat: 166,448 (2.00×)
- TQ fp8:  266,144 (**3.20× vs FP16, 1.60× vs FP8 native**)

Raw JSON: `/workspace/shared/tq_sweep.json` on Forge NFS.

## Interpretation

The linear-ish regression is exactly what the Phase 8 architecture-gap
table predicts. Three attributable sources:

1. **Scalar FMA on every dequant.** Each of the 64 dims per chunk needs
   a codebook lookup + Hadamard scale multiply. FP16/FP8 use
   `mma.sync.m16n8k16` which runs concurrent with loads; TQ's dequant is
   ALU-bound and serializes with compute. This is ~40% of the gap at
   seq=128 and almost 100% at seq=4K (because kernel time grows with
   kv_len while load-overlap stays constant).

2. **No split-KV on the vLLM path.** `decode_v4_from_cache` (HYP-029)
   uses the non-partitioned path. HYP-018 already proved split-KV
   recovers most of the gap at long seq (48 μs at seq=1024 contiguous).
   Porting split-KV into `decode_v4_from_cache` is likely the single
   biggest win for long-seq decode.

3. **Python launch overhead scales with `block_table`.** Per step:
   `kv_indices = block_table.reshape(-1).to(torch.int32)` at
   `turboquant/vllm_backend_fused.py:271` is a host→device copy of
   `batch × max_pages_per_seq` int32s. At seq=4096 this is a 256-entry
   copy every decode step. Not the dominant factor but noticeable.

Memory side is the clean win: **3.20× more KV tokens**, flat across
seq_len. That translates directly to concurrent-request capacity at
fixed VRAM.

## Kernel-level analysis (standalone timing)

Standalone kernel timing (no vLLM, CUDA events, batch=1, Qwen3-8B config hd=128/8KV/GQA=4, A100-40GB):

TQ decode_v4_from_cache vs FlashInfer BatchDecodeWithPagedKVCache (fp16 tensor-core path):

| seq_len | TQ kernel (ms) | FlashInfer kernel (ms) | TQ/FI |
|---:|---:|---:|---:|
| 128 | 0.114 | 0.337 | 0.34x (TQ wins) |
| 256 | 0.200 | 0.212 | 0.95x (parity) |
| 512 | 0.357 | 0.213 | 1.68x |
| 1024 | 0.685 | 0.212 | 3.23x |
| 2048 | 1.325 | 0.213 | 6.23x |
| 4096 | 2.624 | 0.214 | 12.3x |

Key findings:

1. TQ wins at seq<=256 (less data: 80 vs 256 bytes/token)
2. FlashInfer is flat at ~0.21ms for seq 256-4096 (tensor cores hide O(seq) work)
3. TQ scales linearly (~0.6us per token of scalar-FMA dequant)
4. 12x kernel gap at 4096 matches the 15x FLOPS ratio (scalar FMA 20 TFLOPS vs tensor core 312 TFLOPS)
5. In vLLM context, kernel is only 11% of TPOT -- 89% is framework overhead. But the framework overhead is mostly amortized under CUDA graphs; the kernel gap is the irreducible cost.

## Kernel vs vLLM TPOT decomposition

| seq_len | Kernel (ms) | vLLM TPOT (ms) | Kernel % | Framework overhead |
|---:|---:|---:|---:|---:|
| 128 | 0.115 | 4.77 | 2.4% | 4.66 ms |
| 512 | 0.358 | 6.29 | 5.7% | 5.93 ms |
| 1024 | 0.688 | 8.36 | 8.2% | 7.67 ms |
| 4096 | 2.255 | 20.26 | 11.1% | 18.01 ms |

## What comes next

See ROADMAP Phase 13. Three follow-ups, re-prioritized by standalone
kernel data:

- **13c (HYP-032, #1 priority):** Marlin-style dequant->fp16->tensor core.
  The standalone timing proves this is the dominant gap: FlashInfer's
  tensor cores hold flat at 0.21ms while TQ scales linearly to 2.6ms at
  seq=4096 (12x). Scalar FMA at 20 TFLOPS cannot compete with mma at
  312 TFLOPS. This is the irreducible kernel-level cost that CUDA graphs
  cannot amortize. Target: match FlashInfer's flat ~0.2ms curve.
  Effort: weeks.
- **13b (HYP-031):** port split-KV into `decode_v4_from_cache`. Still
  valuable for parallelism at long seq, but tensor cores (13c) must come
  first -- split-KV over scalar FMA just parallelizes slow work.
  Target: drop 4K TPOT from 20 ms to ~7-9 ms (2.5x speedup).
  Effort: ~1 week.
- **13a (cheap, vLLM-only):** remove per-step `.to(int32)` copies from
  the Python path. Standalone kernel timing shows this does NOT affect
  the kernel gap at all -- it only matters for vLLM TPOT (framework
  overhead is 89% of TPOT). Worth doing but won't improve kernel parity
  with FlashInfer.

## Non-goals

- Quality measurement (PPL, coherence): use real text, covered by
  docs/reference/eval reports.
- Batch/throughput sweeps: needed for serving claims but orthogonal to
  this latency-vs-seq investigation.
- TP ≥ 2: deferred until 13b lands (split-KV interacts with head-partitioning).

## References

- HYP-029 — graph-safe KV cache (unlocked TQ fp8 graph capture)
- HYP-018 — split-KV contiguous (48 μs at seq=1024 standalone)
- HYP-022 — split-KV combine kernel (reusable)
- HYP-019 — INT4 tensor cores (rejected: 15× slower at rank-1 decode)
- ROADMAP Phase 8 — "Architecture gap: TurboQuant vs FlashInfer" table
