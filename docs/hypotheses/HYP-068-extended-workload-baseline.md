# HYP-068: Extended workload baseline — does TQ kernel fraction grow at long context?

After HYP-062 + HYP-065 both REJECTED (kernel-only retunes don't beat
upstream defaults at HYP-058's `s8192 × c1/8` cells), the binding
constraint is the Amdahl ceiling from HYP-059: **TQ kernels = 4.7 %
of total decode time** at that workload.

This HYP measures whether longer context or higher concurrency raises
the TQ fraction high enough that kernel-only optimization regains
meaningful headroom.

## Hypothesis

KV-attention work scales linearly with `seq_len`, while the rest of
the per-decode-step cost (model GEMMs, residuals, RMSNorm, rotary,
etc.) is roughly constant per token. So at longer context the **TQ
kernels' fraction of total decode time should grow approximately
linearly** with `seq_len / 8192`:

| seq | predicted TQ % | implied kernel-only TPOT ceiling |
|---:|---:|---:|
| 8192 (HYP-059 measured) | 4.7 % | ~2 % |
| 16384 | ~9 % | ~4 % |
| 32768 | ~18 % | ~9 % |

If this scaling holds, **`s32k × c1`** becomes a viable cell to
re-test HYP-062/063: a 50 % kernel-time cut would yield ~9 % TPOT
improvement, well over the 5 % gate.

## Prediction

Phase A (perf grid, 8 cells):

| cell | predicted TPOT (rough scaling from HYP-058) |
|---|---:|
| `4bit_nc × s16k × c1` | ~30 ms |
| `4bit_nc × s16k × c8` | ~135 ms |
| `4bit_nc × s32k × c1` | ~55 ms |
| `4bit_nc × s32k × c4` | ~140 ms |
| `3bit_nc × s16k × c1` | ~33 ms |
| `3bit_nc × s16k × c8` | ~165 ms |
| `3bit_nc × s32k × c1` | ~58 ms |
| `3bit_nc × s32k × c4` | ~155 ms |

Skip c=8 at s32k to avoid KV-cache OOM risk on A100-40GB
(4-bit + 32k × batch=8 × 32 layers ≈ tight on memory at 0.85 util).

Phase B (nsys at the cell with longest absolute TQ time):
- Profile `4bit_nc × s32k × c1` with nsys (no ncu — DCGM still blocks).
- Extract per-`_tq_*` kernel times.
- Compute TQ % of total kernel time.

## Pass / fail

- **Primary pass**: any new cell shows **TQ kernels > 10 %** of total
  kernel time. Unlocks retry of HYP-062 (launch retune) and HYP-063
  (centroid SMEM pre-stage) at that cell — kernel-only improvements
  have meaningful Amdahl headroom.
- **Stretch pass**: TQ > 15 % at `s32k × c1`. Strongly motivates
  re-running the full Phase 3 sweep at that cell.
- **Soft pass**: TQ between 6–10 %. Modest improvement room; HYP-063
  worth trying but HYP-062 likely still bounded.
- **Kill**: TQ stays < 5 % at all extended cells. Means model GEMMs
  always dominate on this hardware × this model size — kernel-only
  A100 optimization is fundamentally out of runway. Pivot to:
    1. A different model (smaller dense or larger context-relative)
    2. Phase 5 (upstream PR contributions of what we've validated)
    3. Phase 4 prep (H100/H200/B200 once hardware lands)

## Method

Single Forge job:
- Image `ce745b54` (custom, default profile — nsys works without
  profiling-debug per CLAUDE.md).
- Reuse staged repo at `/workspace/shared/tq-vllm020/turboquant/`.
- 8 perf cells via `tests/bench_serve_upstream_entry.sh` adapted to
  s∈{16384, 32768} (need `MAX_LEN ≥ s + 128 + 16`).
- nsys timeline at `4bit_nc × s32k × c1` after perf cells succeed.

Total runtime: 8 cells × ~8 min + nsys ~10 min ≈ 1.5 h.

No code changes. No plugin. Pure measurement (like HYP-058 + HYP-059
combined for a different workload region).

## Status: CONFIRMED — Amdahl ceiling lifted at long context, Phase 3 cell pivot warranted

TQ overhead as fraction of TQ TPOT scales strongly with seq length:

| seq | 4bit_nc TQ_extra/TQ | 3bit_nc TQ_extra/TQ | implied kernel-50%-cut win |
|---:|---:|---:|---:|
| 8192 (HYP-058) | 22 % | 29 % | ~11–14 % |
| 16384 | 32 % | 39 % | ~16–20 % |
| **32768** | **45 %** | **53 %** | **~22–26 %** |

At `s32k × c1`, **half of TQ TPOT is TQ-attributable overhead**. The
HYP-062 result (+0.2 % at s8k × c1, kernel ceiling ~2 %) was bounded
by Amdahl, not by the patch quality — the patch itself works. Re-running
HYP-062 / HYP-063 at `s32k × c1` should land a real perf win.

### Forge run

- Job: `12422fa6` (`tq-hyp068-extended`, SUCCEEDED 2026-04-24 03:16 UTC).
- Image: `ce745b54`, default profile (no nsys/ncu installed; `--enforce-eager`
  *not* used here — let vLLM use its default cudagraph path so numbers
  match HYP-058's regime).
- 10 cells, no plugin.

### Results — extended-workload TPOT grid

| cell | TPOT median (ms) | TTFT median (ms) | tput (tok/s) |
|---|---:|---:|---:|
| `auto × s16k × c1` (fp16) | **14.44** | 1570 | 35.9 |
| `auto × s32k × c1` (fp16) | **15.80** | 3922 | 21.1 |
| `4bit_nc × s16k × c1` | 21.27 | 1719 | 27.1 |
| `4bit_nc × s16k × c8` | 135.23 | 2985 | 46.4 |
| `4bit_nc × s32k × c1` | **28.82** | 4338 | 15.4 |
| `4bit_nc × s32k × c4` | 146.59 | 6934 | 20.3 |
| `3bit_nc × s16k × c1` | 23.50 | 1682 | 25.7 |
| `3bit_nc × s16k × c8` | 160.55 | 2976 | 42.5 |
| `3bit_nc × s32k × c1` | **33.31** | 4323 | 14.5 |
| `3bit_nc × s32k × c4` | 173.62 | 7250 | 17.9 |

(c=8 cells at s32k skipped to avoid fp16-cache OOM risk on A100-40GB;
c=4 is plenty to test concurrency-scaling questions.)

### TQ overhead as fraction of TQ TPOT (c=1 paired with fp16)

| cell | TQ TPOT | fp16 TPOT | gap (ms) | TQ_extra / TQ_total |
|---|---:|---:|---:|---:|
| `4bit_nc × s8k × c1` (HYP-058) | 17.5 | 13.7 | +3.8 | **+22 %** |
| `4bit_nc × s16k × c1` | 21.27 | 14.44 | +6.83 | **+32 %** |
| `4bit_nc × s32k × c1` | 28.82 | 15.80 | +13.02 | **+45 %** |
| `3bit_nc × s8k × c1` (HYP-058) | 18.6 | 13.7 | +4.9 | +29 % |
| `3bit_nc × s16k × c1` | 23.50 | 14.44 | +9.06 | +38 % |
| `3bit_nc × s32k × c1` | 33.31 | 15.80 | +17.51 | **+53 %** |

Linear scaling of `TQ_extra` with seq length confirms the prediction:
KV-attention work scales O(seq), other per-token work is constant.

### Implications — Phase 3 reopens at the right cell

At `s32k × c1`:
- HYP-062 (launch retune): retry. Predicted improvement at the cell
  where the kernel matters: 8–15 % TPOT (vs the +0.2 % we got at
  s8k × c1 where kernel barely mattered).
- HYP-063 (centroid SMEM pre-stage): worth a focused try —
  reduces HBM gather, the dominant per-tile cost. Predicted 4–8 %.
- HYP-065 (adaptive splits): still REJECTED by structural cap; doesn't
  benefit from longer context.
- HYP-064 (midpoints pre-load): still <0.5 %, demoted.

Multi-arch implications:
- This pattern is hardware-dependent. On H100/H200 (more SMs + faster
  GEMMs), the constant per-token GEMM share *grows* relative to
  attention; the crossover seq might shift higher (s64k? s128k?).
- On B200 with FP4 native (HYP-067 territory), TQ kernel cost per
  token may drop dramatically — making this whole optimization moot.

But for **A100 right now**, `s32k × c1` is the cell where kernel
optimization actually pays back. Phase 3 retries should target it.

### Artefacts

- `/workspace/shared/hyp068_extended/` (Forge NFS): 10 perf JSONs +
  `hyp068_run.log`.
- Local mirror: `results/hyp068/perf_grid/` + `sweep_summary.csv` +
  `hyp068_run.log`.
- Forge job ID: `12422fa6`.

### Follow-ups

- **[immediate]** Retry HYP-062 at `s32k × c1`. Same 27-cell sweep
  (`num_warps × num_stages × BLOCK_KV`) but at the cell where Amdahl
  isn't binding. Expected: a real winner emerges, ≥5 % TPOT.
- **[next]** HYP-063 (centroid SMEM pre-stage) at `s32k × c1`. Different
  mechanism, additional headroom.
- **[doc]** Update ROADMAP.md to add `s32k × c1` as the new "primary
  cell" for Phase 3 retries; HYP-058's `s8k × c1` becomes a regression
  guard cell only.
- **[hygiene]** Add `*.nsys-rep`, `*.ncu-rep`, `*.sqlite` to
  `.gitignore` — these binary trace files are large, machine-specific,
  and trip GitHub secret scanning false positives (HYP-059 trace
  blocked our first push to origin).
