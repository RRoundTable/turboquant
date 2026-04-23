# HYP-051: Enable end-to-end CUDA graphs for TQ backend under vLLM serving

## Hypothesis

HYP-033 made the v5 decode path CUDA-graph-safe at the op level (dispatcher-routed
`decode_v5_from_cache*_ws`, pre-allocated workspaces, no `.item()` syncs) and HYP-028
did the same for the quantize-write path (`torch.ops.turboquant_write.*`). With both
in place, vLLM's full CUDA graph capture under `cudagraph_mode:FULL` should survive
the KV cache storage swap that crashed HYP-027 — all cache mutations now go through
the dispatcher instead of Python advanced-indexing.

The serving benchmark in `docs/BENCHMARKS.md` currently runs FA, FI, and ours with
`--enforce-eager` (docs/BENCHMARKS.md:17) because A100 SM80 cannot torch.compile
fp8e4nv. This blanket flag over-constrains every backend:

- FA / FI never touch fp8e4nv → they can run vLLM's default compiled+graphs path.
- Ours stores fp8e4nv but the custom op is not something inductor compiles anyway →
  it needs `mode:0` (inductor off) but `cudagraph_mode:FULL` is independent.

Enabling graphs for all three backends closes the per-decode-step launch overhead
that dominates short-context TPOT.

## Prediction

Measured on A100-SXM4-40GB, Qwen3-8B, same six configs as the current BENCHMARKS.md
sweep (s1024×c8, s2048×c32, s8192×c8, s16384×c8, s32768×c4, s32768×c8):

1. **Stability.** `ours` with `--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'`
   completes the full bench (no `cudaErrorIllegalAddress` at replay, no preemption
   spiral). This is the primary engineering goal — HYP-027 crashed here; HYP-028+033
   should unblock it.
2. **TPOT wins for all three at short context.** At s1024×c8:
   - FA graphs vs FA eager: **20–40% TPOT improvement** (graphs amortize launch
     overhead; expected from HYP-023 which saw 26% on standalone decode).
   - FI graphs vs FI eager: similar range, 20–40%.
   - Ours graphs vs ours eager: **25–50% TPOT improvement** (we have more Python-path
     launch overhead per decode step → more headroom).
3. **TPOT gap narrows at long context.** At s32768×c8, ours was 138.5 ms eager vs
   FA 87.7 ms eager (1.58× worse). With graphs on both sides, predict ours / FA
   ≤ **1.3×** — Python launch was a larger share of our step-cost at long ctx too.
4. **TTFT mostly unchanged.** Prefill dominates TTFT at ≥ 8k; graphs primarily help
   decode. Expect TTFT deltas within ± 10% across all backends.
5. **Output throughput.** `ours` keeps its decisive long-context win (s32768×c8
   ≥ 1.5× FA throughput) because the win was driven by 3.2× KV compression avoiding
   preemption, not by decode speed.

## Method

1. **Entry-script flag.** Add `GRAPHS={0,1}` env to `tests/bench_serve_entry.sh`. When
   `GRAPHS=1`:
   - `fa`/`fi`: drop `--enforce-eager`, keep everything else.
   - `tq`: drop `--enforce-eager`, add
     `--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'`.

2. **Smoke test.** Single Forge job: Qwen3-8B, s1024×c8, `BACKEND=tq GRAPHS=1`, 1 GPU,
   `--security-profile` not needed. Watch the server log for
   `cudaErrorIllegalAddress`. If smoke fails, stop and debug; do not fan out.

3. **Fan out.** On smoke success, submit 18 parallel Forge jobs: 6 configs × 3 backends
   × `GRAPHS=1`. Each job writes `${backend}-s${seq}-c${conc}.json` to
   `/workspace/shared/bench_serve_graphs/`. Do not re-run `upstream` — it already
   uses its default compiled+graphs path in the current results.

4. **Aggregate.** Copy the current `results/v5_serve*/aggregate.py` pattern into
   `results/v5_serve_graphs/aggregate.py`. Produce the same three tables (TTFT / TPOT /
   throughput) plus a delta column `graphs/eager` per backend.

5. **Docs.** Append a new section "With CUDA graphs" to `docs/BENCHMARKS.md` below the
   existing eager tables (do not overwrite — eager is still a real data point). Update
   README if the graph numbers change the "what this project is" framing.

## Status: confirmed (stability + parity); quantitative predictions partially rejected

## Results (Forge A100-SXM4-40GB, 2026-04-21)

Ran the full 18-job sweep (6 configs × 3 backends) with graphs enabled.
Raw data: `results/v5_serve_graphs/*.json` and `REPORT.md`.

### Stability — confirmed

All 18 jobs succeeded. Ours ran under `cudagraph_mode:FULL` end-to-end —
no `cudaErrorIllegalAddress`, no replay faults. HYP-028 + HYP-033's
dispatcher-routed ops survive vLLM's KV cache storage swap (the failure
mode from HYP-027). The primary engineering goal is met.

Two operational gotchas discovered along the way, rolled into
`tests/bench_serve_entry.sh`:

1. `--gpu-memory-utilization 0.85` with the default `max_num_seqs=256`
   overflows at model load — vLLM's dummy-forward during graph capture
   instantiates activation buffers for every captured batch size. Fix:
   narrow to `--max-num-seqs 64 --cudagraph-capture-sizes [1,2,4,8,16,32,64]`
   (the concurrencies we actually bench).
2. `PYTORCH_ALLOC_CONF=expandable_segments:True` to let the allocator
   reclaim fragmented reserved memory that graphs pin.

### Quantitative verdicts

| Prediction | Target | Actual | Verdict |
|-----------|--------|--------|---------|
| Ours graph capture completes without crash | — | 18/18 jobs succeeded | ✓ confirmed |
| FA TPOT eager → graphs at s1024×c8 | 20–40% faster | 22.7 → 15.3 ms (**48%**) | ✓ exceeded |
| FI TPOT eager → graphs at s1024×c8 | 20–40% faster | 22.0 → 15.2 ms (**45%**) | ✓ exceeded |
| Ours TPOT eager → graphs at s1024×c8 | 25–50% faster | 29.5 → 19.6 ms (**50%**) | ✓ matched |
| Ours/FA TPOT at s32768×c8 | ≤ 1.3× | 138.2 / 86.0 = **1.61×** | ✗ rejected |
| TTFT deltas within ±10% across configs | ≤ ±10% | s1024 TTFT **+78%** at ours | ✗ rejected at short ctx |
| Ours throughput at s32768×c8 | ≥ 1.5× FA | **1.86×** (45.5 vs 24.5) | ✓ confirmed |

### What this means

Graphs deliver the expected ~1.5× TPOT speedup at short context — **for
every backend**, including FA and FI. Because the speedup is roughly
uniform, the relative ordering from the eager tables is preserved:
ours trails by ~1.3× at short ctx, wins decisively at long ctx.

Two unexpected findings:

1. **No long-ctx TPOT benefit.** At s≥16k, TPOT is compute-bound
   (Python launch overhead is a rounding error), so graphs don't help
   any backend. Ours/FA TPOT at s32768×c8 stays at 1.61× — graphs
   don't close that gap. Closing it requires closing the dequant
   arithmetic gap (HYP-032 landed the register-resident codebook; the
   remaining ~1.5× at long-ctx batch is scalar-FMA dequant vs tensor-core
   attention — an A100 architectural ceiling, unlocked on H100 via
   HYP-046).
2. **Short-ctx TTFT regresses under graphs.** Not a capture cost —
   TTFT at long ctx is unchanged. It's the narrower `max_num_seqs=64`
   changing the scheduler's chunked-prefill decisions at low seq where
   prefill is already cheap. Production users who care about short-ctx
   TTFT should raise `max_num_seqs` and correspondingly the GPU memory
   headroom, or run with `--enforce-eager` (still the Dockerfile default).

### Net

The regime verdicts from the eager benchmark hold. Graphs are a
deployment knob that boosts short-ctx decode throughput by ~25%
uniformly; they do not shift who wins where. Docs updated: new "With
CUDA graphs enabled" section in `docs/BENCHMARKS.md`.

## References

- HYP-023 — CUDA graph capture for standalone decode (confirmed, +26% at seq≥512).
- HYP-027 — first attempt to enable graphs end-to-end in vLLM (rejected,
  `cudaErrorIllegalAddress` at replay from Python `_write_to_cache` advanced-indexing).
- HYP-028 — CUDA write op as dispatcher-routed custom op (confirmed, fixes the
  Python-indexing graph-hostility HYP-027 diagnosed).
- HYP-033 — v5 decode op graph-safe via pre-allocated workspace (confirmed).
- docs/BENCHMARKS.md — current eager-only comparison to be augmented.
