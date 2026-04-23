# HYP-062: `_tq_decode_stage1` joint launch retune (`num_warps`, `num_stages`, `BLOCK_KV`)

First Phase 3 code HYP. Patches **only** the launch-config of upstream
vLLM v0.20.0's `_tq_decode_stage1` Triton kernel via plugin
monkey-patch. No body changes; the same Triton source is JIT-compiled
under different launch parameters.

## Hypothesis

Upstream's defaults — `num_warps=1, num_stages=1, BLOCK_KV=4` — are
conservative. HYP-059 confirmed `_tq_decode_stage1` is the dominant TQ
kernel (79.4 % of TQ time). Increasing parallelism per CTA
(`num_warps`), pipeline depth (`num_stages`), and tile size
(`BLOCK_KV`) should reduce per-tile overhead and let HBM-load latency
overlap with compute, even without ncu warp-stall confirmation.

## Prediction

A 27-cell sweep over

```
num_warps  ∈ {1, 2, 4}
num_stages ∈ {1, 2, 3}
BLOCK_KV   ∈ {4, 8, 16}
```

run on each in-scope preset (`turboquant_4bit_nc`, `turboquant_k3v4_nc`,
`turboquant_3bit_nc`) finds **at least one config that lands ≥ 8 % TPOT
improvement at `4bit_nc × seq=8192 × concurrency=1`** (HYP-058 baseline
17.5 ms → ≤ 16.1 ms target).

Best-case prediction: a config in the `(num_warps=2, num_stages=2,
BLOCK_KV=8)` neighbourhood lands 12–15 % win at the same cell.

Side prediction: at `s1024 × c1` (lighter compute load) the win is
smaller (3–6 %) since the kernel is launch-overhead-bound at short
sequences. At `c8` cells, the win is preserved or slightly larger
because more concurrent decode requests mean more kernel launches per
cell.

## Pass / fail / opt-out

- **Primary pass**: SHA-256 byte-exact parity with HYP-058 baseline on
  `turboquant_4bit_nc × small_balanced` (85 samples × prediction-string
  hash) AND ≥ 5 % TPOT improvement at `4bit_nc × s8192 × c1`.
- **Stretch pass**: ≥ 8 % TPOT improvement (matches Phase 3 ROADMAP
  prediction).
- **Soft pass**: parity holds AND best cell improves ≥ 3 % AND no cell
  regresses > 0.5 %. Useful even if not big enough for upstream PR yet.
- **Opt-out gate**: NOT NEEDED for this HYP. Launch-config retune is
  parity-preserving by construction (same Triton source, same arithmetic,
  same operand types). SHA-256 must hold — anything else is a bug.
- **Kill**: any cell breaks parity, OR no cell improves at all.

## Method

### Plugin layout (new files)

```
turboquant/
├── kernels/
│   ├── __init__.py                  (new)
│   └── decode_stage1.py             (new) — copy of upstream _tq_decode_stage1
│                                          + parameterized launcher
└── vllm_plugin.py                   (extend register() with monkey-patch)
```

`turboquant/kernels/decode_stage1.py` contains a verbatim copy of the
upstream Triton `_tq_decode_stage1` `@triton.jit` body, plus an
optimized launcher wrapper that reads
`(NUM_WARPS, NUM_STAGES, BLOCK_KV)` from environment variables
(`TQ_DECODE_NUM_WARPS`, `TQ_DECODE_NUM_STAGES`, `TQ_DECODE_BLOCK_KV`)
with upstream defaults as fallback. The wrapper signature is identical
to upstream's `triton_turboquant_decode_attention` so vLLM's caller
sees no change.

`turboquant/vllm_plugin.py:register()` adds a `_patch_triton_kernels()`
call that:
- imports `vllm.v1.attention.ops.triton_turboquant_decode`
- replaces module attr `triton_turboquant_decode_attention` with our
  patched version (only when env `TQ_PATCH_DECODE=1`)
- prints a one-line marker so we can confirm the patch hot-loaded

Plugin is no-op if `TQ_PATCH_DECODE` unset → identical to HYP-058.

### Sweep harness

`tests/bench_hyp062_sweep.sh`:

```bash
for NUM_WARPS in 1 2 4; do
  for NUM_STAGES in 1 2 3; do
    for BLOCK_KV in 4 8 16; do
      LABEL="w${NUM_WARPS}-s${NUM_STAGES}-b${BLOCK_KV}"
      TQ_PATCH_DECODE=1 \
      TQ_DECODE_NUM_WARPS=$NUM_WARPS \
      TQ_DECODE_NUM_STAGES=$NUM_STAGES \
      TQ_DECODE_BLOCK_KV=$BLOCK_KV \
        bash tests/bench_serve_upstream_entry.sh \
        # ... seq=8192 conc=1 turboquant_4bit_nc out=hyp062/${LABEL}.json
    done
  done
done
```

27 cells × ~5 min/cell ≈ 2.5 hours on 1 GPU. Pre-cell SHA-256 parity
check fast-fails the cell if predictions diverge.

### Forge job pattern

Same pattern as HYP-058:
- Image `ce745b54` (custom, default profile — no profiling needed)
- `--shared-nfs --disk-mount tq-models:/mnt/models`
- Repo staged at `/workspace/shared/tq-vllm020/turboquant/` (already there;
  scp the two new files + edited vllm_plugin.py)
- Entrypoint installs vllm v0.20.0 + plugin (`pip install -e
  /workspace/shared/tq-vllm020/turboquant`) so the entry-point hot-loads.

### Parity gate (per cell)

After each cell runs, SHA-256 the prediction strings vs
`/workspace/shared/vllm020_longbench/turboquant_4bit_nc-...json`. Mismatch
on any sample → mark cell as PARITY_BROKEN, skip perf measurement, log
the diff for debugging. Continue with remaining cells.

## Status: REJECTED on TPOT — upstream defaults are the local optimum

The 27-cell sweep found no config that improves TPOT by ≥ 5 % over the
HYP-058 baseline at `4bit_nc × s8192 × c1`. Three configs are within
±0.2 % of baseline (statistically tied); every other config is strictly
slower. **Best `(num_warps, num_stages, BLOCK_KV)` config is
`(1, 3, 4)` at TPOT 17.46 ms vs baseline 17.5 ms = +0.20 %** — well
inside measurement noise. Predicted 8–15 % miss is total.

Plugin infrastructure works (parity verified — see below). The
hypothesis was wrong about *which lever moves the kernel*.

### Forge run

- Job: `2d616eee` (`tq-hyp062-sweep`, SUCCEEDED 2026-04-23 15:50 UTC).
- Image: `ce745b54` (`tq-upstream-nightly:v4`), default profile.
- Plugin install verified — `[tq-plugin] patched
  triton_turboquant_decode_attention in 1 module(s)` printed at every
  cell startup, with the per-cell `(num_warps, num_stages, BLOCK_KV)`
  values echoed correctly. Monkey-patch reach is exactly 1 module
  (the upstream defining module); call-site bindings via `from X
  import Y` weren't separately picked up but the patched function
  runs anyway because vLLM accesses it via attribute lookup on the
  module, not via a local binding.

### Sweep table — TPOT median (ms), Δ vs HYP-058 baseline (17.5 ms)

| cell | TPOT | Δ | TTFT mean | tput |
|---|---:|---:|---:|---:|
| `w1-s3-b4` | 17.46 | **+0.20 %** | 764 | 42.9 |
| `w1-s2-b4` | 17.47 | +0.18 % | 764 | 42.9 |
| `w1-s1-b4` (upstream default) | 17.48 | +0.09 % | 902 | 41.0 |
| `w1-s2-b8` | 17.67 | −0.98 % | 771 | 42.5 |
| `w1-s1-b8` | 17.68 | −1.04 % | 904 | 40.6 |
| `w1-s3-b8` | 17.69 | −1.09 % | 768 | 42.5 |
| `w2-s2-b8` | 17.85 | −1.98 % | 767 | 42.2 |
| `w2-s1-b8` | 17.85 | −2.00 % | 759 | 42.3 |
| `w2-s3-b8` | 17.87 | −2.09 % | 765 | 42.2 |
| `w2-s3-b4` | 18.21 | −4.06 % | 763 | 41.6 |
| `w2-s1-b4` | 18.21 | −4.07 % | 769 | 41.6 |
| `w2-s2-b4` | 18.21 | −4.08 % | 769 | 41.5 |
| `w4-s3-b4` | 18.55 | −5.98 % | 774 | 40.9 |
| `w4-s2-b4` | 18.57 | −6.11 % | 769 | 41.0 |
| `w4-s1-b4` | 18.57 | −6.11 % | 768 | 40.9 |
| `w4-s3-b8` | 18.96 | −8.34 % | 768 | 40.3 |
| `w4-s1-b8` | 18.97 | −8.39 % | 764 | 40.3 |
| `w4-s2-b8` | 18.97 | −8.40 % | 767 | 40.3 |
| `w2-s1-b16` | 19.25 | −9.97 % | 768 | 39.9 |
| `w2-s3-b16` | 19.25 | −10.00 % | 759 | 40.0 |
| `w2-s2-b16` | 19.25 | −10.01 % | 766 | 39.9 |
| `w4-s1-b16` | 19.33 | −10.45 % | 762 | 39.8 |
| `w4-s2-b16` | 19.33 | −10.45 % | 755 | 39.9 |
| `w4-s3-b16` | 19.33 | −10.46 % | 766 | 39.7 |
| `w1-s1-b16` | 20.15 | −15.17 % | 765 | 38.5 |
| `w1-s2-b16` | 20.16 | −15.17 % | 765 | 38.5 |
| `w1-s3-b16` | 20.16 | −15.19 % | 766 | 38.5 |

### SHA-256 parity (winner config)

`w1-s3-b4` LongBench `small_balanced` × `turboquant_4bit_nc`:
**85 / 85 byte-exact** vs HYP-058 baseline. Plugin doesn't perturb
math, as predicted. (Parity is preserved by construction since the
patch only changes launch params, not the `@triton.jit` body.)

### Why the hypothesis missed

Three plausible reasons, none individually sufficient to explain it,
but together they cover the gap:

1. **Tile size already small for the workload.** With `BLOCK_KV=4`
   and `D=128`, each tile is 4 × 80 = 320 B of cache + 4 × {centroid,
   norm} loads. That's already a single warp's worth at one launch —
   bumping `num_warps` to 2 or 4 splits this tiny work across more
   warps that all need to coordinate, costing more in sync than it
   saves in load latency. The strong negative correlation between
   `num_warps` and TPOT (default 17.48 → w4 = 18.57, −6 %) confirms it.
2. **`BLOCK_KV=16` is a clean loss everywhere.** Even at `num_warps=1`,
   `BLOCK_KV=16` lands at 20.16 ms (−15 %). Larger tiles likely blow
   the smem budget for centroid + norm scratch, forcing register
   spills or tile splits Triton handles by serializing further. Per
   `triton_turboquant_decode.py:552` the upstream author chose 4
   intentionally — this sweep validates that choice.
3. **Workload is launch-overhead-bound, not compute-bound.** HYP-059
   showed `_tq_decode_stage1` is only 4.7 % of total kernel time and
   3.7 % of total decode time at this workload. The remaining 95 %+
   is GEMM and prefill flash-attn — neither is touched by this patch.
   By Amdahl, even halving the kernel would yield ~2 % TPOT win.
   At 0.2 % the patch doesn't even hit the kernel-only ceiling because
   the kernel is already at a register-pressure / smem-pressure /
   sync-overhead local optimum.

### Implications for Phase 3 strategy

The clean negative result reshuffles the ROADMAP:

- **HYP-063 (centroid SMEM pre-stage) — still worth trying but lowered priority.**
  HYP-063 changes a different mechanism (replaces per-tile HBM gather
  with one-time SMEM stage), so the result of HYP-062 doesn't predict
  HYP-063's outcome. But Amdahl's bound applies: even a perfect HYP-063
  caps at ~2 % TPOT improvement on this workload.
- **HYP-064 (midpoints pre-load in `_tq_fused_store_mse`) — re-rank way down.**
  Predicted 0.5–1 % was already small; HYP-059 confirmed
  `_tq_fused_store_mse` is 0.6 % of total kernel time. Improvement
  ceiling is ~0.3 %.
- **HYP-065 (adaptive `NUM_KV_SPLITS`) — promote to Phase 3 next-up.**
  This changes the GRID, not block params. At `concurrency=8` cells
  HYP-058 showed the largest TQ-vs-fp16 gaps (`3bit_nc × s8192 × c8`
  +73 % over fp16 = +35 ms). Adaptive split count attacks SM
  saturation rather than per-CTA performance — different lever, larger
  expected ROI.
- **Workload selection matters.** The `s8192 × c1` cell is decode-only
  + 1 request — the kernel is barely on the hot path. Future HYPs
  should target cells where TQ kernels are >10 % of decode time
  (longer context, more concurrent requests) to clear the Amdahl
  ceiling.
- **Don't rewrite the kernel body to chase 8–15 % at this cell.**
  HYP-066/067 (tl.dot) would massively change the kernel and might
  yield ~5 % at best on `s8192 × c1`. Defer until we've exhausted the
  cheaper options OR moved to a workload with a higher TQ%.

### Artefacts

- `/workspace/shared/hyp062_sweep/` (Forge NFS): 27 perf JSONs +
  `winner.txt` + `parity_winner.txt` + `acc_winner.json` +
  `hyp062_run.log`.
- Local mirror: `results/hyp062/perf_grid/{cell}/upstream-s8192-c1.json`
  + `sweep_summary.csv` + `winner.txt` + `parity_winner.txt` +
  `acc_winner.json`.
- Plugin code: `turboquant/kernels/decode_stage1.py` + extended
  `turboquant/vllm_plugin.py` (both kept in repo — the monkey-patch
  infrastructure works and will be reused for HYP-063+).
- Forge job ID: `2d616eee`.

### Follow-ups

- **[immediate]** Update ROADMAP.md to re-rank Phase 3 HYPs:
  HYP-065 promoted, HYP-064 demoted.
- **[infrastructure win]** Plugin monkey-patch pattern is proven —
  reuse for HYP-063, HYP-065 (different patches, same wiring).
- **[future]** When ncu becomes available (DCGM unblocked or H100),
  re-attempt HYP-062 with warp-stall data; it's possible a non-launch-config
  change (e.g. tweaking the centroid gather pattern from inside the
  kernel) could clear the local-optimum ceiling that this sweep ran into.
