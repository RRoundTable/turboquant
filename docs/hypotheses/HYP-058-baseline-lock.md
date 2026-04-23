# HYP-058: Baseline lock — upstream Triton TurboQuant on A100, fresh capture

Phase 1 of the upstream-improvement track (`docs/ROADMAP.md`). Pure
measurement; no plugin code, no Triton edits. Captures the per-kernel
profiling artefacts and SHA-256 parity reference that gate every
subsequent Phase 3+ HYP.

## Hypothesis

Upstream vLLM v0.20.0's native TurboQuant kernels reproduce HYP-057's
LongBench numbers on a fresh Forge capture, the per-kernel runtime is
dominated by `_tq_decode_stage1` at long context, and the dominant
warp stall in that kernel is `long_scoreboard` (HBM-load latency from
the per-tile centroid gather + the `BLOCK_KV=4 / num_warps=1 /
num_stages=1` default launch config).

If true, this confirms the optimization axes named in
`docs/reference/upstream-triton-kernel-baseline.md` and unblocks
HYP-062 (joint launch retune, predicted 8–15 % TPOT) and HYP-063
(centroid SMEM pre-stage, predicted 2–4 % additive).

## Prediction

### Accuracy parity (LongBench `small_balanced`, Llama-3.1-8B-Instruct)

Match HYP-057 §Results within ≤ 0.005 pp per task per preset. SHA-256
of fp16 prediction strings must match HYP-057's `auto` artefact
byte-for-byte (greedy decode, fixed seed, identical tokenizer and
chat-template gating).

| preset | predicted Δ vs HYP-057 |
|---|---:|
| `auto` (fp16) | byte-exact (SHA-256 match) |
| `turboquant_4bit_nc` | ≤ ±0.005 pp per task |
| `turboquant_k3v4_nc` | ≤ ±0.005 pp per task |
| `turboquant_3bit_nc` | ≤ ±0.005 pp per task |

`turboquant_k8v4` is **not run** — known broken on A100 (HYP-057
§k8v4); upstream issue, out of scope per `docs/GOAL.md`.

### Performance baseline (16-cell grid)

`{4bit_nc, k3v4_nc, 3bit_nc, auto}` × `seq ∈ {1024, 8192}` ×
`concurrency ∈ {1, 8}` → median TTFT, TPOT, throughput per cell. No
quantitative prediction — this *is* the reference for every Phase 3
HYP. Expectation: TPOT for `turboquant_4bit_nc × seq=8192 × conc=1`
lands somewhere in the 70–90 ms range (HYP-057 measured 71.8 ms on
Qwen3-8B at the same cell; Llama-3.1-8B at `Hq=32, D=128` is in
roughly the same ballpark).

### Profiling (4bit_nc × small_balanced, single representative kernel launch)

For `_tq_decode_stage1`:
- **Predicted dominant stall**: `long_scoreboard` (HBM latency from
  per-tile centroid gather + low pipeline depth). Quantitative
  prediction: ≥ 35 % of total stall cycles.
- **Predicted register budget**: 32–64 regs/thread (Triton default,
  no spilling).
- **Predicted occupancy**: 25–50 % of theoretical (under-occupied at
  `num_warps=1`).

For `_tq_fused_store_mse`:
- **Predicted dominant stall**: `long_scoreboard` (per-iteration
  midpoints reload at `triton_turboquant_store.py:290`). ≥ 25 % of
  stall cycles.

If the measured stall distribution diverges from these predictions,
the Phase 3 HYP order in `docs/ROADMAP.md` gets reshuffled in HYP-059
analysis.

## Method

Single Forge job under `--security-profile profiling-debug`, 1 ×
A100-SXM4-40GB, ~3 h budget. Image `ce745b54` (`tq-upstream-nightly:v4`)
— same as HYP-057. Mounts: `--shared-nfs` (vLLM v0.20.0 source +
turboquant repo + outputs) + `--disk-mount tq-models:/mnt/models`
(HF cache + LongBench data, both pre-warmed by HYP-057).

Entrypoint runs four phases in order; each phase fails fast on missing
inputs:

1. **Probe + install** — verify NFS staging, install vLLM v0.20.0
   precompiled (`VLLM_USE_PRECOMPILED=1`), import-check the upstream
   `TurboQuantConfig`.
2. **Accuracy** — `tests/bench_longbench_vllm.py` for each of the four
   in-scope presets on `small_balanced`. SHA-256 the resulting fp16
   prediction strings → `sha256_fp16.json`.
3. **Profiling** — nsys timeline (`-t cuda,nvtx`) on a tiny
   `samples-per-task=5` run; ncu (`--section
   {WarpStateStatistics,SpeedOfLight,SpeedOfLight_RooflineChart,MemoryWorkloadAnalysis,Occupancy}
   --kernel-name regex:_tq_.* --launch-skip 100 --launch-count 50`)
   on a `samples-per-task=3` run. Both target
   `turboquant_4bit_nc`.
4. **Perf grid** — 16 cells via `tests/bench_serve_upstream_entry.sh`
   (per-cell `vllm serve` + `vllm bench serve`). All cells share the
   same Llama-3.1-8B-Instruct model + `tq-models` HF cache.

Outputs land under `/workspace/shared/hyp058_phase1/`:

```
hyp058_phase1/
├── acc_auto.json
├── acc_turboquant_4bit_nc.json
├── acc_turboquant_k3v4_nc.json
├── acc_turboquant_3bit_nc.json
├── sha256_fp16.json                  # parity reference for Phase 3+
├── trace.nsys-rep                    # nsys timeline
├── decode.ncu-rep                    # ncu profile (all _tq_* kernels)
├── perf_grid/                        # 16 cells × {.json, .server.log, .bench.log}
└── hyp058_run.log                    # entrypoint stdout
```

## Pass / fail

- **Primary pass**: each in-scope preset's `mean_score` per task lands
  within ≤ 0.005 pp of HYP-057's measured value, AND the fp16 SHA-256
  matches HYP-057's `acc_auto.json` byte-for-byte.
- **Secondary pass**: `decode.ncu-rep` opens cleanly in ncu and contains
  ≥ 1 sample per `_tq_decode_stage1` and `_tq_fused_store_mse`.
- **Soft pass / informational**: warp-stall distribution matches the
  prediction (`long_scoreboard` dominant in both kernels). If
  rejected, document the actual stall class — Phase 2 HYPs handle the
  reshuffle, this HYP still passes on the primary criteria.
- **Fail**: any preset > 0.005 pp drift from HYP-057, OR ncu output
  empty, OR fp16 SHA-256 mismatches HYP-057. Diagnose before
  proceeding to Phase 2.

## Status: pending

### Artefacts (filled on completion)

- Forge job ID: TBD
- `/workspace/shared/hyp058_phase1/` contents: TBD
- Per-preset accuracy table (vs HYP-057 reference): TBD
- 16-cell perf grid summary: TBD
- ncu warp-stall percentages for `_tq_decode_stage1` + `_tq_fused_store_mse`: TBD
- Populated cells in `docs/reference/upstream-triton-kernel-baseline.md`: TBD
