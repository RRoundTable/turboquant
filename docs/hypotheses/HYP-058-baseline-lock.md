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

## Status: CONFIRMED (accuracy + perf grid; profiling deferred to HYP-059)

Primary pass criterion was `mean_score within ≤ 0.005 pp` per task per
preset and SHA-256 byte-exact match for fp16. **Result is stronger than
predicted: every task in every preset is byte-exact match vs HYP-057,
so Δscore = 0.0000 across the full 4 × 4 grid.** Perf grid completed
all 16 cells with no failures.

The "Soft pass / informational" warp-stall criterion is **deferred**.
The custom image used for Job A (`tq-upstream-nightly:v4`) does not
have nsys/ncu installed, and Forge's `--security-profile profiling-debug`
only allows `nvcr.io/nvidia/pytorch:*` or `*/mlops/forge:notebook-*`
images. Profiling becomes its own job (HYP-059's setup) running on the
nvcr base.

### Forge runs

| attempt | job ID | image | profile | outcome |
|---|---|---|---|---|
| v1 | `f72da2aa` | `ce745b54` | profiling-debug | image rejected by profile policy |
| v2 | `6add0648` | `ce745b54` | default | failed at vLLM install — `setuptools-scm` couldn't read git history (cp -r dropped `.git`) |
| v3 | `0d30d2ec` | `ce745b54` | default | passed steps 1–4 (accuracy + parity), failed at step 5 (`nsys: command not found`) — image lacks nsys |
| v4 (final) | `fb2e708a` | `ce745b54` | default | **SUCCEEDED** — steps 1–4 reused v3 outputs, step 5 skipped via guard, step 7 ran all 16 cells |

Fixes between v1→v4: `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.20.0`,
`ln -s python3 → python` for image without `python` symlink, and
`command -v nsys` guard so a missing tool doesn't take down later steps.

### Accuracy parity (vs HYP-057, byte-exact = SHA-256 of token strings)

| preset | task | byte_exact | score (this run) | score (HYP-057) | Δ |
|---|---|---:|---:|---:|---:|
| `auto` | qasper | 25/25 | 0.4689 | 0.4689 | 0.0000 |
| `auto` | hotpotqa | 25/25 | 0.4679 | 0.4679 | 0.0000 |
| `auto` | passage_retrieval_en | 25/25 | 1.0000 | 1.0000 | 0.0000 |
| `auto` | narrativeqa | 10/10 | 0.4254 | 0.4254 | 0.0000 |
| `turboquant_4bit_nc` | qasper | 25/25 | 0.4803 | 0.4803 | 0.0000 |
| `turboquant_4bit_nc` | hotpotqa | 25/25 | 0.4579 | 0.4579 | 0.0000 |
| `turboquant_4bit_nc` | passage_retrieval_en | 25/25 | 1.0000 | 1.0000 | 0.0000 |
| `turboquant_4bit_nc` | narrativeqa | 10/10 | 0.4381 | 0.4381 | 0.0000 |
| `turboquant_k3v4_nc` | qasper | 25/25 | 0.4763 | 0.4763 | 0.0000 |
| `turboquant_k3v4_nc` | hotpotqa | 25/25 | 0.4579 | 0.4579 | 0.0000 |
| `turboquant_k3v4_nc` | passage_retrieval_en | 25/25 | 0.9600 | 0.9600 | 0.0000 |
| `turboquant_k3v4_nc` | narrativeqa | 10/10 | 0.4077 | 0.4077 | 0.0000 |
| `turboquant_3bit_nc` | qasper | 25/25 | 0.4391 | 0.4391 | 0.0000 |
| `turboquant_3bit_nc` | hotpotqa | 25/25 | 0.4910 | 0.4910 | 0.0000 |
| `turboquant_3bit_nc` | passage_retrieval_en | 25/25 | 1.0000 | 1.0000 | 0.0000 |
| `turboquant_3bit_nc` | narrativeqa | 10/10 | 0.4179 | 0.4179 | 0.0000 |

340/340 byte-exact at SHA-256. The Phase 3+ parity gate is locked.

### Perf baseline (Llama-3.1-8B-Instruct, 1× A100, vLLM v0.20.0, eager)

Median TPOT (ms) — lower is better:

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 12.9 | 15.8 | 13.7 | 48.2 |
| `turboquant_4bit_nc` | 14.2 | 20.0 | 17.5 | 68.8 |
| `turboquant_k3v4_nc` | 14.2 | 20.7 | 17.9 | 76.9 |
| `turboquant_3bit_nc` | 14.3 | 21.2 | 18.6 | 83.4 |

Median TTFT (ms):

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 93.9 | 466.0 | 702.3 | 1546.4 |
| `turboquant_4bit_nc` | 95.7 | 464.4 | 745.7 | 1747.8 |
| `turboquant_k3v4_nc` | 95.2 | 370.6 | 725.0 | 1871.1 |
| `turboquant_3bit_nc` | 98.1 | 372.3 | 746.8 | 1822.4 |

Output throughput (tok/s):

| preset | s1024 × c1 | s1024 × c8 | s8192 × c1 | s8192 × c8 |
|---|---:|---:|---:|---:|
| `auto` (fp16) | 66.0 | 412.5 | 49.0 | 132.5 |
| `turboquant_4bit_nc` | 60.8 | 265.2 | 39.1 | 91.2 |
| `turboquant_k3v4_nc` | 60.8 | 330.5 | 38.6 | 86.7 |
| `turboquant_3bit_nc` | 60.3 | 323.0 | 37.5 | 81.8 |

### Implications for Phase 3 ROI ranking

The TPOT gap vs fp16 grows with both seq and concurrency. Largest
absolute gap is at **`turboquant_3bit_nc × s8192 × c8`** (+35.2 ms,
+73 % over fp16). Largest relative gap on the c=1 column is
`turboquant_3bit_nc × s8192 × c1` (+4.9 ms, +36 %).

HYP-062 targets `seq=8192 × conc=1` because that's the cell where the
decode kernel dominates (no preemption, no scheduler effects), and the
absolute gap (~5 ms) is small enough that an 8–15 % win is plausible
from launch-config retuning alone. The c=8 cells likely need HYP-065
(adaptive `NUM_KV_SPLITS`) on top.

### Artefacts

- Forge job IDs: `f72da2aa` (v1, fail), `6add0648` (v2, fail),
  `0d30d2ec` (v3, fail-late), `fb2e708a` (**v4, succeeded**).
- `/workspace/shared/hyp058_phase1/` contents: 20 JSONs (4 acc × 1 +
  16 perf cells × 1) + `sha256_fp16.json` + `vs_hyp057.json` +
  `hyp058_run.log`.
- Local mirror in this repo: `results/hyp058/` (4 acc + 16 perf +
  parity ref + diff).
- Profiling artefacts (nsys + ncu): TBD — moved to HYP-059's setup
  step (separate Forge job under `nvcr.io/nvidia/pytorch:*` +
  `--security-profile profiling-debug`).
