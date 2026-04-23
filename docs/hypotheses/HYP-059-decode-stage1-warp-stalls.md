# HYP-059: `_tq_decode_stage1` warp-stall attribution + occupancy ceiling

Phase 2 of the upstream-improvement track. Pure measurement. Target:
populate the TBD cells in
`docs/reference/upstream-triton-kernel-baseline.md` for
`_tq_decode_stage1` and resolve the optimization-axis ranking that
gates HYP-062 (joint launch retune) and HYP-063 (centroid SMEM
pre-stage).

Companions HYP-060 (`_tq_fused_store_mse`) and HYP-061
(`_tq_full_dequant_kv`) reuse the same Forge job — one ncu run profiles
all `_tq_*` kernels at once.

## Hypothesis

In upstream vLLM v0.20.0's default-config `_tq_decode_stage1`
(`num_warps=1, num_stages=1, BLOCK_KV=4` per
`triton_turboquant_decode.py:586-587,552`), HBM-load latency dominates
warp time. Concretely:

1. **`long_scoreboard` is the dominant stall class** — at least 35 % of
   active-warp cycles are waiting on a memory load to retire. Source:
   the per-tile centroid gather at `triton_turboquant_decode.py:193-197`
   reissues HBM reads each `BLOCK_KV` iteration, and `BLOCK_KV=4` keeps
   the loop tile small so the load is repeatedly exposed.
2. **Occupancy is well below SoL ceiling** — `num_warps=1` means each
   CTA holds 1 warp; A100's per-SM warp ceiling is 64. Active-warp
   occupancy lands in 25–50 % of theoretical (limited by either the
   kernel's smem footprint or `num_warps` directly).
3. **Register spills are 0** — Triton at `num_warps=1` typically lands
   in 32–64 regs/thread on this kernel; spill stores/loads should be 0.

If the stall distribution lines up with prediction (1), HYP-062's
launch retune (`num_warps↑`, `num_stages↑`, `BLOCK_KV↑`) attacks the
right problem and the predicted 8–15 % TPOT improvement is plausible.
If `long_scoreboard` is < 35 %, the optimization axis re-ranks and
HYP-062's prediction has to be revisited before code is written.

## Prediction

| metric | predicted value | source |
|---|---|---|
| `long_scoreboard` % of stall cycles | **≥ 35 %** | per-tile HBM gather + small `BLOCK_KV` |
| `short_scoreboard` % | < 10 % | `scores`/`acc` smem layout is dense, no obvious bank conflict |
| `math_pipe_throttle` % | < 5 % | scalar fp32 dot at this kernel; tensor cores not engaged (HYP-066/067 territory) |
| `wait` % (`__syncthreads`) | < 5 % | `num_warps=1` → no inter-warp sync inside the CTA |
| register budget | 32–64 regs/thread | Triton default at `num_warps=1` |
| spill stores / spill loads | 0 / 0 | no register pressure expected |
| occupancy (active warps / SM peak) | 25–50 % | bounded by `num_warps=1` × CTA count |

If reality lands in these ranges, write Status: CONFIRMED. If
`long_scoreboard < 30 %` OR another stall class > 30 %, write Status:
REJECTED with the actual distribution + a re-ranking of Phase 3 HYPs.

## Method

Single Forge job (~30–60 min):

- Image: `nvcr.io/nvidia/pytorch:<tag>` (only family `profiling-debug`
  accepts; needed for ncu hardware counters per ADR-010).
- GPU: 1 × A100-SXM4-40GB.
- Mounts: `--shared-nfs` (vLLM source + bench scripts) +
  `--disk-mount tq-models:/mnt/models` (HF + LongBench cache).
- Profile: `--security-profile profiling-debug`.

Entrypoint phases:

1. Probe NFS staging (same paths as HYP-058).
2. **Source-install vLLM v0.20.0** from
   `/workspace/shared/tq-vllm020/vllm-v0.20.0`. nvcr image's torch
   ABI may not match precompiled wheels, so accept a 10–30 min compile
   if `VLLM_USE_PRECOMPILED=1` doesn't take.
3. **nsys timeline** (`-t cuda,nvtx`, `--stats=true`) on a tiny
   `samples-per-task=5` `turboquant_4bit_nc` workload. Confirms
   `_tq_*` kernels actually appear in the trace and gives a
   per-kernel time attribution table.
4. **ncu profile** (`--section
   {WarpStateStatistics,SpeedOfLight,SpeedOfLight_RooflineChart,MemoryWorkloadAnalysis,Occupancy}`,
   `--kernel-name regex:_tq_.*`, `--launch-skip 100 --launch-count 50`)
   on a `samples-per-task=3` workload. Skip-100 dodges vLLM warmup
   kernels; count-50 caps profile size + runtime.

Outputs to `/workspace/shared/hyp059_profile/`:

```
hyp059_profile/
├── trace.nsys-rep                    # nsys timeline (Tier 1)
├── decode.ncu-rep                    # ncu profile (Tier 2)
├── _nsys_throwaway.json              # bench output (discarded)
├── _ncu_throwaway.json               # bench output (discarded)
└── hyp059_run.log                    # entrypoint stdout
```

## Pass / fail

- **Primary pass**: ncu output contains ≥ 1 sample per `_tq_decode_stage1`,
  `_tq_fused_store_mse`, `_tq_full_dequant_kv`. Warp-stall + occupancy
  + register tables extractable.
- **Hypothesis pass**: `long_scoreboard` is the modal stall class for
  `_tq_decode_stage1` and ≥ 35 % of stall cycles.
- **Hypothesis kill**: any other stall class > 30 % OR `long_scoreboard`
  < 30 %. Phase 3 HYP order in `docs/ROADMAP.md` gets rewritten before
  HYP-062 starts.

## Status: PARTIAL — nsys kernel-time attribution captured, ncu blocked by Forge DCGM contention

The hypothesis-test substance is half-confirmed. Kernel-level
**timing** attribution (which kernel dominates) was captured via nsys
and matches the prediction. But ncu HW-counter sections
(WarpStateStats, Occupancy, MemoryWorkloadAnalysis) were blocked at
runtime by Forge cluster's DCGM (Data Center GPU Manager) holding the
GPU's perf counters — see "Forge job iterations" below.

The **Phase 3 ROI ranking** that this HYP gates is unblocked:
`_tq_decode_stage1` is the dominant TQ kernel by a wide margin (79.4%
of TQ kernel time). HYP-062 (joint launch retune of `_tq_decode_stage1`)
proceeds. Warp-stall attribution (`long_scoreboard` vs others) remains
TBD until Forge resolves DCGM coordination or we get H100 access.

### Forge job iterations

| attempt | image | profile | outcome |
|---|---|---|---|
| v1 | `ce745b54` (custom) | profiling-debug | image rejected by profile policy |
| v2 | nvcr/pytorch:25.01 | profiling-debug | `setuptools_scm` missing in build env |
| v3 | nvcr/pytorch:25.01 | profiling-debug | same install path; `setuptools<77` was wrong fix |
| v4 | nvcr/pytorch:25.01 | profiling-debug | numpy 2 ABI breaks cv2 (vllm pulled numpy 2.x as transitive) |
| v5 | nvcr/pytorch:25.01 | profiling-debug | `WarpStateStatistics` section name wrong |
| v6 | nvcr/pytorch:25.01 | profiling-debug | EngineCore CUDA re-init (fork in subprocess) |
| v7 | nvcr/pytorch:25.01 | profiling-debug | flash-attn .so ABI mismatch with nvcr torch |
| v8 | nvcr/pytorch:25.01 | profiling-debug | flash-attn rebuild failed |
| **v9 (final usable)** | nvcr/pytorch:25.01 | profiling-debug | nsys passed; ncu blocked: "Profiling failed because a driver resource was unavailable" — DCGM/CUPTI contention |

Job IDs: `17b7bee7` (v1), `51a5d63c` (v2), `27e2d42e` (v3), `b36d9278`
(v4), `ba4dac92` (v5), `a3a7f8d9` (v6), `097c8802` (v7), `96382bf6`
(v8), `a10d3ae4` (v9 — partial pass).

Fixes pinned for the next profiling job:
`setuptools>=77`, numpy `>=1.26,<2`, vllm install with `--no-deps`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `pip uninstall flash-attn` plus
`VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`, `--section WarpStateStats`
(no "istics"). All baked into `/tmp/hyp059_entry.sh`.

### Results — kernel-time attribution (nsys, A100, samples-per-task=5)

Per `_tq_*` kernel:

| kernel | n_calls | total ms | avg μs | % of TQ time | % of total kernel time |
|---|---:|---:|---:|---:|---:|
| `_tq_decode_stage1` | 2,576 | 990.5 | 384 | **79.4 %** | 3.7 % |
| `_tq_fused_store_mse` | 2,856 | 153.8 | 54 | 12.3 % | 0.6 % |
| `_tq_full_dequant_kv` | 1,372 | 103.9 | 76 | 8.3 % | 0.4 % |

Total TQ kernel time: 1,248.2 ms / 26,593.1 ms total = **4.7 % of
all GPU kernel time** for this small_balanced + samples-per-task=5
workload.

Top-12 kernels overall (full table at `results/hyp059/kernel_top20.csv`):

1. `ampere_fp16_s16816gemm_fp16_128x256_*` — 36.9 % (model linear layers, GEMM)
2. `flash::flash_fwd_kernel<...>` — 23.0 % (prefill attention)
3. `cutlass_80_tensorop_f16_s16816gemm_relu_*` — 13.9 %
4. `ampere_fp16_s16816gemm_fp16_256x128_*` — 5.4 %
5. `flash_fwd_splitkv_kernel` — 3.8 % (decode attention for fp16-equivalent path)
6. **`_tq_decode_stage1`** — 3.7 %
7. `vllm::act_and_mul_kernel` — 2.1 %
8. (smaller GEMMs / RMSNorm / rotary / cat / others)

### Implications for Phase 3+

- **`_tq_decode_stage1` dominance (79.4 % of TQ time) confirmed.** HYP-062
  (joint launch retune) and HYP-063 (centroid SMEM pre-stage) target the
  right kernel. Predicted 8–15 % TPOT improvement at HYP-058's
  `4bit_nc × s8192 × c1` cell (17.5 ms baseline → 14.9–16.1 ms target)
  remains plausible.
- **`_tq_fused_store_mse` is small (12.3 % of TQ).** HYP-064's predicted
  0.5–1 % TPOT win is right-sized — not worth re-ordering before HYP-062/063.
- **`_tq_full_dequant_kv` is rarely-hit (8.3 %).** Continuation prefill
  isn't the hot path on `small_balanced`. Defer HYP-061's deep dive.
- **Warp-stall attribution stays a follow-up.** Without ncu, HYP-062's
  sweep is empirical — pick the winner config from TPOT data and
  document the *measured* win, even if we can't yet say which stall
  class it relieved. Revisit ncu when DCGM coordination is fixed
  upstream.

### Artefacts

- `/workspace/shared/hyp059_profile/` (Forge NFS): `trace.nsys-rep` (38 MB)
  + `trace.sqlite` (119 MB) + `hyp059_run.log` (200 KB).
- Local mirror: `results/hyp059/trace.nsys-rep` + `kernel_top20.csv`
  + `hyp059_run.log`.
- Forge job ID: `a10d3ae4`.

### Follow-ups

- **[infrastructure]** File a Forge ticket: ncu HW counter access blocked
  by DCGM despite `--security-profile profiling-debug`. Workaround
  request: pause DCGM during ncu profiling job runs, or expose
  `dcgmi profile --pause` to user.
- **[deferred to HYP-067]** Re-attempt ncu when H100 access lands; H100
  + Hopper profiling tooling has different DCGM coordination.
- **[merged into HYP-062]** No separate HYP-060/061 needed — the kernel
  time data here resolves the question they would have answered.
