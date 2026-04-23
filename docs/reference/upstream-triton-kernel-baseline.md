# Upstream Triton TurboQuant — kernel-level baseline

Per-kernel reference for upstream vLLM v0.20.0 (commit `579602aa4be6`)
TurboQuant Triton kernels. Captures launch geometry, smem layout, and
HBM traffic for each kernel; warp-stall attribution and register budget
cells are filled by Phase 2 (HYP-059/060/061).

This is the baseline that every Phase 3+ HYP optimizes against.
Companion to `vllm-upstream-turboquant-architecture.md` (which covers
the cross-kernel data flow + cache slot layout) — this doc zooms in on
each kernel as a profiling target.

> **Status — most "after Phase 2" cells are TBD.** Phase 1 (HYP-058)
> populates the launch grid and smem map cells from source. Phase 2
> (HYP-059/060/061) populates the register-budget, warp-stall, and
> occupancy cells from ncu output.

---

## 1. `_tq_fused_store_mse`

**File**: `vllm/v1/attention/ops/triton_turboquant_store.py:_tq_fused_store_mse`.
**Role**: Stage-2 of MSE store path. Bucketize Hadamard-rotated K to
MSE indices; pack; write packed bytes + `‖k‖₂`. Quantize V uniformly
(min/max scale + zero) and write packed bytes + scale + zero.

### Launch geometry

| field | value | source |
|---|---|---|
| grid | `(num_tokens, num_kv_heads)` | `triton_turboquant_store.py` |
| block | one program per (token, head) | implicit |
| `num_warps` | 1 (default) | `triton_turboquant_store.py:default` |
| `num_stages` | 1 (default) | `triton_turboquant_store.py:default` |
| `BLOCK_D` | `D` (head_dim, e.g. 128) | constexpr |
| `MSE_BITS` | 3 or 4 (preset-dependent) | constexpr |
| `VAL_BITS` | 3 or 4 (preset-dependent) | constexpr |
| register budget | TBD (Phase 2) | ncu `launch__registers_per_thread` |
| occupancy | TBD (Phase 2) | ncu `sm__warps_active.avg.pct_of_peak_sustained_active` |

### SMEM map (per program)

```
┌────────────────────────────────────────────────────────────┐
│ y_rotated[D]              fp32   — Hadamard-rotated K input │
│ centroid_midpoints[2^MSE_BITS-1] fp32 — Lloyd-Max midpoints │
│   (reloaded every binary-search iteration — HYP-064 target) │
│ packed_idx[ceil(D · MSE_BITS / 8)] uint8 — staged pack buf  │
│ v_min, v_max              fp32   — per-vector V min/max     │
└────────────────────────────────────────────────────────────┘
total: TBD bytes (Phase 2 — depends on Triton's actual allocation)
```

### HBM traffic per invocation (one (token, head))

| direction | bytes | source |
|---|---:|---|
| load `y_rotated` (post-cuBLAS GEMM) | `D · 4` | input arg |
| load `centroids` (`2^MSE_BITS` fp32) | `2^MSE_BITS · 4` × MSE_BITS iters (binary search) | hot loop — HYP-064 target |
| store packed K bytes | `ceil(D · MSE_BITS / 8)` | slot output |
| store `‖k‖₂` fp16 | 2 | slot output |
| load raw V | `D · 2` (fp16) | input arg |
| store packed V bytes | `ceil(D · VAL_BITS / 8)` | slot output |
| store V scale + zero (fp16 + fp16) | 4 | slot output |

### Warp-stall attribution (TBD Phase 2)

| stall class | % | dominant? | implication |
|---|---:|---|---|
| `long_scoreboard` (HBM latency) | TBD | TBD | If dominant: HYP-064 (midpoint pre-load) lifts ceiling. |
| `short_scoreboard` (smem bank conflicts) | TBD | TBD | If dominant: pad smem; revisit centroid_midpoints layout. |
| `math_pipe_throttle` | TBD | TBD | If dominant: surprising — bucketize is light math. |
| `wait` (`__syncthreads`) | TBD | TBD | Likely low — `num_warps=1`. |

**Predicted dominant stall**: `long_scoreboard` from the per-iteration
midpoints reload. Confirm with HYP-060.

---

## 2. `_tq_fused_store_fp8`

**File**: `vllm/v1/attention/ops/triton_turboquant_store.py:_tq_fused_store_fp8`.
**Role**: FP8 K + uniform V store. Cast K fp16→`float8e4nv`/`float8e4b15`,
write 1 byte/dim. Quantize V uniformly + write.

> **Out of upstream-track scope on A100.** HYP-057 confirmed the
> Ampere FP8 path (`float8e4b15`) silently corrupts on SM80. Profile
> for completeness but do not target as a primary optimization until
> upstream fixes the path or H100/H200 access lands.

### Launch geometry

| field | value | source |
|---|---|---|
| grid | `(num_tokens, num_kv_heads)` | `triton_turboquant_store.py` |
| `num_warps` | 1 (default) | constexpr |
| `num_stages` | 1 (default) | constexpr |
| `FP8_E4B15` | `1` if SM<8.9 else `0` | `_use_fp8_e4b15(dev)` |
| register budget | TBD (Phase 2) | — |
| occupancy | TBD (Phase 2) | — |

### SMEM map

```
TBD (Phase 2) — kernel is light; likely register-only for K cast.
```

### HBM traffic per invocation

| direction | bytes |
|---|---:|
| load raw K (fp16) | `D · 2` |
| store FP8 K | `D` |
| load raw V (fp16) | `D · 2` |
| store packed V + scale + zero | `ceil(D · VAL_BITS / 8) + 4` |

### Warp-stall attribution (TBD Phase 2)

Same template; not a primary Phase 3 target.

---

## 3. `_tq_decode_stage1`

**File**: `vllm/v1/attention/ops/triton_turboquant_decode.py:_tq_decode_stage1`.
**Role**: Decode-step attention per KV split. Unpack K MSE indices →
centroids → score; online softmax; unpack V indices → dequant →
weighted accumulate.

### Launch geometry

| field | value | source |
|---|---|---|
| grid | `(batch, Hq_per_kv_head, NUM_KV_SPLITS)` | `triton_turboquant_decode.py:586-587` |
| block | one program per (request, q_head, KV split) | implicit |
| `num_warps` | **1** (default) | `triton_turboquant_decode.py:586-587` — **HYP-062 target** |
| `num_stages` | **1** (default) | `triton_turboquant_decode.py:586-587` — **HYP-062 target** |
| `BLOCK_KV` | **4** (default) | `triton_turboquant_decode.py:552` — **HYP-062 target** |
| `BLOCK_DV` | `D` | constexpr |
| register budget | TBD (Phase 2) | — |
| occupancy | TBD (Phase 2) | — |

### SMEM map (per program)

```
┌─────────────────────────────────────────────────────────────────┐
│ q_rot[BLOCK_DV]            fp32   — pre-rotated Q (input arg)   │
│ centroids[2^MSE_BITS][D]   fp32   — gathered from HBM per tile  │
│   (HYP-063 target: pre-stage once at kernel entry)              │
│ scores[BLOCK_KV]           fp32   — QK output, softmax input    │
│ acc[BLOCK_DV]              fp32   — running attention output    │
│ m_i, l_i                   fp32   — softmax running max + sum   │
└─────────────────────────────────────────────────────────────────┘
total: TBD bytes (Phase 2)
```

### HBM traffic per (program × tile)

For each `BLOCK_KV` tile the program processes:

| direction | bytes (D=128, 4-bit) | source |
|---|---:|---|
| load packed K (`MSE_BYTES`) | `BLOCK_KV · 64` | slot[0:64] |
| load `‖k‖₂` (fp16) | `BLOCK_KV · 2` | slot[64:66] |
| **gather centroids[mse_idx]** | `BLOCK_KV · D · 4` (worst case fp32) | `triton_turboquant_decode.py:193-197` — **HYP-063 target** |
| load packed V | `BLOCK_KV · 64` | slot[KPS+0:KPS+64] |
| load V scale + zero | `BLOCK_KV · 4` | slot[KPS+64:KPS+68] |

Per program: `BLOCK_KV` tiles × per-tile traffic, summed over all tiles
in the assigned KV split. With `BLOCK_KV=4` and a long context, the
program iterates many tiles — every tile re-gathers centroids from
HBM. This is the dominant Phase 3 lever.

### Warp-stall attribution (TBD Phase 2)

| stall class | % | dominant? | implication |
|---|---:|---|---|
| `long_scoreboard` | TBD | TBD | Predicted dominant — centroid HBM gather + small `BLOCK_KV` keeps latency exposed. HYP-062 (larger `BLOCK_KV`, more `num_stages`) and HYP-063 (smem pre-stage) attack this. |
| `short_scoreboard` | TBD | TBD | If dominant: smem layout of `scores` / `acc` needs padding. |
| `math_pipe_throttle` | TBD | TBD | Unlikely on A100 with scalar fp32 path — would be the case post HYP-066/067 (`tl.dot`). |
| `wait` | TBD | TBD | `num_warps=1` → low. |

**Predicted dominant stall**: `long_scoreboard` (HBM gather +
under-staged tile pipeline). Confirm with HYP-059.

---

## 4. `_tq_full_dequant_kv`

**File**: `vllm/v1/attention/ops/triton_turboquant_decode.py:_tq_full_dequant_kv`.
**Role**: Bulk dequant of cached KV → fp16 buffer for the
continuation-prefill path (when `q_len > 128`). Output goes to
`flash_attn_varlen_func`.

### Launch geometry

| field | value | source |
|---|---|---|
| grid | `(max_seq, batch · num_kv_heads)` | `triton_turboquant_decode.py` |
| `num_warps` | 1 (default) | constexpr |
| `num_stages` | 1 (default) | constexpr |
| `BLOCK_D` | `D` | constexpr |
| register budget | TBD (Phase 2) | — |
| occupancy | TBD (Phase 2) | — |

### SMEM map

```
┌─────────────────────────────────────────────────────────────────┐
│ centroids[2^MSE_BITS]      fp32   — Lloyd-Max indices → fp32    │
│ k_dequant_buf[D]           fp16   — output K tile               │
│ v_dequant_buf[D]           fp16   — output V tile               │
└─────────────────────────────────────────────────────────────────┘
total: TBD bytes (Phase 2)
```

### HBM traffic per (program × token)

| direction | bytes (D=128, 4-bit) |
|---|---:|
| load packed K | 64 |
| load `‖k‖₂` | 2 |
| load centroids (per token) | `2^MSE_BITS · 4` |
| load packed V | 64 |
| load V scale + zero | 4 |
| store dequant K (fp16) | `D · 2 = 256` |
| store dequant V (fp16) | `D · 2 = 256` |

Per layer: `max_seq · batch · num_kv_heads` programs. The dequant
buffer (`layer._tq_k_dequant_buf` / `_tq_v_dequant_buf`) is reused
across layers but is the largest single allocation in this path
(~16 MB at typical shapes).

### Warp-stall attribution (TBD Phase 2)

Same template. Lower priority than `_tq_decode_stage1` because
continuation-prefill is rare for `small_balanced`-style workloads.

---

## Cross-kernel summary

Filled by Phase 2 from ncu output.

| kernel | dominant stall (predicted) | dominant stall (Phase 2 measured) | top Phase 3 lever | predicted ROI |
|---|---|---|---|---:|
| `_tq_fused_store_mse` | `long_scoreboard` (midpoint reload) | TBD | HYP-064 (midpoints pre-load) | 0.5–1 % TPOT |
| `_tq_fused_store_fp8` | TBD | TBD | (out of scope on A100) | — |
| `_tq_decode_stage1` | `long_scoreboard` (centroid gather + small `BLOCK_KV`) | TBD | HYP-062 (joint launch retune) + HYP-063 (smem pre-stage) | 10–19 % TPOT |
| `_tq_full_dequant_kv` | TBD | TBD | (deferred) | — |

---

## Reproducibility — Phase 1 capture

Single Forge job under `--security-profile profiling-debug`:

```bash
forge job submit --name tq-hyp058-baseline --gpu 1 \
    --disk-mount tq-models:/mnt/models \
    --shared-nfs --security-profile profiling-debug \
    --entrypoint-file /tmp/hyp058.sh
```

`/tmp/hyp058.sh`:

```bash
#!/bin/bash
set -euo pipefail
export HF_HOME=/mnt/models/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /workspace/shared/tq-vllm020/turboquant
pip install -e .   # installs the (currently no-op) plugin

OUT=/workspace/shared/hyp058_phase1
mkdir -p "$OUT"

# 1. nsys timeline — confirms which kernel dominates
nsys profile --stats=true -t cuda,nvtx \
    -o "$OUT/trace.nsys-rep" \
    python tests/bench_longbench_vllm.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend turboquant_4bit_nc --preset small_balanced \
        --out "$OUT/_warmup.json"

# 2. ncu — warp stalls + speed-of-light + memory analysis for all _tq_* kernels
ncu --section WarpStateStatistics \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --kernel-name regex:_tq_.* \
    --target-processes all \
    -o "$OUT/decode.ncu-rep" \
    python tests/bench_longbench_vllm.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend turboquant_4bit_nc --preset small_balanced \
        --out "$OUT/_warmup2.json"

# 3. accuracy baseline — every preset, SHA-256 of prediction strings
for BACKEND in auto turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc; do
    python tests/bench_longbench_vllm.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --backend "$BACKEND" --preset small_balanced \
        --out "$OUT/$BACKEND.json"
done

# 4. perf grid — 16 cells via bench_serve_upstream_entry.sh
bash tests/bench_serve_upstream_entry.sh "$OUT/serve_grid"
```

Phase 2 (HYP-059/060/061) reads `$OUT/decode.ncu-rep` and fills the TBD
cells above.
