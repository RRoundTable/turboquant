# HYP-065: Adaptive `NUM_KV_SPLITS` per (batch, max_kv_len) bucket

Second Phase 3 code HYP, **promoted** to first-up after HYP-062's
clean rejection (which showed within-CTA launch retunes can't move the
needle on this kernel — Amdahl bound at 4.7 % of total time).

Different lever: changes the launch **GRID**, not block params.

## Hypothesis

Upstream's `max_num_kv_splits=32` is a constant chosen for cudagraph
compatibility (the value baked into the captured graph per shape
bucket). At low batch + short seq it over-splits (each split has tiny
work, stage2 merge dominates). At high batch + short seq it
over-decomposes (`batch × Hq × 32` CTAs vastly over-subscribes the
SM array, queueing overhead grows).

Replacing the constant with a **picker that reads `(batch, Hq, max_kv_len, num_sms)`**
at metadata-build time should cut both failure modes. Specifically at
HYP-058's biggest absolute gap cells (`{4bit_nc, 3bit_nc} × s8192 × c8`)
where TQ is +20–35 ms over fp16, the binding constraint is plausibly
SM scheduling (8192 CTA queue depth on 108 SMs), not per-CTA kernel
internals.

### Why this dodges HYP-062's Amdahl ceiling

HYP-062 shrunk `_tq_decode_stage1`'s per-CTA time → bounded by 4.7 %
TQ kernel fraction × s. HYP-065 changes (a) **CTA count**, which
affects launch + stage2 cost outside the kernel, and (b) **per-CTA
work distribution**, which affects when adjacent kernels can begin.
Neither is Amdahl-bounded by `_tq_decode_stage1`'s share alone.

## Prediction

| cell | HYP-058 baseline | predicted with adaptive |
|---|---:|---:|
| `4bit_nc × s8192 × c8` | 68.8 ms | 65.0–67.0 ms (−3 to −6 %) |
| `3bit_nc × s8192 × c8` | 83.4 ms | 78.5–81.0 ms (−3 to −6 %) |
| `4bit_nc × s8192 × c1` | 17.5 ms | ≈ 17.5 ms (no change expected; high-batch effect) |
| `3bit_nc × s8192 × c1` | 18.6 ms | ≈ 18.6 ms (same) |

Best-factor pick is one of `{1, 2, 4, 8}` — sweep finds the A100 winner
empirically. H100/H200/B200 will be re-swept when hardware lands and
their factor entries updated in `turboquant/dispatch.py`.

## Method

### Picker formula (in `turboquant/dispatch.py`)

```python
factor = _SM_TARGETS.get(torch.cuda.get_device_capability(), 4)
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
target_ctas = factor * num_sms                          # A100 108×4=432
naive_splits = ceil(target_ctas / (batch * hq))          # 432/(8×32)=2
splits = clamp(naive_splits, lo=1, hi=upstream_max_splits)
# Floor on per-split chunk size to avoid sub-warp work
min_chunk = 32
splits = min(splits, max(1, max_kv_len // min_chunk))
```

`_SM_TARGETS` table:
- `(8, 0)` A100 — sweep determines value (this HYP).
- `(9, 0)` H100/H200 — sweep TBD when access lands.
- `(10, 0)` B200 — sweep TBD.

### Plugin patch surface

Two changes:

1. `turboquant/dispatch.py` — new module with `pick_adaptive_splits()`.
2. `turboquant/kernels/decode_stage1.py:make_patched_launcher()` —
   when `TQ_ADAPTIVE_SPLITS=1` env is set, call
   `pick_adaptive_splits(B, Hq, max(seq_lens.tolist()))` and override
   the launcher's `max_num_kv_splits` argument before launch.

The metadata builder approach (patch `TurboQuantMetadataBuilder.build`)
is rejected as more invasive — the launcher already has access to
`(B, Hq, seq_lens)`, so we can compute splits there without bothering
upstream's metadata struct.

### Sweep design

Phase A (perf):
- factor ∈ {0=no-patch (upstream 32), 1, 2, 4, 8}
- preset ∈ {`turboquant_4bit_nc`, `turboquant_3bit_nc`}
- seq=8192, conc=8 only (the cells where the hypothesis predicts win)
- 10 cells × ~10 min ≈ 1.7 h

Phase B (winner sanity at low concurrency):
- best_factor × {`4bit_nc`, `3bit_nc`} × seq=8192 × conc=1
- 2 cells × ~5 min ≈ 10 min

Phase C (parity check):
- best_factor × `turboquant_4bit_nc` × `small_balanced`
- mean_score within ±0.002 pp per task gate (opt-out per `docs/GOAL.md`).
- 1 run × ~1 min

Total ≈ 2 h.

### Multi-arch hooks (no run yet, design only)

`dispatch.py` reads compute capability at startup and indexes
`_SM_TARGETS`. When H100 access lands, the sweep entrypoint reruns
on H100 unchanged; only `_SM_TARGETS[(9, 0)]` gets updated with the
H100-tuned factor. No code branching needed at the call site.

H100/H200/B200 also have other levers (TMA, wgmma, FP4) that are
HYP-066/067 territory — orthogonal to HYP-065. The picker is
arch-portable.

## Pass / fail

- **Primary pass**: ≥ 3 % TPOT improvement at one of the c=8 cells.
- **Stretch pass**: ≥ 6 % at the same cell.
- **Soft pass**: any c=8 cell improves ≥ 1.5 % AND no c=1 cell
  regresses > 0.5 %. Picker promoted as default-on for c≥4.
- **Kill**: best factor's TPOT is within ±0.5 % of upstream-32 across
  all c=8 cells. Adaptive picker doesn't pay back on A100.

## Pass / fail (parity)

`mean_score within ±0.002 pp per task` for the winner config.
SHA-256 byte-exact is **not required** (split count change → different
fp32 reduction order → fp16 cast result may differ in ULPs).
Mathematical justification (per `docs/GOAL.md` §1): stage2 is an
LSE-style fp32 reduction; varying `NUM_KV_SPLITS` only changes the
order in which partials are combined. Both orderings compute the same
mathematical sum.

## Status: REJECTED — adaptive picker direction was wrong

The hypothesis predicted lowering splits at high concurrency would
help. Sweep proves the opposite: **every adaptive factor is strictly
slower than upstream's `splits=32`**, by 19–125 % at c=8 cells. f8
(the largest factor in the sweep, `splits=⌈108·8/(8·32)⌉=4`) is the
least bad and still −19 to −26 %. f1 (splits=1) is catastrophic at
−100 to −125 %.

### Forge run

- Job: `ab6c0a9b` (`tq-hyp065-sweep`, SUCCEEDED 2026-04-24 02:39 UTC).
- Image: `ce745b54`, default profile.
- Dispatch import-check at startup printed
  `pick_adaptive_splits(B=8, Hq=32, kv=8192) = 2` — picker active and
  computing as designed.
- `--enforce-eager` enabled in serve script (`.item()` on `seq_lens`
  needs eager mode; otherwise cudagraph capture would error). All
  cells share this so the f0 vs f≥1 comparison is internally consistent.

### Phase A — c=8 perf sweep

Median TPOT at `s8192 × c8`:

| factor | preset | TPOT (ms) | Δ vs f0 (upstream 32) | tput (tok/s) |
|---:|---|---:|---:|---:|
| **f0 (upstream)** | `4bit_nc` | **69.62** | — | 84.3 |
| **f0 (upstream)** | `3bit_nc` | **82.90** | — | 79.7 |
| f1 | `4bit_nc` | 156.77 | **−125.20 %** | 44.6 |
| f1 | `3bit_nc` | 166.13 | −100.40 % | 42.4 |
| f2 | `4bit_nc` | 150.31 | −115.91 % | 47.3 |
| f2 | `3bit_nc` | 159.81 | −92.77 % | 44.8 |
| f4 | `4bit_nc` | 107.55 | −54.49 % | 64.0 |
| f4 | `3bit_nc` | 120.47 | −45.32 % | 57.9 |
| f8 | `4bit_nc` | 87.38 | −25.52 % | 76.1 |
| f8 | `3bit_nc` | 98.41 | −18.72 % | 68.9 |

Picker outputs at `(B=8, Hq=32, kv=8192)`:

| factor | target_ctas | naive_splits | clamped |
|---:|---:|---:|---:|
| f1 | 108 | ceil(108/256) | **1** |
| f2 | 216 | ceil(216/256) | **1** |
| f4 | 432 | ceil(432/256) | **2** |
| f8 | 864 | ceil(864/256) | **4** |

Trend: more splits (closer to upstream's 32) → less bad. The picker
direction is *backwards*; we'd want the picker to **increase** splits
beyond 32 at low batch+seq combinations, which is structurally
impossible without changing upstream's `mid_o` buffer-shape contract
(which is fixed for cudagraph-bucket compatibility).

### Phase B — c=1 sanity at f=8

Picker at `(B=1, Hq=32, kv=8192) = ⌈864/32⌉ = 27` (close to upstream's 32).

| cell | TPOT (ms) | HYP-058 baseline (cudagraph mode) |
|---|---:|---:|
| `f8 × 4bit_nc × c1` | 24.67 | 17.5 |
| `f8 × 3bit_nc × c1` | 24.66 | 18.6 |

The 41 % / 33 % gap at c=1 is **not adaptive's fault** — it's eager
vs cudagraph-mode overhead (HYP-058 ran without `--enforce-eager`,
this sweep added it). That's a confound; the c=1 numbers don't isolate
adaptive's effect cleanly. What we *can* read: f=8 at c=1 produces
splits=27, very close to upstream's 32, so the regression here is
mostly the eager-mode tax, not adaptive's tweak.

### Phase C — accuracy parity at f=8

`max|Δscore|=0.0005` per task. **Pass** (gate ≤ 0.002). Adaptive splits
preserves the math; our concern about reduction-order divergence was
real but bounded — fp16 cast cancels the small fp32-LSE differences.

### Why the hypothesis missed

The mental model was wrong:
- I assumed `NUM_KV_SPLITS=32` exists for SM saturation. With `108`
  SMs, `batch=8 × Hq=32 × splits=32 = 8192` CTAs vs SMs is heavily
  oversubscribed → "we don't need that many".
- Reality: splits expose **per-request parallelism**, not just
  cluster-level oversubscription. Each split processes
  `seq/splits` tokens serially in the per-tile loop. At seq=8192:
  - splits=32 → 256 tokens/CTA → 64 BLOCK_KV iters (`BLOCK_KV=4`)
  - splits=4 → 2048 tokens/CTA → 512 iters (8× longer per CTA)
- Even with SMs over-subscribed, that 8× longer per-CTA serial loop
  dominates wall-time. The launch overhead of more CTAs is dwarfed
  by per-CTA work.

In other words: `NUM_KV_SPLITS=32` is upstream's *as-aggressive-as-the-
buffer-allows* setting, not a conservative one. Going lower means
serializing per-request work more.

### Implications for Phase 3 strategy (after HYP-062 + HYP-065 both REJECTED)

Two clean negative results in a row from kernel-level retunes on this
workload. Pattern:

1. **HYP-062** within-CTA tuning — couldn't beat upstream's `(1,1,4)`.
2. **HYP-065** between-CTA grid tuning — couldn't beat upstream's
   `splits=32`. The cap at 32 is structural.

Both confirm the **Amdahl ceiling from HYP-059**: TQ kernels are
4.7 % of total kernel time at this workload. Kernel-only
optimizations on A100 are bounded to roughly that fraction.

What's left in Phase 3:
- **HYP-063 (centroid SMEM pre-stage)** — different mechanism (HBM
  load reduction). Worth ~1-2 % but won't move the headline.
- **HYP-064 (midpoints pre-load)** — already demoted. ≤0.3 %.

Realistic next steps:
1. **Skip the rest of Phase 3.** Two clean rejections + Amdahl bound
   say kernel-only A100 optimization is mostly out of runway.
2. **Move to Phase 4 setup** — H100/H200/B200 access becomes the
   real lever (TMA, wgmma, FP4 — fundamentally different mechanisms).
   `dispatch.py` already has `_SM_TARGETS` table primed for that.
3. **OR pivot to non-kernel work** — vLLM scheduler / batching, KV
   cache layout, or Phase 5 (upstream PR contributions) on what we've
   confirmed (HYP-058 baseline reproduction is upstream-PR-able as a
   regression test addition).

### Artefacts

- `/workspace/shared/hyp065_sweep/` (Forge NFS): 12 perf JSONs +
  `winner_factor.txt` + `acc_winner.json` + `parity_winner.txt`.
- Local mirror: `results/hyp065/perf_grid/{cell}/upstream-s8192-c*.json`
  + `sweep_summary.csv` + `winner_factor.txt` + `parity_winner.txt`.
- Plugin code (`turboquant/dispatch.py` + `turboquant/kernels/decode_stage1.py`
  + `turboquant/vllm_plugin.py`) **kept in repo** — defaults to
  `factor=4` from `_SM_TARGETS[(8,0)]` but only activates when
  `TQ_PATCH_DECODE=1 + TQ_ADAPTIVE_SPLITS=1` are both set. Default-off
  is no-op; infrastructure stays in place for HYP-066/067 reuse and
  for re-sweeping when H100/H200/B200 access lands.
- Forge job ID: `ab6c0a9b`.

### Follow-ups

- **[strategy]** Update ROADMAP with Phase 3 effective close-out.
  Promote Phase 4 to "Now-Blocked-on-Hardware".
- **[multi-arch hook]** When H100/H200/B200 access lands, the
  same sweep harness reruns unchanged — only `_SM_TARGETS` entries
  get filled. Prediction: factor will likely still be ineffective,
  because the underlying issue (per-request parallelism vs
  buffer-shape cap) is arch-independent.
- **[possible future]** A more invasive HYP could remove upstream's
  cudagraph buffer-shape constraint (allow per-bucket `mid_o` shape)
  and let splits go > 32 at low concurrency. That's an upstream PR
  question, not a plugin patch.
