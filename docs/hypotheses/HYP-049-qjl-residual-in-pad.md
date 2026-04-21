# HYP-049: QJL residual in the 80-B tile pad — unbiased inner product at long context

## Context

The production kernel (v5) currently stores only MSE-quantized nibbles +
per-chunk L2 norm. The 12 B of cp.async alignment padding in the 80 B
tile is zeroed and unused. Decoded attention score has the form

```
<Q, K̂_mse> = <Q, K_true> + ε_mse_t
```

where `ε_mse_t` is **biased** — it correlates with the rotated 16-level
Lloyd-Max grid the dequant step lands on. Over a long-context attention
sum

```
out = Σ_{t=1..T} softmax(Q·K̂_t / √d)_t · V̂_t
```

a per-token bias compounds linearly in T, and the softmax amplifies it
through `exp()`. The original TurboQuant paper (arXiv:2504.19874 §3)
patches this by adding a **1-bit QJL residual** on top of the MSE
codebook, giving an **unbiased** inner-product estimator with variance
`(π/2m)·‖residual‖²`. We dropped QJL from the kernel in HYP-031 for
perf reasons; we have not yet measured the long-context cost of doing so.

SPEC.md §2 binds us to an unbiased inner-product mode. We currently don't
satisfy it. HYP-049 is the engineering path to close that gap without
growing the cache footprint or regressing decode latency.

## Hypothesis

1. 4-bit Lloyd-Max alone has **biased** per-token inner-product error
   that compounds at long context, and the bias is measurable on
   long-context retrieval tasks (NIAH, LongBench-QA) even though it is
   invisible on WikiText-2 next-token PPL.
2. Storing `m = 32` QJL sign bits per 64-dim chunk inside the existing
   12 B cp.async pad, and accumulating the QJL correction inside the
   decode kernel, recovers unbiasedness with **zero HBM overhead** and
   **≤ 3 % decode TPOT regression** at seq ≥ 8k.
3. The tensor-core projection `sign(S · residual)` added to the write
   kernel costs ≤ 15 % of prefill TTFT.

## Prediction

### Long-context quality (primary)

Running Qwen3-8B on a 32 k prompt at the 4-way serving configs in
`docs/BENCHMARKS.md`, with `kv-cache-dtype = fp8_qjl`:

| metric               | mse-only (today) | mse + qjl (predicted) | fp16 baseline |
|----------------------|------------------|-----------------------|---------------|
| NIAH accuracy @ 4 k  | ≥ 0.95           | ≥ 0.98                | ≥ 0.98        |
| NIAH accuracy @ 16 k | **0.80–0.90**    | **≥ 0.95**            | ≥ 0.97        |
| NIAH accuracy @ 32 k | **0.60–0.80**    | **≥ 0.93**            | ≥ 0.95        |
| LongBench-QA F1      | fp16 − 1.0 pt    | fp16 − 0.3 pt         | baseline      |
| WikiText-2 PPL       | 14.91            | 14.91                 | 14.91         |

(If the "mse-only today" column comes in at fp16 parity even at 32 k, the
hypothesis is *rejected at its premise* — nothing to fix — and we stop.
Running this measurement before any kernel work is the step-0 gate.)

### Latency (secondary)

| phase                     | today    | with QJL (predicted) | ratio |
|---------------------------|----------|----------------------|-------|
| Tile HBM load             | 80 B     | 80 B                 | 1.00× |
| Decode TPOT @ seq 1024×8  | 29.5 ms  | ≤ 30.4 ms            | ≤ 1.03× |
| Decode TPOT @ seq 32768×8 | 138.5 ms | ≤ 142.7 ms           | ≤ 1.03× |
| Prefill TTFT @ 32768 × 4  | 3.36 s   | ≤ 3.86 s             | ≤ 1.15× |

### Correctness (blocking)

At seq ∈ {1 k, 4 k, 16 k, 32 k}, the Python-reference attention with
`mse + qjl` reconstruction must have score-vs-fp16 **cosine monotonically
higher than mse-only** at every length, and the gap should **grow with
seq_len** (QJL pays off more at long context).

## Method

The work is gated at three points. Each gate reads a cheap leading
indicator; if it fails, we stop before the next (more expensive) step.

### Gate 0 — does the problem actually exist on our stack?

**Cheapest test, Python-only, no kernel code.**

1. `tests/test_qjl_long_context_bias.py`: for seq ∈ {1 k, 4 k, 16 k, 32 k}:
   - Sample Q, K, V from Qwen3-8B-style activation distributions
   - Compute fp16 reference attention score vector s_fp16
   - Compute s_mse from mse-only reconstruction (existing `quantizer.py`
     without QJL)
   - Compute s_both from mse + QJL residual (existing Python reference)
   - Report: `cos(s_mse, s_fp16)`, `cos(s_both, s_fp16)`, and the
     per-token `|s_mse_t − s_fp16_t|` vs `|s_both_t − s_fp16_t|`
2. **Pass condition:** at seq ≥ 8 k, mse-only score cosine drops below
   0.995 and mse+qjl stays above 0.998.
3. **If it fails:** the kernel is already above the noise floor even at
   32 k. Close the hypothesis and update SPEC.md §2 to note empirical
   spec-parity despite theoretical non-unbiasedness.

Expected runtime: ~2 minutes on a single A100 notebook.

### Gate 1 — does a full Python prototype of the modified layout work end-to-end?

Only if Gate 0 passes.

1. `tests/test_qjl_tile_layout.py`: encode/decode round-trip at the
   proposed 80-B layout with `m=32` QJL signs per chunk, using the
   existing Python `QJL` in `turboquant/qjl.py`.
2. Verify: tile is exactly 80 B, bit layout parses back correctly,
   reconstruction cosine ≥ 0.998 at hd=128.
3. `tests/test_qjl_attention_pytorch.py`: replicate the decode kernel's
   tile loop in PyTorch using the Python QJL path. Compare vs fp16
   reference at seq ∈ {1 k, 4 k, 16 k, 32 k}, batch ∈ {1, 8}.
4. **Pass condition:** end-to-end attention cosine ≥ 0.998 at every
   config; no NaNs; reconstruction scales correctly with norm.

### Gate 2 — kernel prototype and long-context serving measurement

Only if Gate 1 passes.

1. Extend `quantize_write_kv_cache` → emit `(mse_nibbles, mse_norm,
   res_norm, qjl_signs_32bit_per_chunk)` into a 80-B tile. Projection
   `sign(S · residual)` uses cuBLAS-batched tensor-core fp16 matmul on
   `(T·H) × 64 @ 64 × 32`.
2. Modify `dequant_row_to_fp16_v5` → `dequant_and_qjl_correct` that
   reads the QJL bits from the tile (already in smem from cp.async),
   adds `norm_res · sqrt(π/2)/32 · Σ_i Q̃_i · (2·bit_i − 1)` into the
   per-tile QK score before softmax.
3. Add `TQ_FORCE_QJL=0|1` env var and new `kv_cache_dtype = "fp8_qjl"`.
   Existing `fp8` path unchanged.
4. Correctness: `tests/test_v5_qjl.py` — cosine vs un-quantized
   reference at the 6 serving configs from BENCHMARKS.md.
5. Latency: run the BENCHMARKS.md sweep with `--kv-cache-dtype fp8_qjl`
   and record TTFT / TPOT / throughput deltas.
6. Quality: NIAH + LongBench-QA at 4 k / 16 k / 32 k, ours vs ours-qjl
   vs fp16 vs upstream.

## Kill criteria

Any one of the following terminates the hypothesis with a "rejected"
status and no merge:

- Gate 0 shows mse-only already matches fp16 at 32 k → hypothesis
  is empirically vacuous.
- Gate 2 shows decode TPOT regression > 5 % at any seq × concurrency
  config in BENCHMARKS.md → "free compute during ldmatrix stall"
  assumption is wrong on SM80.
- Gate 2 shows prefill TTFT regression > 15 % → tensor-core projection
  isn't amortizing; revisit as a separate prefill-kernel optimisation.
- Gate 2 shows long-ctx NIAH/LongBench quality does **not** improve
  with QJL → QJL-at-m=32 isn't strong enough; the fallback is a new
  tile size (96 B with m=64) but that is a separate hypothesis, not
  this one.

## Scheme: Variant A — QJL-in-pad (chosen 2026-04-21)

Keeps today's 4-bit Lloyd-Max MSE **bit-for-bit identical**. Adds a
QJL residual stored in the existing 12 B cp.async pad:

```
byte  0           31 32           63 64  67 68  71 72      79
      ┌─────────────┬─────────────┬───────┬───────┬────────┐
      │ chunk-0 MSE │ chunk-1 MSE │ MSE   │ QJL   │ QJL    │
      │  32 B 4-bit │  32 B 4-bit │ norms │ res-  │ signs  │
      │  (today)    │  (today)    │ 2×fp16│ norms │ 2 × 32 │
      │             │             │  4 B  │ 4 B   │ = 8 B  │
      └─────────────┴─────────────┴───────┴───────┴────────┘
```

- MSE at 4 bits = unchanged from today → short-ctx cosine and PPL are
  guaranteed to match today's numbers bit-for-bit.
- `m = 32` QJL sign bits per 64-dim chunk → JL variance
  `(π/2·32)·‖residual‖² ≈ 0.049·‖residual‖²`.
- Effective bit rate: 4.5 bits/dim — pays a 0.5-bit-per-dim "tax" in
  information content, but 0 B extra HBM because the pad was already
  16-byte-aligned and loaded on every tile.

**Rejected alternative: Variant B (same-budget 3 + 1, paper's
Algorithm 2 / `TurboQuantProd`).** Would replace 4-bit MSE with
3-bit MSE + 1-bit QJL. Same bit budget, risk of short-ctx regression
because 3-bit Lloyd-Max is coarser. User chose conservative A.

## Gate 0 — updated for Variant A only

The leading indicator now compares:

1. **fp16 baseline** — attention scores from unquantized K/V.
2. **MSE-only today** — 4-bit Lloyd-Max reconstruction.
3. **Variant A** — 4-bit Lloyd-Max + rectangular QJL on residual with
   `m = 32` per 64-dim chunk.

This requires a rectangular QJL (`QJL(dim=64, m=32)`) the existing
`turboquant/qjl.py` doesn't support. Gate 0 ships a local helper in
the test file (not touching `qjl.py`) that rolls a `32×64` Gaussian
projection with `dequant_scale = sqrt(π/2)/32`. The `qjl.py` change
is deferred to Gate 1 where the production code will need it.

Decision tree after Gate 0:

| outcome                                            | next action                         |
|----------------------------------------------------|-------------------------------------|
| Variant A reduces bias ≥ 2× vs MSE-only at seq ≥ 8 k AND `out_cos(A) − out_cos(MSE) ≥ 0.001` | Gate 1 with Variant A |
| Variant A tied with MSE-only even at 32 k          | hypothesis rejected as vacuous      |
| Variant A regresses on softmax cosine anywhere     | hypothesis rejected — scheme broken |

## Status: pending

Gate 0 scaffolding is in worktree branch `worktree-agent-a8d9f08d`
(`tests/test_qjl_long_context_bias.py`). That commit measures Variant B
only (via `TurboQuantProd`). Needs:

1. Extend the test with Variant A (4-bit MSE + rectangular QJL at
   m=32) alongside MSE-only. Drop Variant B rows — decision is made.
2. Run on Forge A100 at seq ∈ {1 k, 4 k, 16 k, 32 k}.
3. Post-run: decision per the table above. If pass → Gate 1; else close
   "rejected" here.

## Paper references

- Zandieh et al., "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate", arXiv:2504.19874, 2025 (the base
  algorithm and the MSE + QJL decomposition).
- Zandieh et al., "QJL: 1-bit Quantized JL Transform for KV Cache
  Quantization with Zero Overhead", 2024 (the specific 1-bit JL
  variant this project adopts, already implemented in
  `turboquant/qjl.py`).
