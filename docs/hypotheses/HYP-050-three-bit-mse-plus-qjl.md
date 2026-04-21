# HYP-050: Same-budget 3-bit MSE + 1-bit QJL (the paper's Algorithm 2)

## Context

HYP-049 Variant A (4-bit MSE + QJL-in-pad) was **rejected at Gate 0** on
synthetic Gaussian data. The theoretical reason generalises:

```
QJL correction variance  = (π / 2m) · ‖residual‖² · ‖Q‖² · ‖K‖²
MSE inner-product bias  ≈ O(Δ_codebook²) · ‖Q‖² · ‖K‖²
```

QJL pays off only when `(π/2m) · ‖residual‖² < Δ²`. At 4-bit MSE the
residual is small (`‖res‖²/‖x‖² ≈ 0.02`) and Δ is already small, so QJL
adds more variance than bias it corrects. At 3-bit MSE the residual is
roughly **4× larger** (one extra bit of resolution lost → 2× larger Δ),
and QJL's fixed-`m` variance becomes relatively cheaper.

TurboQuant's paper formalises this as **Algorithm 2**: at total budget
`b`, spend `b−1` bits on MSE and 1 bit on QJL. `turboquant/quantizer.py`
lines 113–158 (`TurboQuantProd`) already implements this exact
decomposition — it's the design we dropped when the production kernel
went 4-bit pure in HYP-031.

## Hypothesis

At the same 4-bit total budget, **3-bit MSE + 1-bit QJL is
quality-equivalent to 4-bit pure MSE at short context, and strictly
better at long context** on real Qwen3-8B activations, because:

1. 3-bit MSE's residual is large enough that QJL correction is
   information-rich relative to its own JL noise.
2. QJL provides unbiasedness that 4-bit MSE has but coarser 3-bit MSE
   alone lacks.
3. The bit budget is preserved, so cache footprint (and all
   long-context throughput wins in BENCHMARKS.md) is preserved.

## Prediction

### Gate 0 — leading indicator on real Qwen3-8B K/V

Running the HYP-049 Gate 0 harness extended to **four methods** against
K/V tensors captured from a real Qwen3-8B forward pass on a 32 k
LongBench prompt:

| method                    | `out_cos` @ 1 k | `out_cos` @ 32 k | bias-compounding? |
|---------------------------|-----------------|------------------|--------------------|
| fp16 (reference)          | 1.000           | 1.000            | no                 |
| 4-bit MSE (today)         | ≥ 0.99          | ≥ 0.99           | possibly (test)    |
| Variant A (4-bit + pad)   | ≤ 4-bit MSE     | = 4-bit MSE      | no (but net worse) |
| **Variant B (3+1)**       | **≥ 4-bit MSE − 0.005** | **≥ 4-bit MSE + 0.005** | **no** |

Pass for HYP-050: Variant B matches 4-bit MSE to within 0.005 at short
context AND beats it by ≥ 0.005 on `out_cos` at seq ≥ 16 k.

### Storage impact

| scheme              | per head @ hd=128             | tile size (16 B-aligned) |
|---------------------|-------------------------------|--------------------------|
| today (4-bit MSE)   | 64 B quant + 4 B norms        | **80 B** (12 B pad)      |
| **Variant B (3+1)** | 48 B quant + 4 B norms + 16 B QJL signs + 4 B res-norms | **80 B** (8 B pad) |

Same tile size. Fewer MSE bytes, more QJL bytes. `get_kv_cache_shape`
output is identical; the production Dockerfile and vLLM patches don't
need rebuilds beyond the kernel itself.

### Latency (secondary, measured only if Gate 0 passes)

Same cost structure as HYP-049 Variant A:

- Decode: +64 fp16 FMAs per tile (QJL correction, 2× of Variant A
  because `m = 64` per chunk here). Still << WMMA cost. Predicted
  ≤ 3 % TPOT regression.
- Prefill: 3-bit MSE is **cheaper** than 4-bit (fewer codebook levels,
  simpler packing), but the QJL projection adds an `m·d` tensor-core
  matmul per token. Net prefill: predicted ≤ 10 % regression (smaller
  than Variant A because MSE step is cheaper).

## Method

### Gate 0 (this hypothesis) — real-data Python reference

1. **Real K/V capture** (one-shot, ~5 min): Qwen3-8B forward pass on a
   32 k LongBench-QA prompt, register hooks on `k_proj`/`v_proj` of
   layers {8, 16, 24} (spread across depth), save
   `(num_layers, batch, seq_len, num_kv_heads, head_dim)` fp16 tensor
   to `/workspace/shared/hyp050_kv_real.pt`.
2. Extend `tests/test_qjl_long_context_bias.py` to load the real K/V
   (fall back to synthetic if not present, for CI), and compare **four
   methods** per seq length:
   - fp16 baseline
   - 4-bit MSE only (today's production)
   - Variant A (from HYP-049 — for cross-reference)
   - **Variant B: `TurboQuantProd(bit_width=4)`** (already implemented
     in `quantizer.py`)
3. Measure at seq ∈ {1 k, 4 k, 16 k, 32 k}. Report score cosine,
   abs-err, bias, softmax cosine, output cosine (same columns as
   HYP-049).
4. Also folds into the **HYP-049 real-data hedge** — Variant A's row
   on real data decides whether HYP-049 closes `rejected` or reopens.

### Gate 1 (only if Gate 0 passes) — rectangular QJL in production code

1. Extend `turboquant/qjl.py` to accept `m` separate from `d`
   (~20 LOC: `S ∈ R^{m×d}`, rescale `dequant_scale = sqrt(π/2)/m`).
   This is a prerequisite both HYP-049 and HYP-050 would have needed.
2. Python end-to-end reference with the target tile layout
   (48 B quant + 4 B MSE-norms + 4 B res-norms + 16 B QJL signs + 8 B pad).
3. Round-trip encode/decode cosine ≥ 0.998 at hd=128.

### Gate 2 (only if Gate 1 passes) — kernel implementation

Changes vs HYP-049 Variant A's kernel plan:
- Codebook: `kCodebook3bit[8]` (already present in `page_turbo.cuh`,
  unused). Dequant LUT is narrower; `__shfl_sync` broadcasts are the
  same pattern.
- Quant packing: 3-bit GGML-packed (8 dims per 3 bytes) vs current
  4-bit nibble-packed. Reference packer exists in `pack.cpp`.
- QJL: 64 signs per chunk × 2 chunks = 16 B, sits after the MSE data
  and norms in the 80 B tile.

### Forge run plan

Single Forge job that produces the evidence for both HYP-049 (real-data
close-out) and HYP-050 (Gate 0):

1. Capture Qwen3-8B real K/V (5 min, 1 A100).
2. Run the 4-method × 4-seq sweep (1 min).
3. Report aggregate table to `/workspace/shared/hyp050_gate0.txt`.

## Kill criteria

- **At Gate 0:** Variant B does not beat 4-bit MSE by ≥ 0.005 on
  `out_cos` at seq ≥ 16 k on real data → hypothesis rejected.
- **At Gate 0:** Variant B regresses `out_cos` at seq ≤ 4 k by more
  than 0.005 → hypothesis rejected (paid short-ctx tax for no
  long-ctx gain).
- **At Gate 2:** decode TPOT regression > 5 % → revisit; kernel
  architecture may not support the narrower 3-bit codebook efficiently.

## Relationship to other hypotheses

- **HYP-049:** Alternate scheme (4+QJL-in-pad). Rejected on synthetic,
  real-data verdict lands in the same Forge job as HYP-050 Gate 0.
- **GOAL.md Success Criterion #3:** "5× compression with <1% PPL" is
  unreachable at pure 4-bit MSE. HYP-050 is the gateway because
  extending this framework to 2-bit MSE + 2-bit QJL (or similar) is
  how we get below 4 bits/dim without quality collapse.
- **SPEC.md §2 "Unbiased Attention Estimation":** HYP-050 is the
  direct path to satisfying this spec behavior in production.

## Status: pending

Dispatched to the same Forge job as HYP-049's real-data rerun.
