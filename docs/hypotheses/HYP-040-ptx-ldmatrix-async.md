# HYP-040: Raw PTX ldmatrix.async proof-of-concept

## Hypothesis

After HYP-038 (pipelined `nvcuda::wmma::load_matrix_sync`) was rejected
because the C++ API's loads are synchronous, the remaining question was:
does dropping to **raw PTX `ldmatrix`** unlock async semantics that would
actually overlap smem→register loads with mma.sync compute?

If yes → the "kernel rewrite D" (~2-3 weeks) is justified.
If no → A100 architecture ceiling is reached; stop optimizing v5 on A100.

## Prediction

Three probe variants, all running 1000 iters of 8 WMMA-tile load+mma:
- **A**: `nvcuda::wmma::load_matrix_sync` + `mma_sync` (baseline).
- **B**: raw PTX `ldmatrix.sync.aligned.m8n8.x4` + `mma.sync.m16n8k16.f32.f16.f16.f32`.
- **C**: as B but with explicit prefetch — load tile kt+1's a/b regs BEFORE
  mma on tile kt fires.

Prediction:
- B within 5% of A (raw PTX lowers to similar SASS).
- **If C < A by ≥15%**: prefetch works → full PTX rewrite justified.
- **If C ≈ A**: `ldmatrix` is synchronous even at PTX level → A100 ceiling.

## Results (Forge A100, NGC pytorch:24.01, SM80, 2026-04-17)

```
A (nvcuda::wmma)          1043 cycles   740 ns
B (raw PTX ldmatrix.sync) 1032 cycles   732 ns   (1.01× A — noise)
C (PTX + prefetch)        1028 cycles   729 ns   (1.00× A — noise)
```

All within 1.5% of each other. **No speedup from raw PTX, no speedup from
explicit prefetch.**

Register count identical (32 regs, 0 spill) across all three variants. The
SASS they generate is effectively identical.

## Analysis

`ldmatrix` on Ampere has **no asynchronous variant**. Variants exist:
- `ldmatrix.sync.aligned.m8n8.x{1,2,4}.shared.b16` — all synchronous
- `ldmatrix.sync.aligned.m8n8.x{1,2,4}.trans.shared.b16` — also synchronous

The actually-async instructions are Hopper-only (SM90+):
- `cp.async.bulk.tensor.{2,3,4,5}d.shared::cluster.global` — tile-level async
- Warpgroup primitives via TMA (Tensor Memory Accelerator)

On SM80, every `ldmatrix` variant blocks the warp until the data is in
registers. Issuing it earlier in the instruction stream has no effect — the
warp scheduler serializes it against the following `mma.sync` that uses
the loaded registers, because the dependency is real at hardware level.

What `cp.async.ca` (which we DO use on A100) provides is **global→smem**
async. Our kernel already uses it to load quant bytes in Phase 1. But
**smem→register loads (ldmatrix) have no async equivalent on SM80**.

## Prediction verdicts

| Prediction | Target | Result | Verdict |
|-----------|--------|--------|---------|
| B within 5% of A | ≤ 5% | 1.01× (−1%) | ✓ confirmed |
| C < A by ≥15% (prefetch works) | ≥ 15% speedup | 0% | ✗ **rejected** |

## Decision

**Stop optimizing the v5 WMMA_QK phase on A100.** The 1043-cycle cost is
the architectural floor — we've now measured it three ways:
1. `nvcuda::wmma` (current production)
2. Raw PTX ldmatrix.sync
3. Raw PTX + software prefetch

All identical. There's no cheaper instruction on SM80. Full HYP-038 D-path
kernel rewrite is rejected without needing the weeks of work.

## What WOULD close the gap

1. **Hopper (H100, SM90)** has async ldmatrix variants + TMA. On H100:
   - `cp.async.bulk.tensor` for tile loads
   - Warpgroup-level mma with producer/consumer warps
   Full FA3-style kernel redesign would map naturally to H100 hardware.
2. **Architectural kernel restructure on A100**: HYP-020 (warp specialization)
   attempted producer/consumer warps on A100 and was rejected — gain only
   appears when compute:load ratio is right, which isn't our case.
3. **Different tiling**: 2D splits over multiple kv_heads per block to share
   load amortization. Speculative; non-trivial to prototype.

## Conclusion: v5 on A100 is at its optimization ceiling

The journey: HYP-033 (1323μs) → 034 (226) → 035 (110) → 032 (64) at seq=4096.
Remaining 1.47× gap vs FlashInfer at seq=4096 and 2.56× gap at seq=32768
are A100 architectural limits on the current block structure, not
optimization opportunities we've left open.

**Ship current state.** Further v5 perf work requires either (a) moving to
H100, or (b) a fundamentally different block structure — both are
roadmap-level decisions rather than single-file HYPs.

## References

- HYP-038 REJECTED (pipelined wmma loads) — led to this deeper probe
- HYP-020 REJECTED (warp specialization producer/consumer) — the other
  structural fix for load→mma stall, also rejected on A100
- FlashAttention-3 paper — SM90 async tensor core pipeline that works
  because Hopper has the hardware primitives A100 lacks
- Ampere PTX ISA reference — confirms no async ldmatrix on SM80
- Probe code: `/tmp/ptx_probe.py` (ephemeral, but reproducible)
