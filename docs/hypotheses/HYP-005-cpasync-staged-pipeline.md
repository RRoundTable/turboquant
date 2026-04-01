# HYP-005: cp_async staged pipeline will overlap VRAM load with compute

## Hypothesis
Splitting the load into two phases — (1) cp_async packed bytes to smem staging (HW DMA, overlaps with compute), then (2) short ALU dequant from staging to fp16 — will hide the dominant VRAM latency behind compute work.

## Prediction
3-5× speedup over v2. VRAM load (~200 cycles) hidden behind compute, leaving only ~25-cycle dequant serial.

## Method
v3 kernel: cp_async packed bytes → smem staging, wait, dequant → fp16 smem, then FlashInfer compute. Pipeline: overlap cp_async of next tile with current compute.

## Results
Correctness: **cos=1.0** (7/7 configs).
Performance: **18% SLOWER than v2** across all seq lengths.

| seq | v2 (μs) | v3 (μs) | v3/v2 |
|-----|---------|---------|-------|
| 512 | 183 | 215 | 1.18× slower |
| 1024 | 313 | 369 | 1.18× slower |

## Analysis
The compute phases (compute_qk ~0.5μs, V_accum ~0.3μs) are far too short to hide the VRAM load (~1.5μs per K or V tile). The kernel has ~4:1 memory:compute ratio. Pipelining only helps when compute ≈ load. The 6 syncs per iteration (vs 4 in v2) + staging indirection added more overhead than the overlap saved.

**Key insight: cp_async pipelining helps FlashInfer because FlashInfer loads 4× more data per tile (fp16 vs 4-bit). With 4-bit data, loads are fast but compute (dequant ALU) is the bottleneck — the kernel is compute-bound, not memory-bound.**

## Status: rejected
cp_async overhead > overlap benefit when compute << load time.
