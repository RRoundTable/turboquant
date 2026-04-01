# HYP-003: In-kernel FWHT will eliminate 203μs Python overhead

## Hypothesis
Moving the Fast Walsh-Hadamard Transform from Python to CUDA (warp shuffle-based) will eliminate the 203μs Python FWHT overhead that dominates end-to-end decode latency.

## Prediction
203μs savings. Standalone FWHT kernel verified correct (cosine=1.0).

## Method
Implemented warp_fwht_64 using warp shuffles. Standalone test passes. Integrated into decode kernel with signs parameter.

## Results
Standalone FWHT: **correct** (cosine=1.0, involution verified).
Inside decode kernel: **cos=-0.004** (completely wrong output).
Root cause: suspected smem context or thread scheduling issue. `block.sync()` inside `if (tz==0)` caused deadlock. Moving sync outside conditional fixed deadlock but output remained wrong.

Reverted.

## Status: rejected
FWHT works standalone but fails in multi-warp kernel context. The interaction between FWHT's shared memory usage and the decode kernel's smem layout is unresolved. Needs fundamentally different approach (e.g., separate FWHT kernel, or register-only FWHT).
