# HYP-004: Reusing FlashInfer's compute functions will close the SDPA gap

## Hypothesis
The standalone kernel is slow because it reimplements QK/softmax/V without tensor cores or pipelining. Using FlashInfer's optimized compute_qk, update_local_state, sync_state will bring performance close to SDPA.

## Prediction
~25-30μs (1.2-1.5× vs SDPA) by leveraging FlashInfer's tensor core compute path.

## Method
v2 kernel: inject dequant-load into FlashInfer's decode pipeline. Use FlashInfer's anonymous-namespace functions directly. num_stages=1 (synchronous dequant).

## Results
Correctness: **cos=1.0** (9/9 configs on A100).
Performance (seq=1024): **351μs** (11.4× vs SDPA's 31μs).

The compute functions themselves are fast, but the synchronous dequant-load serializes the pipeline. FlashInfer's original kernel uses cp_async to overlap load with compute — our dequant requires ALU work during load, breaking the overlap.

## Analysis
FlashInfer's compute functions ARE being used correctly, but the performance is dominated by the serial load path, not compute. The 11× gap is from load serialization, not from compute quality.

## Status: rejected
Correct but slow. FlashInfer's compute path alone doesn't close the gap — the load/compute overlap (cp_async pipelining) is what makes FlashInfer fast, and our ALU-based dequant can't use it.
