# HYP-021: CUDA write kernel to fix prefill overhead

## Hypothesis
Prefill KV write is 15-1500× slower than FP16 memcpy due to Python quantize loop.
A CUDA kernel fusing normalize + bucketize + pack will bring it to near-memcpy speed.

## Prediction
- 2048 tokens: 5.2ms (vectorized Python) → <1ms (CUDA) → 346μs target (memcpy parity)
- TTFT overhead: 14-20% → <3%

## Method
Fuse into one CUDA kernel per layer: normalize → codebook quantize → nibble pack → store.
Each thread handles one (token, head, chunk) independently — embarrassingly parallel.

## Status: pending
