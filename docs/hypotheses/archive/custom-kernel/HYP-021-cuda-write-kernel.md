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

## Results (A100, Qwen3-1.7B)

Per-layer write (1024 tokens, 8 KV heads, hd=128):
  FP16 memcpy: 21μs
  Python quantize: 424μs (20× slower)
  CUDA kernel: 41μs (2.0× vs memcpy, 10.3× faster than Python)

Full model (28 layers, K+V, 2048 tokens):
  FP16: 1.2ms, Python: 23.8ms, CUDA: 3.1ms
  TTFT overhead: 3.7% (was 44%)

Correctness: bit-exact on quant bytes, norms within 1 ULP.

## Status: confirmed
CUDA write kernel reduces prefill overhead from 44% to 3.7%.
