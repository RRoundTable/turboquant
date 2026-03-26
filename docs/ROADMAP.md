# Roadmap

## Now

### 1. GPU-native TurboQuant

Port TurboQuant core to GPU-only execution. Currently the algorithm works on GPU tensors but several components (codebook lookups, Hadamard sign generation, QJL projection matrix) are initialized on CPU and lazily moved. Make everything GPU-native:

- [ ] Codebook centroids and boundaries initialized directly on GPU
- [ ] Hadamard random signs generated on GPU
- [ ] QJL projection matrix generated on GPU
- [ ] Remove all CPU-to-GPU transfers in hot paths (quantize/dequantize)
- [ ] Validate correctness with existing test suite on CUDA
- [ ] Benchmark: quantize/dequantize latency on GPU vs current implementation

### 2. FlashInfer integration with kernel fusion

Integrate TurboQuant into FlashInfer so quantization/dequantization happens inside the attention kernel, not as a separate Python step:

- [ ] Study FlashInfer's custom KV cache layout API and kernel extension points
- [ ] Register TurboQuant as a custom KV cache data type/layout in FlashInfer
- [ ] Fuse dequantization into FlashInfer's prefill and decode attention kernels
- [ ] Fuse quantization into KV cache write path
- [ ] Support FlashInfer's paged KV cache (page table management with quantized entries)
- [ ] Integration test: end-to-end attention with quantized KV through FlashInfer
- [ ] Throughput benchmark vs unquantized FlashInfer attention

## Next

- [ ] vLLM production integration — wire FlashInfer+TurboQuant as a vLLM KV cache backend
- [ ] Benchmark suite — perplexity, throughput, memory across model sizes
- [ ] Outlier calibration pipeline — automatic outlier channel detection from calibration data

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node tensor parallelism
