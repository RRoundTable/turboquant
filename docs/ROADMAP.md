# Roadmap

## Now

### 1. Eager-mode vLLM integration (without FlashInfer)

Implement TurboQuant core algorithm on GPU and integrate with vLLM in eager mode (no kernel fusion). This validates correctness and measures quality impact before optimizing performance.

- [ ] Implement GPU-native TurboQuant core (codebook, Hadamard, QJL — all on CUDA)
- [ ] Research vLLM's KV cache architecture and integration points
- [ ] Integrate as a vLLM KV cache quantization backend (eager mode, not monkey-patch)
- [ ] Validate: end-to-end text generation with quantized KV cache
- [ ] Measure: perplexity impact, memory savings, throughput overhead

### 2. FlashInfer kernel fusion

Fuse TurboQuant quantization/dequantization into FlashInfer's attention kernels to eliminate overhead from eager mode.

- [ ] Fuse dequantization into FlashInfer's prefill and decode attention kernels
- [ ] Fuse quantization into KV cache write path
- [ ] Support FlashInfer's paged KV cache with quantized entries
- [ ] Throughput benchmark: fused vs eager vs unquantized

## Next

- [ ] Benchmark suite — perplexity (WikiText-2, MMLU), throughput, memory across model sizes (7B, 13B, 70B)
- [ ] Outlier calibration pipeline — automatic outlier channel detection from calibration data

## Later

- [ ] Upstream contribution to vLLM
- [ ] Speculative decoding compatibility
- [ ] Multi-node tensor parallelism
