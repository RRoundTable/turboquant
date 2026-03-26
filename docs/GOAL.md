# Goal

Production-grade KV cache quantization for LLM inference, implementing the TurboQuant algorithm with kernel-level FlashInfer integration for 4-6x memory compression with minimal quality loss.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv:2504.19874, 2025.

## Success Criteria

1. **GPU-native TurboQuant** — All quantization/dequantization runs on GPU with no CPU fallback
2. **FlashInfer kernel fusion** — TurboQuant quantization is fused into FlashInfer's attention kernels, not a Python-level wrapper
3. **Compression with quality** — 3-bit quantization achieves >= 5x compression vs fp16 with < 1% perplexity degradation on standard benchmarks (WikiText-2, MMLU)

## Out of Scope

- CPU device support — GPU only
- Weight quantization (GPTQ, AWQ, etc.) — this project is KV cache only
- Training or fine-tuning — inference only
- Beam search with quantized cache
- Non-PyTorch backends (JAX, TensorFlow)
