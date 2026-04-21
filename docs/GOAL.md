# Goal

Production-grade KV cache quantization for LLM inference, implementing the
TurboQuant algorithm with kernel-level FlashInfer integration to deliver
4.5× memory compression at quality parity with fp16 on long-context
workloads.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, 2025. Paper's
recommended operating points (Section 4.3):

- **3.5 bits/dim avg** (outlier-aware + QJL): 4.5× compression, LongBench
  and NIAH **matches full-precision** (50.06 vs 50.06, NIAH 0.997).
- **2.5 bits/dim avg** (outlier-aware + QJL): 4.5× compression,
  marginal degradation (LongBench −0.62 pt).

QJL's contribution is largest at bit-widths b ≤ 2 where MSE bias is
significant (paper Figure 2); at b ≥ 4 QJL is vestigial, which our
HYP-049 and HYP-050 experiments confirm empirically.

## Success Criteria

1. **GPU-native TurboQuant** — all quantization/dequantization runs on
   GPU with no CPU fallback.
2. **FlashInfer kernel fusion** — TurboQuant quantization is fused into
   FlashInfer-style attention kernels, not a Python-level wrapper.
3. **Compression at quality parity (target)** — outlier-aware mixed
   precision at **3.5 bits/dim avg** achieves **≥ 4.5× compression** vs
   fp16 with **LongBench parity** (≤ 0.3 pt drop) on Qwen3-8B, WikiText-2
   PPL drift < 0.5 %, and NIAH ≥ 0.995 across 4k–32k contexts.
4. **Aggressive-tier compression (stretch)** — same framework at
   **2.5 bits/dim avg** with ≤ 1.0 pt LongBench drop and NIAH ≥ 0.97 at
   32k. This is the regime where QJL is load-bearing and becomes the
   binding theoretical win.
5. **Current shipped baseline (measured)** — pure 4-bit MSE, 3.2×
   compression, fp16-parity on WikiText-2 PPL and 100 % exact token
   match (HYP-029, HYP-045). This stays supported as the
   zero-quality-loss path for users who want the safest setting.

## Out of Scope

- CPU device support — GPU only.
- Weight quantization (GPTQ, AWQ, etc.) — this project is KV cache only.
- Training or fine-tuning — inference only.
- Beam search with quantized cache.
- Non-PyTorch backends (JAX, TensorFlow).
- Pure sub-4-bit without outlier detection — the paper does not prove
  this works, and `TurboQuantProd(bit_width=4)` (3-bit MSE + 1-bit QJL)
  was rejected by HYP-050 as strictly worse than pure 4-bit MSE at the
  same budget. Anything below 4 bits must use outlier-aware mixed
  precision (SPEC §3) to stay within the paper's proven envelope.
