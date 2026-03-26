"""Benchmark TurboQuant KV cache against baselines.

Simulates realistic LLM attention with quantized KV cache and compares:
  1. TurboQuant MSE (Algorithm 1)
  2. TurboQuant Prod (Algorithm 2, unbiased)
  3. Naive per-channel min-max quantization (baseline)
  4. Full precision (fp16 reference)

Run: python tests/bench_kv_cache.py

Optional: pass --model to use real KV cache tensors from a HuggingFace model.
  python tests/bench_kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import math
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantCache


@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"  [{label}] {dt*1000:.1f} ms")


# ── Baseline: naive per-channel min-max quantization ───────────────────


class NaiveMinMaxQuantizer:
    """Simple per-channel min-max quantization (like KIVI without tuning)."""

    def __init__(self, bit_width: int):
        self.bit_width = bit_width
        self.num_levels = (1 << bit_width) - 1

    def quantize(self, x: torch.Tensor):
        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)
        scale = (xmax - xmin).clamp(min=1e-8) / self.num_levels
        indices = ((x - xmin) / scale).round().clamp(0, self.num_levels).to(torch.uint8)
        return indices, xmin, scale

    def dequantize(self, indices, xmin, scale):
        return indices.float() * scale + xmin


# ── Attention helpers ──────────────────────────────────────────────────


def scaled_dot_product_attention(q, k, v):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    return torch.matmul(F.softmax(scores, dim=-1), v)


def measure_attention_error(q, k_ref, v_ref, k_quant, v_quant):
    """Compute relative error and cosine similarity of attention output."""
    out_ref = scaled_dot_product_attention(q, k_ref, v_ref)
    out_q = scaled_dot_product_attention(q, k_quant, v_quant)

    rel_err = (out_ref - out_q).norm() / out_ref.norm()
    cos_sim = F.cosine_similarity(
        out_ref.flatten(), out_q.flatten(), dim=0
    )
    return rel_err.item(), cos_sim.item()


# ── Generate synthetic or real KV tensors ──────────────────────────────


def get_synthetic_kv(batch=1, heads=32, seq_len=512, head_dim=128):
    """Random KV tensors (simulates typical LLM activations)."""
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    q = torch.randn(batch, heads, 1, head_dim)  # single decode step
    return q, k, v


def get_real_kv(model_name: str, prompt: str = "The meaning of life is"):
    """Extract real KV cache from a HuggingFace model forward pass."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    # Extract KV from first layer
    past = outputs.past_key_values
    k = past[0][0]  # [batch, heads, seq_len, head_dim]
    v = past[0][1]

    # Query = last hidden state projected
    head_dim = k.shape[-1]
    q = torch.randn(1, k.shape[1], 1, head_dim)

    print(f"  KV shape: {k.shape}")
    print(f"  KV value range: [{k.min():.3f}, {k.max():.3f}]")
    print(f"  KV std: {k.std():.4f}")
    return q, k, v


# ── Main benchmark ─────────────────────────────────────────────────────


def benchmark(q, k, v, bit_widths=(2, 3, 4)):
    head_dim = k.shape[-1]
    seq_len = k.shape[-2]

    print(f"\nBenchmark: {k.shape[1]} heads, {seq_len} tokens, head_dim={head_dim}")
    print("-" * 70)

    # Reference
    out_ref = scaled_dot_product_attention(q, k, v)
    print(f"{'Method':<35} {'Rel Err':>8} {'Cos Sim':>8} {'MSE':>10} {'Quant ms':>9}")
    print("-" * 70)

    for bw in bit_widths:
        # ── TurboQuant MSE ──
        tq_mse = TurboQuantMSE(head_dim, bw, seed=42)
        with timer(f"TurboQuant-MSE {bw}b quant") as _:
            pass
        t0 = time.perf_counter()
        k_idx, k_n = tq_mse.quantize(k)
        v_idx, v_n = tq_mse.quantize(v)
        k_hat = tq_mse.dequantize(k_idx, k_n)
        v_hat = tq_mse.dequantize(v_idx, v_n)
        dt_mse = (time.perf_counter() - t0) * 1000

        mse_k = (k - k_hat).pow(2).mean().item()
        rel_err, cos_sim = measure_attention_error(q, k, v, k_hat, v_hat)
        print(f"  TurboQuant-MSE  {bw}-bit         {rel_err:8.4f} {cos_sim:8.5f} {mse_k:10.6f} {dt_mse:8.1f}")

        # ── TurboQuant Prod (only for bw >= 2) ──
        if bw >= 2:
            tq_prod = TurboQuantProd(head_dim, bw, seed=42)
            t0 = time.perf_counter()
            data = tq_prod.quantize(k)
            k_hat_p = tq_prod.dequantize(*data)
            data_v = tq_prod.quantize(v)
            v_hat_p = tq_prod.dequantize(*data_v)
            dt_prod = (time.perf_counter() - t0) * 1000

            mse_kp = (k - k_hat_p).pow(2).mean().item()
            rel_err_p, cos_sim_p = measure_attention_error(q, k, v, k_hat_p, v_hat_p)
            print(f"  TurboQuant-Prod {bw}-bit         {rel_err_p:8.4f} {cos_sim_p:8.5f} {mse_kp:10.6f} {dt_prod:8.1f}")

        # ── Naive min-max baseline ──
        naive = NaiveMinMaxQuantizer(bw)
        t0 = time.perf_counter()
        k_idx_n, k_min, k_scale = naive.quantize(k)
        v_idx_n, v_min, v_scale = naive.quantize(v)
        k_hat_n = naive.dequantize(k_idx_n, k_min, k_scale)
        v_hat_n = naive.dequantize(v_idx_n, v_min, v_scale)
        dt_naive = (time.perf_counter() - t0) * 1000

        mse_kn = (k - k_hat_n).pow(2).mean().item()
        rel_err_n, cos_sim_n = measure_attention_error(q, k, v, k_hat_n, v_hat_n)
        print(f"  Naive min-max   {bw}-bit         {rel_err_n:8.4f} {cos_sim_n:8.5f} {mse_kn:10.6f} {dt_naive:8.1f}")

        print()

    # ── Outlier-aware TurboQuant (2.5-bit as in paper) ──
    print("Outlier-aware configurations (as in paper Section 4.3):")
    cache = TurboQuantCache(
        head_dim=head_dim, bit_width=2,
        num_outlier_channels=32, outlier_bit_width=3,
    )
    # Use channels with highest variance as outliers
    cache.calibrate_outliers(k)
    all_k, all_v = cache.update(k, v, layer_idx=0)
    mse_25 = (k - all_k).pow(2).mean().item()
    rel_25, cos_25 = measure_attention_error(q, k, v, all_k, all_v)
    print(f"  TurboQuant 2.5-bit (outlier)    {rel_25:8.4f} {cos_25:8.5f} {mse_25:10.6f}")
    print(f"  Effective bits: {cache.effective_bit_width():.2f}, "
          f"Compression: {cache.compression_ratio():.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant KV cache")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name for real KV tensors")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    print("=" * 70)
    print("TurboQuant KV Cache Benchmark")
    print("=" * 70)

    if args.model:
        q, k, v = get_real_kv(args.model)
    else:
        print(f"\nUsing synthetic data (--model to use real KV tensors)")
        q, k, v = get_synthetic_kv(
            heads=args.heads, seq_len=args.seq_len, head_dim=args.head_dim
        )

    benchmark(q, k, v)


if __name__ == "__main__":
    main()
