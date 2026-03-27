"""TurboQuant representation accuracy evaluation.

Compares measured distortion against paper's theoretical bounds.
Run: python tests/eval_accuracy.py
"""

import math
import torch

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

DEVICE = "cuda"


def eval_mse_distortion():
    """1. MSE distortion vs paper Table 1.

    Paper Theorem 1: D_mse(b) <= sqrt(3*pi)/2 * 4^{-b} for unit vectors.
    """
    print("=" * 70)
    print("1. MSE Distortion vs Paper Bounds (Theorem 1)")
    print("=" * 70)

    # Paper's theoretical upper bound: sqrt(3*pi)/2 * 4^{-b}
    def theory_bound(b):
        return math.sqrt(3 * math.pi) / 2 * 4 ** (-b)

    dims = [64, 128, 256, 512]
    bit_widths = [1, 2, 3, 4]
    n_vectors = 2000
    n_seeds = 5

    print(f"\n{'dim':>6} {'bits':>5} {'MSE':>10} {'theory':>10} {'ratio':>8} {'pass':>6}")
    print("-" * 50)

    for d in dims:
        for b in bit_widths:
            mses = []
            for seed in range(n_seeds):
                q = TurboQuantMSE(d, b, device=DEVICE, seed=seed)
                x = torch.randn(n_vectors, d, device=DEVICE)
                x = x / x.norm(dim=-1, keepdim=True)

                packed, norms = q.quantize(x)
                x_hat = q.dequantize(packed, norms)

                mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
                mses.append(mse)

            avg_mse = sum(mses) / len(mses)
            bound = theory_bound(b)
            ratio = avg_mse / bound
            ok = "OK" if avg_mse < bound * 2.0 else "FAIL"

            print(f"{d:>6} {b:>5} {avg_mse:>10.6f} {bound:>10.6f} {ratio:>8.3f} {ok:>6}")

        print()


def eval_mse_non_unit():
    """MSE distortion for non-unit vectors (norm scaling check)."""
    print("=" * 70)
    print("1b. MSE on Non-Unit Vectors (Norm Scaling)")
    print("=" * 70)

    d = 128
    n_vectors = 2000

    print(f"\n{'norm':>10} {'bits':>5} {'abs_mse':>12} {'norm_mse':>12} {'cosine':>10}")
    print("-" * 55)

    for norm_scale in [0.01, 1.0, 100.0, 10000.0]:
        for b in [2, 3, 4]:
            q = TurboQuantMSE(d, b, device=DEVICE, seed=42)

            x = torch.randn(n_vectors, d, device=DEVICE) * norm_scale
            packed, norms = q.quantize(x)
            x_hat = q.dequantize(packed, norms)

            abs_mse = (x - x_hat).pow(2).sum(dim=-1).mean().item()
            norm_mse = ((x - x_hat).pow(2).sum(dim=-1) / x.pow(2).sum(dim=-1)).mean().item()
            cos = torch.nn.functional.cosine_similarity(
                x.flatten(0, -2), x_hat.flatten(0, -2), dim=-1
            ).mean().item()

            print(f"{norm_scale:>10.2f} {b:>5} {abs_mse:>12.4f} {norm_mse:>12.6f} {cos:>10.6f}")
        print()


def eval_inner_product():
    """2. Inner-product bias and variance (Theorem 2)."""
    print("=" * 70)
    print("2. Inner Product Bias & Variance (Theorem 2)")
    print("=" * 70)

    d = 128
    n_trials = 500

    print(f"\n{'bits':>5} {'true_ip':>10} {'mean_est':>10} {'bias':>10} {'variance':>10} {'unbiased':>10}")
    print("-" * 60)

    for b in [2, 3, 4]:
        x = torch.randn(d, device=DEVICE)
        x = x / x.norm()
        y = torch.randn(d, device=DEVICE)
        true_ip = (x * y).sum().item()

        estimates = []
        for seed in range(n_trials):
            q = TurboQuantProd(d, b, device=DEVICE, seed=seed)
            data = q.quantize(x.unsqueeze(0))
            x_hat = q.dequantize(*data).squeeze(0)
            est = (y * x_hat).sum().item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        bias = abs(mean_est - true_ip)
        variance = sum((e - mean_est) ** 2 for e in estimates) / len(estimates)
        ok = "OK" if bias < 0.1 else "FAIL"

        print(f"{b:>5} {true_ip:>10.4f} {mean_est:>10.4f} {bias:>10.4f} {variance:>10.4f} {ok:>10}")

    # Compare MSE vs Prod for inner product bias
    print(f"\n--- MSE quantizer bias (should be biased) ---")
    print(f"{'bits':>5} {'true_ip':>10} {'mean_est':>10} {'bias':>10}")
    print("-" * 40)

    x = torch.randn(d, device=DEVICE)
    x = x / x.norm()
    y = torch.randn(d, device=DEVICE)
    true_ip = (x * y).sum().item()

    for b in [2, 3, 4]:
        estimates = []
        for seed in range(n_trials):
            q = TurboQuantMSE(d, b, device=DEVICE, seed=seed)
            packed, norms = q.quantize(x.unsqueeze(0))
            x_hat = q.dequantize(packed, norms).squeeze(0)
            est = (y * x_hat).sum().item()
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        bias = abs(mean_est - true_ip)
        print(f"{b:>5} {true_ip:>10.4f} {mean_est:>10.4f} {bias:>10.4f}")


def eval_attention_quality():
    """3. Attention output quality at different configs."""
    print("\n" + "=" * 70)
    print("3. Attention Output Quality")
    print("=" * 70)

    def attention(q, k, v):
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    d = 128
    n_heads = 8

    print(f"\n{'bits':>5} {'seq_len':>8} {'rel_err':>10} {'cos_sim':>10} {'max_err':>10}")
    print("-" * 50)

    for seq_len in [64, 256, 1024, 4096]:
        for b in [2, 3, 4]:
            torch.manual_seed(42)
            q = torch.randn(1, n_heads, 1, d, device=DEVICE)
            k = torch.randn(1, n_heads, seq_len, d, device=DEVICE)
            v = torch.randn(1, n_heads, seq_len, d, device=DEVICE)

            out_fp = attention(q, k, v)

            quant = TurboQuantMSE(d, b, device=DEVICE, seed=42)
            k_p, k_n = quant.quantize(k)
            v_p, v_n = quant.quantize(v)
            k_hat = quant.dequantize(k_p, k_n)
            v_hat = quant.dequantize(v_p, v_n)
            out_q = attention(q, k_hat, v_hat)

            rel_err = (out_fp - out_q).pow(2).sum().sqrt() / out_fp.pow(2).sum().sqrt()
            cos = torch.nn.functional.cosine_similarity(
                out_fp.flatten(), out_q.flatten(), dim=0
            )
            max_err = (out_fp - out_q).abs().max()

            print(f"{b:>5} {seq_len:>8} {rel_err.item():>10.4f} {cos.item():>10.6f} {max_err.item():>10.4f}")
        print()

    # Multi-query attention quality
    print(f"--- Multi-query (query_len > 1) ---")
    print(f"{'bits':>5} {'q_len':>6} {'kv_len':>7} {'rel_err':>10} {'cos_sim':>10}")
    print("-" * 45)

    for q_len in [16, 64]:
        for b in [3, 4]:
            torch.manual_seed(42)
            q = torch.randn(1, n_heads, q_len, d, device=DEVICE)
            k = torch.randn(1, n_heads, 512, d, device=DEVICE)
            v = torch.randn(1, n_heads, 512, d, device=DEVICE)

            out_fp = attention(q, k, v)

            quant = TurboQuantMSE(d, b, device=DEVICE, seed=42)
            k_hat = quant.dequantize(*quant.quantize(k))
            v_hat = quant.dequantize(*quant.quantize(v))
            out_q = attention(q, k_hat, v_hat)

            rel_err = (out_fp - out_q).pow(2).sum().sqrt() / out_fp.pow(2).sum().sqrt()
            cos = torch.nn.functional.cosine_similarity(
                out_fp.flatten(), out_q.flatten(), dim=0
            )
            print(f"{b:>5} {q_len:>6} {512:>7} {rel_err.item():>10.4f} {cos.item():>10.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    eval_mse_distortion()
    eval_mse_non_unit()
    eval_inner_product()
    eval_attention_quality()
    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)
