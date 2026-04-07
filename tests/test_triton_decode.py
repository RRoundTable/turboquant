"""Correctness tests for TurboQuant Triton decode kernel.

Compares Triton kernel output against PyTorch reference on the same
quantized paged KV data. Uses cosine similarity as the metric since
quantization introduces inherent error vs original fp16.
"""

import math
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from turboquant.triton_decode import turboquant_decode
from turboquant.triton_decode_ref import (
    setup_paged_kv,
    reference_decode,
    _padded_dim,
)

DEVICE = "cuda"

# Test configurations: (batch, qo_heads, kv_heads, head_dim, seq_len, page_size)
TEST_CONFIGS = [
    (1, 1, 1, 64, 16, 16),       # minimal: single head, single page
    (1, 1, 1, 128, 16, 16),      # single head, head_dim=128
    (1, 16, 8, 128, 64, 16),     # GQA 2:1, multi-page
    (1, 16, 8, 128, 256, 16),    # GQA 2:1, longer sequence
    (1, 16, 8, 128, 1024, 16),   # GQA 2:1, long sequence
]


def run_test(batch, num_qo_heads, num_kv_heads, head_dim, seq_len, page_size):
    """Run one test configuration and return cosine similarity."""
    padded_dim = _padded_dim(head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Setup paged KV cache with random data
    paged = setup_paged_kv(batch, num_kv_heads, head_dim, seq_len, page_size, DEVICE)

    # Random Q
    torch.manual_seed(123)
    q = torch.randn(batch, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16)

    # Reference: attention on dequantized KV
    ref_out = reference_decode(q.float(), paged["k_deq"], paged["v_deq"], sm_scale)

    # Triton kernel
    triton_out = turboquant_decode(
        q=q,
        k_quant=paged["k_quant"],
        v_quant=paged["v_quant"],
        k_norms=paged["k_norms"],
        v_norms=paged["v_norms"],
        kv_indices=paged["kv_indices"],
        kv_indptr=paged["kv_indptr"],
        last_page_len=paged["last_page_len"],
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        padded_dim=padded_dim,
        page_size=page_size,
        seq_len=seq_len,
        quant_stride_page=paged["quant_stride_page"],
        quant_stride_h=paged["quant_stride_h"],
        quant_stride_n=paged["quant_stride_n"],
        norm_stride_page=paged["norm_stride_page"],
        norm_stride_h=paged["norm_stride_h"],
        norm_stride_n=paged["norm_stride_n"],
        sm_scale=sm_scale,
    )

    # Compare via cosine similarity (per QO head, averaged)
    ref_flat = ref_out.float().view(batch * num_qo_heads, head_dim)
    tri_flat = triton_out.float().view(batch * num_qo_heads, head_dim)

    cos_per_head = torch.nn.functional.cosine_similarity(ref_flat, tri_flat, dim=1)
    cos_min = cos_per_head.min().item()
    cos_mean = cos_per_head.mean().item()

    return cos_min, cos_mean


def main():
    print("TurboQuant Triton Decode Kernel — Correctness Tests")
    print("=" * 72)
    print(f"{'Config':<45} {'cos_min':>8} {'cos_mean':>9} {'Result':>7}")
    print("-" * 72)

    all_pass = True
    for batch, qo, kv, hd, sl, ps in TEST_CONFIGS:
        config_str = f"b={batch} qo={qo} kv={kv} hd={hd} sl={sl} ps={ps}"
        try:
            cos_min, cos_mean = run_test(batch, qo, kv, hd, sl, ps)
            passed = cos_min > 0.99
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"{config_str:<45} {cos_min:>8.5f} {cos_mean:>9.5f} {status:>7}")
        except Exception as e:
            all_pass = False
            print(f"{config_str:<45} {'ERROR':>8} {'':>9} {'FAIL':>7}")
            print(f"  Exception: {e}")

    print("-" * 72)
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
