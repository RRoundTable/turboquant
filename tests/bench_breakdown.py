"""Breakdown: where does the fused kernel spend its time?

Compares:
1. Full kernel (load + dequant + QK + softmax + V)
2. Load-only estimate (how much time reading 4-bit data)
3. Compute-only estimate (how much time doing attention math)
"""

import torch
import math

DEVICE = "cuda"
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16
SEQ_LEN = 1024


def main():
    print("=" * 60)
    print("Kernel Time Breakdown Analysis")
    print("=" * 60)

    # 1. Full kernel (already measured with bdz=16: ~142μs)
    # 2. Pure memory read: how long to read the quantized KV data?
    num_pages = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    dim_chunks = HEAD_DIM // 64

    # Quantized data size
    quant_bytes_per_token = dim_chunks * 32 + dim_chunks * 2  # 64 + 4 = 68 bytes
    total_k_bytes = SEQ_LEN * NUM_KV_HEADS * quant_bytes_per_token
    total_v_bytes = total_k_bytes
    total_kv_bytes = total_k_bytes + total_v_bytes
    print(f"\nData sizes:")
    print(f"  KV quantized: {total_kv_bytes:,} bytes ({total_kv_bytes/1024:.1f} KB)")

    fp16_bytes = SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * 2 * 2  # K+V, 2 bytes each
    print(f"  KV fp16:      {fp16_bytes:,} bytes ({fp16_bytes/1024:.1f} KB)")
    print(f"  Ratio: {fp16_bytes / total_kv_bytes:.2f}x")

    # 3. Pure memory bandwidth test
    data = torch.randn(total_kv_bytes // 4, device=DEVICE)  # float32
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(1000):
        _ = data.sum()
    en.record(); torch.cuda.synchronize()
    read_us = st.elapsed_time(en) / 1000 * 1000
    print(f"\n  Pure read {total_kv_bytes/1024:.0f} KB: {read_us:.1f} μs")

    # 4. SDPA on same-sized data
    q = torch.randn(1, NUM_QO_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k = torch.randn(1, NUM_QO_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v = torch.randn(1, NUM_QO_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(HEAD_DIM))
    torch.cuda.synchronize()
    st.record()
    for _ in range(200):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(HEAD_DIM))
    en.record(); torch.cuda.synchronize()
    sdpa_us = st.elapsed_time(en) / 200 * 1000

    # 5. Codebook lookup test (simulate dequant compute)
    indices = torch.randint(0, 16, (SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), dtype=torch.long, device=DEVICE)
    codebook = torch.randn(16, device=DEVICE)
    for _ in range(10):
        _ = codebook[indices]
    torch.cuda.synchronize()
    st.record()
    for _ in range(200):
        _ = codebook[indices]
    en.record(); torch.cuda.synchronize()
    lookup_us = st.elapsed_time(en) / 200 * 1000

    # 6. Hadamard rotation (single vector, 128 dims)
    x = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

    def fwht(x):
        d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
        while h < d:
            x = x.view(*shape[:-1], d // (2 * h), 2, h)
            a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
            x[..., 0, :] = a + b; x[..., 1, :] = a - b
            x = x.view(shape); h *= 2
        return x * (1.0 / math.sqrt(d))

    for _ in range(10):
        fwht(x)
    torch.cuda.synchronize()
    st.record()
    for _ in range(200):
        fwht(x)
    en.record(); torch.cuda.synchronize()
    fwht_us = st.elapsed_time(en) / 200 * 1000

    print(f"\nTime breakdown (seq_len={SEQ_LEN}):")
    print(f"  SDPA (fp16, target):     {sdpa_us:>8.1f} μs")
    print(f"  TQ fused (bdz=16):       ~142.0 μs  (from previous benchmark)")
    print(f"  Codebook lookup only:    {lookup_us:>8.1f} μs")
    print(f"  FWHT (Q un-rotation):    {fwht_us:>8.1f} μs")
    print(f"  Pure memory read:        {read_us:>8.1f} μs")

    print(f"\nAnalysis:")
    compute_estimate = 142.0 - read_us  # rough
    print(f"  Estimated compute time: ~{compute_estimate:.0f} μs (kernel - read)")
    print(f"  Compute/total ratio: ~{compute_estimate/142:.0f}%")
    print(f"  Memory/total ratio: ~{read_us/142*100:.0f}%")
    print(f"\n  The kernel is {'compute' if compute_estimate > read_us else 'memory'}-bound")
    print(f"  Double-buffering helps if memory-bound. Tensor cores help if compute-bound.")


if __name__ == "__main__":
    main()
