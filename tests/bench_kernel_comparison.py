"""Phase 6a: Standalone kernel benchmark.

Compares:
  - FP16 baseline: PyTorch SDPA reading fp16 KV
  - TurboQuant: Fused CUDA kernel reading 4-bit quantized KV

Both operate on paged KV cache with identical content (quantize-dequant roundtrip).
Measures decode latency per token at various sequence lengths.
"""

import time
import math
import torch
import torch.nn.functional as F

DEVICE = "cuda"

# Config matching Qwen3-1.7B
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16

# Codebook
C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
       0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
       0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def setup_data(seq_len):
    """Create fp16 KV and quantized KV for the same data."""
    pd = _next_pow2(HEAD_DIM)
    s = 1.0 / math.sqrt(pd)
    c4 = torch.tensor([c * s for c in C4], device=DEVICE, dtype=torch.float32)
    b4 = torch.tensor([b * s for b in B4], device=DEVICE, dtype=torch.float32)

    gen = torch.Generator(device="cpu"); gen.manual_seed(42)
    signs = torch.sign(torch.randn(pd, generator=gen))
    signs[signs == 0] = 1.0
    signs = signs.to(DEVICE)

    def fwht(x):
        d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
        while h < d:
            x = x.view(*shape[:-1], d // (2 * h), 2, h)
            a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
            x[..., 0, :] = a + b; x[..., 1, :] = a - b
            x = x.view(shape); h *= 2
        return x * (1.0 / math.sqrt(d))

    def rotate(x): return fwht(x * signs)
    def pack_4bit(idx): return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)

    torch.manual_seed(42)
    K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

    # FP16 baseline: GQA-expanded K, V as fp16
    gqa = NUM_QO_HEADS // NUM_KV_HEADS
    K_fp16 = K.half().repeat_interleave(gqa, dim=1)  # [seq, qo_heads, hd]
    V_fp16 = V.half().repeat_interleave(gqa, dim=1)
    Q_fp16 = Q.half()

    # Quantized: store in HND paged format
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    dim_chunks = pd // 64
    quant_bytes = dim_chunks * 32

    k_quant = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, quant_bytes, dtype=torch.uint8, device=DEVICE)
    v_quant = torch.zeros_like(k_quant)
    k_norms = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_norms = torch.zeros_like(k_norms)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)
        rotated = rotate(normalized)

        qs = k_quant if kv_idx == 0 else v_quant
        ns = k_norms if kv_idx == 0 else v_norms

        for chunk in range(dim_chunks):
            ds = chunk * 64
            cd = rotated[..., ds:ds + 64]
            indices = torch.bucketize(cd.contiguous(), b4).to(torch.uint8)
            packed = pack_4bit(indices)
            bo = chunk * 32
            for t in range(seq_len):
                bid = t // PAGE_SIZE
                boff = t % PAGE_SIZE
                qs[bid, :, boff, bo:bo + 32] = packed[t]
                ns[bid, :, boff, chunk] = norms_fp16[t]

    # Rotated Q for fused kernel
    Q_rotated = rotate(Q.float()).half()

    # Page table
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    lpl = seq_len - (num_pages - 1) * PAGE_SIZE if num_pages > 0 else 0
    kv_last_page_len = torch.tensor([lpl], dtype=torch.int32, device=DEVICE)

    return {
        "Q_fp16": Q_fp16, "K_fp16": K_fp16, "V_fp16": V_fp16,
        "Q_rotated": Q_rotated,
        "k_quant": k_quant, "v_quant": v_quant,
        "k_norms": k_norms, "v_norms": v_norms,
        "kv_indices": kv_indices, "kv_indptr": kv_indptr,
        "kv_last_page_len": kv_last_page_len,
        "signs": signs,
    }


def bench_fp16_sdpa(data, warmup=5, iters=50):
    """Baseline: PyTorch SDPA with fp16 KV."""
    q = data["Q_fp16"].transpose(0, 1).unsqueeze(0)  # [1, heads, 1, hd]
    k = data["K_fp16"].transpose(0, 1).unsqueeze(0)  # [1, heads, seq, hd]
    v = data["V_fp16"].transpose(0, 1).unsqueeze(0)
    sm = 1.0 / math.sqrt(HEAD_DIM)

    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, scale=sm)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v, scale=sm)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # μs


def bench_turboquant_fused(data, module, signs, warmup=5, iters=50):
    """TurboQuant: Fused CUDA kernel with quantized KV."""
    sm = 1.0 / math.sqrt(HEAD_DIM)
    pd = _next_pow2(HEAD_DIM)

    def fwht(x):
        d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
        while h < d:
            x = x.view(*shape[:-1], d // (2 * h), 2, h)
            a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
            x[..., 0, :] = a + b; x[..., 1, :] = a - b
            x = x.view(shape); h *= 2
        return x * (1.0 / math.sqrt(d))

    def run():
        result = module.decode_attention(
            data["Q_rotated"].unsqueeze(0),
            data["k_quant"].view(-1), data["v_quant"].view(-1),
            data["k_norms"].view(-1).view(torch.uint8).view(torch.float16),
            data["v_norms"].view(-1).view(torch.uint8).view(torch.float16),
            data["kv_indices"], data["kv_indptr"], data["kv_last_page_len"],
            NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE,
            HEAD_DIM, pd, sm,
        )
        # Un-rotate output
        return fwht(result.float()) * signs

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # μs


def main():
    print("=" * 65)
    print("Phase 6a: Standalone Kernel Benchmark")
    print(f"Config: head_dim={HEAD_DIM}, kv_heads={NUM_KV_HEADS}, qo_heads={NUM_QO_HEADS}, page_size={PAGE_SIZE}")
    print("=" * 65)

    # Load fused kernel
    try:
        from turboquant.decode_kernel import _get_module
        module = _get_module()
        has_fused = True
        print("Fused CUDA kernel: loaded")
    except Exception as e:
        print(f"Fused CUDA kernel: FAILED ({e})")
        has_fused = False

    print()
    print(f"{'seq_len':>8} {'FP16 SDPA (μs)':>15} {'TQ Fused (μs)':>15} {'Speedup':>10} {'TQ BW saved':>12}")
    print("-" * 65)

    for seq_len in [64, 256, 1024, 4096]:
        data = setup_data(seq_len)

        # FP16 baseline
        fp16_us = bench_fp16_sdpa(data)

        # TurboQuant fused
        if has_fused:
            tq_us = bench_turboquant_fused(data, module, data["signs"])
            speedup = fp16_us / tq_us
            # Bandwidth: fp16 reads 2 bytes/element, TQ reads ~0.53 bytes/element (4-bit + norm)
            bw_saved = f"{(1 - 0.53/2)*100:.0f}%"
        else:
            tq_us = float("nan")
            speedup = float("nan")
            bw_saved = "N/A"

        print(f"{seq_len:>8} {fp16_us:>15.1f} {tq_us:>15.1f} {speedup:>10.2f}x {bw_saved:>12}")

    # Memory comparison
    print()
    print("Memory per KV token per head:")
    print(f"  FP16:       {HEAD_DIM * 2:>4} bytes (K) + {HEAD_DIM * 2:>4} bytes (V) = {HEAD_DIM * 4:>4} bytes")
    quant_per_head = (_next_pow2(HEAD_DIM) // 64) * 32 + (_next_pow2(HEAD_DIM) // 64) * 2
    print(f"  TurboQuant: {quant_per_head:>4} bytes (K) + {quant_per_head:>4} bytes (V) = {quant_per_head * 2:>4} bytes")
    print(f"  Compression: {HEAD_DIM * 4 / (quant_per_head * 2):.2f}x")


if __name__ == "__main__":
    main()
