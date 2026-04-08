"""Benchmark: contiguous v4 vs paged v4 vs SDPA.

Measures the paging overhead by comparing:
  1. SDPA (fp16, contiguous KV) -- baseline
  2. TQ v4 paged (4-bit, page table addressing)
  3. TQ v4 contiguous (4-bit, direct addressing, no paging)

HYP-008 showed 32us paging overhead at seq=1024 (89us paged vs 57us contiguous).
This benchmark validates that the contiguous kernel eliminates that overhead.

Config: Qwen3-1.7B (16 QO heads, 8 KV heads, head_dim=128)
"""

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = "cuda"

# Qwen3-1.7B config
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16

# Codebook boundaries for quantization
B4 = [-2.4009919, -1.8436908, -1.4371418, -1.0993087, -0.7995637,
      -0.5224207, -0.2582488, 0.0,
       0.2582488,  0.5224207,  0.7995637,  1.0993087,  1.4371418,
       1.8436908,  2.4009919]

_CSRC_DIR = Path(os.environ.get("TURBOQUANT_CSRC",
    os.path.expanduser("~/workdir/turboquant/csrc")))
_INCLUDE_DIR = _CSRC_DIR / "include"
_FLASHINFER_INCLUDE = Path(os.environ.get("FLASHINFER_INCLUDE_DIR",
    os.path.expanduser("~/workdir/flashinfer/include")))


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _build_module():
    """JIT-compile the contiguous vs paged comparison kernel."""
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "test_v4_contiguous.cu"
    return load(
        name="tq_v4_contiguous_bench",
        sources=[str(src)],
        extra_include_paths=[str(_INCLUDE_DIR), str(_FLASHINFER_INCLUDE)],
        extra_cuda_cflags=[
            "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def _fwht_signs():
    """Generate deterministic Hadamard signs."""
    pd = _next_pow2(HEAD_DIM)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    signs = torch.sign(torch.randn(pd, generator=gen))
    signs[signs == 0] = 1.0
    return signs.to(DEVICE)


def _fwht(x, signs=None):
    """Fast Walsh-Hadamard Transform."""
    d = x.shape[-1]
    x = x.clone()
    shape = x.shape
    h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(shape)
        h *= 2
    x = x * (1.0 / math.sqrt(d))
    if signs is not None:
        x = x * signs
    return x


def _pack_4bit(indices):
    """Pack pairs of 4-bit indices into bytes."""
    return ((indices[..., 0::2] << 4) | (indices[..., 1::2] & 0x0F)).to(torch.uint8)


@torch.no_grad()
def setup_data(seq_len):
    """Create quantized KV data in BOTH paged and contiguous layouts.

    Returns dict with:
      - SDPA inputs (fp16 contiguous KV, expanded for GQA)
      - Paged v4 inputs (quantized, page table)
      - Contiguous v4 inputs (quantized, no page table)
      - Q (rotated for TQ kernels)
    """
    pd = _next_pow2(HEAD_DIM)
    dim_chunks = pd // 64
    quant_bytes = dim_chunks * 32
    signs = _fwht_signs()
    b4 = torch.tensor(B4, device=DEVICE, dtype=torch.float32)

    torch.manual_seed(seq_len)  # reproducible per seq_len
    K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

    # -- SDPA fp16 inputs (GQA-expanded) --
    gqa = NUM_QO_HEADS // NUM_KV_HEADS
    K_fp16 = K.half().repeat_interleave(gqa, dim=1)
    V_fp16 = V.half().repeat_interleave(gqa, dim=1)
    Q_fp16 = Q.half()

    # -- Quantize --
    # Contiguous: [1, num_kv_heads, seq_len, quant_bytes] / [1, num_kv_heads, seq_len, dim_chunks]
    k_quant_c = torch.zeros(1, NUM_KV_HEADS, seq_len, quant_bytes, dtype=torch.uint8, device=DEVICE)
    v_quant_c = torch.zeros_like(k_quant_c)
    k_norms_c = torch.zeros(1, NUM_KV_HEADS, seq_len, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_norms_c = torch.zeros_like(k_norms_c)

    # Paged: [num_pages, num_kv_heads, page_size, quant_bytes]
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    k_quant_p = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, quant_bytes, dtype=torch.uint8, device=DEVICE)
    v_quant_p = torch.zeros_like(k_quant_p)
    k_norms_p = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_norms_p = torch.zeros_like(k_norms_p)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)

        # Pad to padded_dim and rotate
        if pd > HEAD_DIM:
            padded = torch.zeros(seq_len, NUM_KV_HEADS, pd, device=DEVICE)
            padded[..., :HEAD_DIM] = normalized
            normalized = padded
        rotated = _fwht(normalized, signs)

        qs_c = k_quant_c if kv_idx == 0 else v_quant_c
        ns_c = k_norms_c if kv_idx == 0 else v_norms_c
        qs_p = k_quant_p if kv_idx == 0 else v_quant_p
        ns_p = k_norms_p if kv_idx == 0 else v_norms_p

        for chunk in range(dim_chunks):
            ds = chunk * 64
            cd = rotated[..., ds:ds + 64]
            indices = torch.bucketize(cd.contiguous(), b4).to(torch.uint8)
            packed = _pack_4bit(indices)  # [seq_len, num_kv_heads, 32]
            bo = chunk * 32

            # Contiguous layout
            qs_c[0, :, :, bo:bo + 32] = packed.permute(1, 0, 2)
            ns_c[0, :, :, chunk] = norms_fp16.permute(1, 0)

            # Paged layout
            for t in range(seq_len):
                bid = t // PAGE_SIZE
                boff = t % PAGE_SIZE
                qs_p[bid, :, boff, bo:bo + 32] = packed[t]
                ns_p[bid, :, boff, chunk] = norms_fp16[t]

    # Rotated Q for TQ kernels
    Q_padded = Q.float()
    if pd > HEAD_DIM:
        Q_padded = torch.zeros(1, NUM_QO_HEADS, pd, device=DEVICE)
        Q_padded[..., :HEAD_DIM] = Q.float()
    Q_rotated = _fwht(Q_padded, signs).half()

    # Page table
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    lpl = seq_len - (num_pages - 1) * PAGE_SIZE if num_pages > 0 else 0
    kv_last_page_len = torch.tensor([lpl], dtype=torch.int32, device=DEVICE)

    return {
        # SDPA
        "Q_fp16": Q_fp16, "K_fp16": K_fp16, "V_fp16": V_fp16,
        # TQ common
        "Q_rotated": Q_rotated, "signs": signs,
        # TQ paged
        "k_quant_p": k_quant_p, "v_quant_p": v_quant_p,
        "k_norms_p": k_norms_p, "v_norms_p": v_norms_p,
        "kv_indices": kv_indices, "kv_indptr": kv_indptr,
        "kv_last_page_len": kv_last_page_len,
        # TQ contiguous
        "k_quant_c": k_quant_c, "v_quant_c": v_quant_c,
        "k_norms_c": k_norms_c, "v_norms_c": v_norms_c,
    }


def bench_sdpa(data, warmup=10, iters=100):
    """FP16 SDPA baseline."""
    q = data["Q_fp16"].transpose(0, 1).unsqueeze(0)
    k = data["K_fp16"].transpose(0, 1).unsqueeze(0)
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
    return start.elapsed_time(end) / iters * 1000  # us


def bench_v4_paged(data, module, warmup=10, iters=100):
    """TQ v4 paged kernel."""
    sm = 1.0 / math.sqrt(HEAD_DIM)
    pd = _next_pow2(HEAD_DIM)

    def run():
        return module.decode_v4_paged(
            data["Q_rotated"].unsqueeze(0),
            data["k_quant_p"].reshape(-1).contiguous(),
            data["v_quant_p"].reshape(-1).contiguous(),
            data["k_norms_p"].reshape(-1).contiguous().view(torch.uint8).view(torch.float16),
            data["v_norms_p"].reshape(-1).contiguous().view(torch.uint8).view(torch.float16),
            data["kv_indices"], data["kv_indptr"], data["kv_last_page_len"],
            NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, pd, sm,
        )

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
    return start.elapsed_time(end) / iters * 1000  # us


def bench_v4_contiguous(data, module, seq_len, warmup=10, iters=100):
    """TQ v4 contiguous kernel."""
    sm = 1.0 / math.sqrt(HEAD_DIM)
    pd = _next_pow2(HEAD_DIM)

    def run():
        return module.decode_v4_contiguous(
            data["Q_rotated"].unsqueeze(0),
            data["k_quant_c"],
            data["v_quant_c"],
            data["k_norms_c"],
            data["v_norms_c"],
            seq_len, NUM_KV_HEADS, HEAD_DIM, pd, sm,
        )

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
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    print("=" * 80)
    print("Contiguous vs Paged v4 Benchmark")
    print(f"Config: Qwen3-1.7B (qo={NUM_QO_HEADS}, kv={NUM_KV_HEADS}, "
          f"hd={HEAD_DIM}, page_size={PAGE_SIZE})")
    print("=" * 80)

    print("\nCompiling kernel...", flush=True)
    module = _build_module()
    print("Kernel compiled.\n")

    seq_lens = [128, 256, 512, 1024, 2048]

    print(f"{'seq_len':>8} | {'SDPA (us)':>10} | {'v4 paged (us)':>14} | "
          f"{'v4 contig (us)':>15} | {'paged/SDPA':>11} | {'contig/SDPA':>12} | "
          f"{'paging OH (us)':>15}")
    print("-" * 100)

    for seq_len in seq_lens:
        data = setup_data(seq_len)

        sdpa_us = bench_sdpa(data)
        paged_us = bench_v4_paged(data, module)
        contig_us = bench_v4_contiguous(data, module, seq_len)

        paging_oh = paged_us - contig_us
        paged_ratio = paged_us / sdpa_us
        contig_ratio = contig_us / sdpa_us

        print(f"{seq_len:>8} | {sdpa_us:>10.1f} | {paged_us:>14.1f} | "
              f"{contig_us:>15.1f} | {paged_ratio:>10.2f}x | {contig_ratio:>11.2f}x | "
              f"{paging_oh:>15.1f}")

    # Summary
    print()
    print("Notes:")
    print("  - SDPA uses fp16 KV (2x memory, tensor core eligible)")
    print("  - v4 paged uses 4-bit KV with page table (divmod + indirect load)")
    print("  - v4 contiguous uses 4-bit KV with direct addressing (no paging)")
    print("  - 'paging OH' = paged_us - contig_us (overhead from page table lookups)")
    print("  - All kernels use bdz=16 (256 threads for hd=128, bdy=2)")

    # Memory comparison
    pd = _next_pow2(HEAD_DIM)
    dim_chunks = pd // 64
    quant_per_head = dim_chunks * 32 + dim_chunks * 2  # quant + norms
    fp16_per_head = HEAD_DIM * 2
    print(f"\nMemory per KV token per head:")
    print(f"  FP16 (SDPA):   {fp16_per_head:>4} bytes")
    print(f"  TurboQuant:    {quant_per_head:>4} bytes ({fp16_per_head / quant_per_head:.1f}x smaller)")


if __name__ == "__main__":
    main()
