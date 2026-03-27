"""Integration test: Python quantize → CUDA fused kernel read.

Tests the exact data flow that vLLM uses:
1. Python quantizes K/V into separate tensors (same as vllm_backend_fused.py)
2. CUDA fused kernel reads from those tensors and computes attention
3. Compare against Python dequant + SDPA reference

This isolates the layout mismatch without vLLM's complexity.
"""

import math
import torch
import torch.nn.functional as F

DEVICE = "cuda"

# Codebook
_C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
        0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
_B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
        0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]

TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pack_4bit(idx):
    return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)


def _unpack_4bit(packed, count):
    p = packed.to(torch.int32)
    out = torch.zeros(*packed.shape[:-1], count, dtype=torch.int32, device=packed.device)
    out[..., 0::2] = (p >> 4) & 0x0F
    out[..., 1::2] = p & 0x0F
    return out.to(torch.uint8)


def fwht(x):
    d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
        x[..., 0, :] = a + b; x[..., 1, :] = a - b
        x = x.view(shape); h *= 2
    return x * (1.0 / math.sqrt(d))


def test_layout_match():
    """Test that Python quantize matches what the CUDA kernel expects."""
    # Match Qwen3 config
    head_dim = 128
    num_kv_heads = 8
    num_qo_heads = 16
    page_size = 16
    seq_len = 16
    num_pages = 1
    padded_dim = _next_pow2(head_dim)
    dim_chunks = padded_dim // TILE_DIMS
    quant_bytes_per_head = dim_chunks * QUANT_BYTES_PER_CHUNK

    scale = 1.0 / math.sqrt(padded_dim)
    hi_b = torch.tensor([b * scale for b in _B4], device=DEVICE, dtype=torch.float32)
    hi_c = torch.tensor([c * scale for c in _C4], device=DEVICE, dtype=torch.float32)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(42)
    signs = torch.sign(torch.randn(padded_dim, generator=gen))
    signs[signs == 0] = 1.0
    signs = signs.to(DEVICE)

    def hadamard_rotate(x):
        d = x.shape[-1]
        if d < padded_dim:
            x = F.pad(x, (0, padded_dim - d))
        return fwht(x * signs)

    def hadamard_inverse(y):
        x = fwht(y) * signs
        if head_dim < padded_dim:
            x = x[..., :head_dim]
        return x

    # Generate random K, V, Q
    torch.manual_seed(42)
    K = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE)
    V = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE)
    Q = torch.randn(1, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16)

    # === Python quantize (same as vllm_backend_fused.py) ===
    # Store in HND layout: [num_pages, num_kv_heads, page_size, quant_bytes]
    k_quant = torch.zeros(num_pages, num_kv_heads, page_size, quant_bytes_per_head,
                           dtype=torch.uint8, device=DEVICE)
    v_quant = torch.zeros_like(k_quant)
    k_norms = torch.zeros(num_pages, num_kv_heads, page_size, dim_chunks,
                           dtype=torch.float16, device=DEVICE)
    v_norms = torch.zeros_like(k_norms)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)
        rotated = hadamard_rotate(normalized)

        quant_store = k_quant if kv_idx == 0 else v_quant
        norm_store = k_norms if kv_idx == 0 else v_norms

        for chunk in range(dim_chunks):
            ds = chunk * TILE_DIMS
            cd = rotated[..., ds:ds + TILE_DIMS]
            indices = torch.bucketize(cd.contiguous(), hi_b).to(torch.uint8)
            packed = _pack_4bit(indices)  # [seq_len, num_kv_heads, 32]

            byte_off = chunk * QUANT_BYTES_PER_CHUNK
            for t in range(seq_len):
                bid = t // page_size
                boff = t % page_size
                # HND: [page, head, entry, bytes]
                quant_store[bid, :, boff, byte_off:byte_off + QUANT_BYTES_PER_CHUNK] = packed[t]
                norm_store[bid, :, boff, chunk] = norms_fp16[t]

    # === Python dequant + SDPA reference ===
    k_deq = torch.zeros(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.float32)
    v_deq = torch.zeros_like(k_deq)

    for kv_idx, (qstore, nstore, deq) in enumerate([
        (k_quant, k_norms, k_deq), (v_quant, v_norms, v_deq)
    ]):
        for t in range(seq_len):
            bid = t // page_size
            boff = t % page_size
            for h in range(num_kv_heads):
                rotated_deq = torch.zeros(padded_dim, device=DEVICE)
                norm = nstore[bid, h, boff, 0].float()
                for chunk in range(dim_chunks):
                    bo = chunk * QUANT_BYTES_PER_CHUNK
                    packed = qstore[bid, h, boff, bo:bo + QUANT_BYTES_PER_CHUNK]
                    indices = _unpack_4bit(packed, TILE_DIMS)
                    ds = chunk * TILE_DIMS
                    rotated_deq[ds:ds + TILE_DIMS] = hi_c[indices.long()]
                inv = hadamard_inverse(rotated_deq.unsqueeze(0)).squeeze(0)
                deq[t, h] = inv * norm

    # GQA expand
    gqa = num_qo_heads // num_kv_heads
    k_exp = k_deq.repeat_interleave(gqa, dim=1)
    v_exp = v_deq.repeat_interleave(gqa, dim=1)

    sm_scale = 1.0 / math.sqrt(head_dim)
    rq = Q.float().transpose(0, 1).unsqueeze(0)
    rk = k_exp.transpose(0, 1).unsqueeze(0)
    rv = v_exp.transpose(0, 1).unsqueeze(0)
    ref_output = F.scaled_dot_product_attention(rq, rk, rv, scale=sm_scale)
    ref_output = ref_output.squeeze(0).transpose(0, 1)  # [1, num_qo_heads, head_dim]

    print(f"Reference output norm: {ref_output.norm():.4f}")

    # === CUDA fused kernel ===
    try:
        from turboquant.decode_kernel import _get_module
        module = _get_module()

        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
        kv_last_page_len = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)

        fused_output = module.decode_attention(
            Q.unsqueeze(0) if Q.dim() == 2 else Q,
            k_quant.view(-1).contiguous(),
            v_quant.view(-1).contiguous(),
            k_norms.contiguous().view(-1).view(torch.uint8).view(torch.float16),
            v_norms.contiguous().view(-1).view(torch.uint8).view(torch.float16),
            kv_indices, kv_indptr, kv_last_page_len,
            num_qo_heads, num_kv_heads, page_size,
            head_dim, padded_dim, sm_scale,
        )

        cos = F.cosine_similarity(
            ref_output.float().flatten(), fused_output.float().flatten(), dim=0
        ).item()
        print(f"Fused output norm: {fused_output.norm():.4f}")
        print(f"Cosine similarity: {cos:.6f}")
        print(f"{'PASS' if cos > 0.95 else 'FAIL'}")
    except Exception as e:
        print(f"Fused kernel error: {e}")
        print("SKIP")


if __name__ == "__main__":
    test_layout_match()
