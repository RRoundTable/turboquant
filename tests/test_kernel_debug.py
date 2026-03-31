"""Incremental debugging: add one variable at a time.
Test 1: 1 head, 1 token (PASS — baseline)
Test 2: 1 head, 16 tokens
Test 3: 8 heads, 1 token
Test 4: 8 heads, 16 tokens
"""
import torch, math
import torch.nn.functional as F
from turboquant.decode_kernel import _get_module

DEVICE = "cuda"
C4 = torch.tensor([-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
     0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368], device=DEVICE)
B4 = torch.tensor([-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
     0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919], device=DEVICE)

hd = 128; pd = 128; s = 1.0 / math.sqrt(pd)
c4 = C4 * s; b4 = B4 * s

gen = torch.Generator(device="cpu"); gen.manual_seed(42)
signs = torch.sign(torch.randn(pd, generator=gen)); signs[signs == 0] = 1.0; signs = signs.to(DEVICE)

def fwht(x):
    d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
        x[..., 0, :] = a + b; x[..., 1, :] = a - b
        x = x.view(shape); h *= 2
    return x * (1.0 / math.sqrt(d))

def rotate(x): return fwht(x * signs)
def inv_rotate(y): return fwht(y) * signs
def pack_4bit(idx): return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)


def run_test(num_kv_heads, num_qo_heads, seq_len, page_size=16):
    dim_chunks = pd // 64
    num_pages = (seq_len + page_size - 1) // page_size

    torch.manual_seed(42)
    K = torch.randn(seq_len, num_kv_heads, hd, device=DEVICE)
    V = torch.randn(seq_len, num_kv_heads, hd, device=DEVICE)
    Q = torch.randn(1, num_qo_heads, hd, device=DEVICE)

    # Quantize K and V → HND: [pages, heads, entries, 64 bytes]
    k_q = torch.zeros(num_pages, num_kv_heads, page_size, 64, dtype=torch.uint8, device=DEVICE)
    v_q = torch.zeros_like(k_q)
    k_n = torch.zeros(num_pages, num_kv_heads, page_size, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_n = torch.zeros_like(k_n)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [seq, heads, 1]
        normalized = xf / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)  # [seq, heads]
        rotated = rotate(normalized)  # [seq, heads, pd]

        qs = k_q if kv_idx == 0 else v_q
        ns = k_n if kv_idx == 0 else v_n

        for chunk in range(dim_chunks):
            ds = chunk * 64
            cd = rotated[..., ds:ds + 64]
            indices = torch.bucketize(cd.contiguous(), b4).to(torch.uint8)
            packed = pack_4bit(indices)  # [seq, heads, 32]

            bo = chunk * 32
            for t in range(seq_len):
                bid = t // page_size
                boff = t % page_size
                qs[bid, :, boff, bo:bo + 32] = packed[t]
                ns[bid, :, boff, chunk] = norms_fp16[t]

    # Python reference: rotated attention
    RQ = rotate(Q.float())
    sm = 1.0 / math.sqrt(hd)

    # Dequant all KV in rotated space
    deq_RK = torch.zeros(seq_len, num_kv_heads, pd, device=DEVICE)
    deq_RV = torch.zeros_like(deq_RK)
    for kv_idx, (qs, ns, deq) in enumerate([(k_q, k_n, deq_RK), (v_q, v_n, deq_RV)]):
        for t in range(seq_len):
            bid = t // page_size
            boff = t % page_size
            for h in range(num_kv_heads):
                norm = ns[bid, h, boff, 0].float()
                for chunk in range(dim_chunks):
                    bo = chunk * 32
                    packed = qs[bid, h, boff, bo:bo + 32]
                    # unpack
                    p = packed.to(torch.int32)
                    idx = torch.zeros(64, dtype=torch.long, device=DEVICE)
                    idx[0::2] = (p >> 4) & 0x0F
                    idx[1::2] = p & 0x0F
                    ds = chunk * 64
                    deq[t, h, ds:ds + 64] = c4[idx] * norm

    # GQA expand
    gqa = num_qo_heads // num_kv_heads
    if gqa > 1:
        deq_RK_exp = deq_RK.repeat_interleave(gqa, dim=1)
        deq_RV_exp = deq_RV.repeat_interleave(gqa, dim=1)
    else:
        deq_RK_exp = deq_RK
        deq_RV_exp = deq_RV

    # Rotated SDPA
    rq = RQ.transpose(0, 1).unsqueeze(0)  # [1, heads, 1, pd]
    rk = deq_RK_exp.transpose(0, 1).unsqueeze(0)  # [1, heads, seq, pd]
    rv = deq_RV_exp.transpose(0, 1).unsqueeze(0)
    ref_rot = F.scaled_dot_product_attention(rq, rk, rv, scale=sm)
    ref_out = inv_rotate(ref_rot.squeeze(0).transpose(0, 1))  # [1, heads, hd]

    # CUDA kernel
    module = _get_module()
    ki = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kp = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    lpl_val = seq_len - (num_pages - 1) * page_size if num_pages > 0 else 0
    kl = torch.tensor([lpl_val], dtype=torch.int32, device=DEVICE)

    # Pass rotated Q, empty signs (kernel does not rotate internally yet)
    empty_signs = torch.empty(0, dtype=torch.float32, device=DEVICE)
    fused = module.decode_attention(
        RQ.half().unsqueeze(0),
        k_q.view(-1), v_q.view(-1),
        k_n.view(-1).view(torch.uint8).view(torch.float16),
        v_n.view(-1).view(torch.uint8).view(torch.float16),
        ki, kp, kl,
        empty_signs,
        num_qo_heads, num_kv_heads, page_size,
        hd, pd, sm)

    fused_unrot = inv_rotate(fused.float().squeeze(0))
    cos = F.cosine_similarity(ref_out.flatten(), fused_unrot.flatten(), dim=0).item()
    return cos


print("=== Incremental Kernel Debug ===")
for label, nkv, nqo, seq in [
    ("1 head, 1 token", 1, 1, 1),
    ("1 head, 4 tokens", 1, 1, 4),
    ("1 head, 16 tokens", 1, 1, 16),
    ("2 heads, 1 token", 2, 2, 1),
    ("8 heads, 1 token", 8, 8, 1),
    ("8 heads, 16 tokens", 8, 8, 16),
    ("8kv 16qo, 1 token (GQA)", 8, 16, 1),
    ("8kv 16qo, 16 tokens (GQA)", 8, 16, 16),
]:
    cos = run_test(nkv, nqo, seq)
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  {label:35s}  cos={cos:.6f}  {status}")
