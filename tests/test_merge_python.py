"""Test: run bdz=1 twice with different token ranges, merge in Python.
This isolates whether the CUDA merge or the CUDA main loop is buggy.
"""
import torch, math, functools, os
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda"
HEAD_DIM = 128; NUM_KV_HEADS = 1; NUM_QO_HEADS = 1; PAGE_SIZE = 16

C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
       0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
       0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]
def _next_pow2(n): return 1 if n <= 1 else 1 << (n - 1).bit_length()
def pack_4bit(idx): return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)

pd = _next_pow2(HEAD_DIM); s = 1.0/math.sqrt(pd)
c4 = torch.tensor([c*s for c in C4], device=DEVICE); b4 = torch.tensor([b*s for b in B4], device=DEVICE)
gen = torch.Generator(device="cpu"); gen.manual_seed(42)
signs = torch.sign(torch.randn(pd, generator=gen)); signs[signs==0]=1.0; signs=signs.to(DEVICE)
def fwht(x):
    d=x.shape[-1]; x=x.clone(); shape=x.shape; h=1
    while h<d:
        x=x.view(*shape[:-1],d//(2*h),2,h); a=x[...,0,:].clone(); b=x[...,1,:].clone()
        x[...,0,:]=a+b; x[...,1,:]=a-b; x=x.view(shape); h*=2
    return x*(1.0/math.sqrt(d))
def rotate(x): return fwht(x*signs)
def inv_rotate(y): return fwht(y)*signs
def unpack_4bit(p,n):
    pi=p.to(torch.int32); o=torch.zeros(*p.shape[:-1],n,dtype=torch.int32,device=p.device)
    o[...,0::2]=(pi>>4)&0x0F; o[...,1::2]=pi&0x0F; return o.to(torch.uint8)

seq_len = 8
torch.manual_seed(42)
K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

num_pages = 1; dim_chunks = pd // 64
k_q = torch.zeros(1, 1, PAGE_SIZE, 64, dtype=torch.uint8, device=DEVICE)
v_q = torch.zeros_like(k_q)
k_n = torch.zeros(1, 1, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
v_n = torch.zeros_like(k_n)
for kv_idx, tensor in enumerate([K, V]):
    xf = tensor.float(); norms = xf.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    norms_fp16 = norms.squeeze(-1).to(torch.float16); rotated = rotate(xf/norms)
    qs = k_q if kv_idx==0 else v_q; ns = k_n if kv_idx==0 else v_n
    for chunk in range(dim_chunks):
        ds=chunk*64; indices=torch.bucketize(rotated[...,ds:ds+64].contiguous(),b4).to(torch.uint8)
        packed=pack_4bit(indices); bo=chunk*32
        for t in range(seq_len):
            qs[0,:,t,bo:bo+32]=packed[t]; ns[0,:,t,chunk]=norms_fp16[t]

RQ = rotate(Q.float())

# Full reference (all 8 tokens, bdz=1)
# Python dequant + SDPA
deq_RK = torch.zeros(seq_len, 1, pd, device=DEVICE)
deq_RV = torch.zeros_like(deq_RK)
for t in range(seq_len):
    norm = k_n[0,0,t,0].float()
    for chunk in range(dim_chunks):
        bo = chunk*32
        idx = unpack_4bit(k_q[0,0,t,bo:bo+32], 64)
        deq_RK[t,0,chunk*64:chunk*64+64] = c4[idx.long()] * norm
    norm = v_n[0,0,t,0].float()
    for chunk in range(dim_chunks):
        bo = chunk*32
        idx = unpack_4bit(v_q[0,0,t,bo:bo+32], 64)
        deq_RV[t,0,chunk*64:chunk*64+64] = c4[idx.long()] * norm

sm = 1.0/math.sqrt(HEAD_DIM)
rq = RQ.transpose(0,1).unsqueeze(0)
rk = deq_RK.transpose(0,1).unsqueeze(0)
rv = deq_RV.transpose(0,1).unsqueeze(0)
ref = F.scaled_dot_product_attention(rq, rk, rv, scale=sm).squeeze(0).transpose(0,1)

# Partial results: tokens 0-3 and tokens 4-7
# Simulate what tz=0 and tz=1 would compute
rk0 = deq_RK[:4].transpose(0,1).unsqueeze(0)
rv0 = deq_RV[:4].transpose(0,1).unsqueeze(0)
rk1 = deq_RK[4:].transpose(0,1).unsqueeze(0)
rv1 = deq_RV[4:].transpose(0,1).unsqueeze(0)

# Compute partial attention (unnormalized)
scores0 = (rq @ rk0.transpose(-2,-1) * sm)  # [1,1,1,4]
scores1 = (rq @ rk1.transpose(-2,-1) * sm)

m0 = scores0.max(dim=-1, keepdim=True).values.squeeze()  # scalar
m1 = scores1.max(dim=-1, keepdim=True).values.squeeze()

exp0 = torch.exp(scores0 - m0)  # [1,1,1,4]
exp1 = torch.exp(scores1 - m1)
d0 = exp0.sum(dim=-1).squeeze()
d1 = exp1.sum(dim=-1).squeeze()
o0 = (exp0 @ rv0).squeeze()  # [128] (unnormalized)
o1 = (exp1 @ rv1).squeeze()

# Online softmax merge
m_new = max(m0.item(), m1.item())
s0 = math.exp(m0.item() - m_new)
s1 = math.exp(m1.item() - m_new)
d_merged = d0.item() * s0 + d1.item() * s1
o_merged = o0 * s0 + o1 * s1
o_final = o_merged / d_merged

# Compare
cos_ref = F.cosine_similarity(ref.flatten(), inv_rotate(o_final.unsqueeze(0)).flatten(), dim=0)
print(f"Python merge vs full SDPA: cos={cos_ref.item():.6f}")

# Now compare against CUDA bdz=1 (all 8 tokens)
import sys; sys.path.insert(0, "/workspace/turboquant/tests")
from test_merge_minimal import build_kernel
m_bdz1 = build_kernel(1)
cuda_out = m_bdz1.run(RQ.half().unsqueeze(0), k_q.view(-1), v_q.view(-1),
    k_n.view(-1).view(torch.uint8).view(torch.float16),
    v_n.view(-1).view(torch.uint8).view(torch.float16),
    torch.tensor([0], dtype=torch.int32, device=DEVICE),
    torch.tensor([0,1], dtype=torch.int32, device=DEVICE),
    torch.tensor([8], dtype=torch.int32, device=DEVICE),
    1, 1, PAGE_SIZE, HEAD_DIM, pd, sm)

cos_cuda_ref = F.cosine_similarity(ref.flatten(), inv_rotate(cuda_out.float().squeeze()).flatten(), dim=0)
print(f"CUDA bdz=1 vs full SDPA: cos={cos_cuda_ref.item():.6f}")

cos_merge_cuda = F.cosine_similarity(inv_rotate(o_final.unsqueeze(0)).flatten(),
                                      inv_rotate(cuda_out.float().squeeze()).flatten(), dim=0)
print(f"Python merge vs CUDA bdz=1: cos={cos_merge_cuda.item():.6f}")
print("Python merge is " + ("CORRECT" if cos_ref.item() > 0.999 else "WRONG"))
