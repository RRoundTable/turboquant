"""Test bdz=2 without GQA (num_qo_heads == num_kv_heads)."""
import torch, math
import torch.nn.functional as F
import sys, os
sys.path.insert(0, "/workspace/turboquant/tests")

# Override to build with bdz=2 and test without GQA
from test_flashinfer_turbo import _build_module, pack_4bit, unpack_4bit, C4, B4, _next_pow2

DEVICE = "cuda"
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 8  # NO GQA: same as kv
PAGE_SIZE = 16

module = _build_module()
print("Module loaded")

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

for seq_len in [1, 4, 16]:
    torch.manual_seed(42)
    K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    dim_chunks = pd // 64
    k_q = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks*32, dtype=torch.uint8, device=DEVICE)
    v_q = torch.zeros_like(k_q)
    k_n = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_n = torch.zeros_like(k_n)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float(); norms = xf.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        norms_fp16 = norms.squeeze(-1).to(torch.float16); rotated = rotate(xf/norms)
        qs = k_q if kv_idx==0 else v_q; ns = k_n if kv_idx==0 else v_n
        for chunk in range(dim_chunks):
            ds=chunk*64; indices=torch.bucketize(rotated[...,ds:ds+64].contiguous(),b4).to(torch.uint8)
            packed=pack_4bit(indices); bo=chunk*32
            for t in range(seq_len):
                bid=t//PAGE_SIZE; boff=t%PAGE_SIZE
                qs[bid,:,boff,bo:bo+32]=packed[t]; ns[bid,:,boff,chunk]=norms_fp16[t]

    # Reference
    RQ = rotate(Q.float()); sm = 1.0/math.sqrt(HEAD_DIM)
    deq_RK = torch.zeros(seq_len, NUM_KV_HEADS, pd, device=DEVICE)
    deq_RV = torch.zeros_like(deq_RK)
    for kv_idx, (qs, ns, deq) in enumerate([(k_q, k_n, deq_RK), (v_q, v_n, deq_RV)]):
        for t in range(seq_len):
            bid=t//PAGE_SIZE; boff=t%PAGE_SIZE
            for h in range(NUM_KV_HEADS):
                norm = ns[bid,h,boff,0].float()
                for chunk in range(dim_chunks):
                    bo=chunk*32; idx=unpack_4bit(qs[bid,h,boff,bo:bo+32],64)
                    deq[t,h,chunk*64:chunk*64+64]=c4[idx.long()]*norm

    rq = RQ.transpose(0,1).unsqueeze(0)
    rk = deq_RK.transpose(0,1).unsqueeze(0)
    rv = deq_RV.transpose(0,1).unsqueeze(0)
    ref = inv_rotate(F.scaled_dot_product_attention(rq,rk,rv,scale=sm).squeeze(0).transpose(0,1))

    # Kernel
    ki=torch.arange(num_pages,dtype=torch.int32,device=DEVICE)
    kp=torch.tensor([0,num_pages],dtype=torch.int32,device=DEVICE)
    kl=torch.tensor([seq_len-(num_pages-1)*PAGE_SIZE],dtype=torch.int32,device=DEVICE)
    fused = module.decode(RQ.half().unsqueeze(0), k_q.view(-1), v_q.view(-1),
        k_n.view(-1).view(torch.uint8).view(torch.float16),
        v_n.view(-1).view(torch.uint8).view(torch.float16),
        ki, kp, kl, NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, pd, sm)
    fused_unrot = inv_rotate(fused.float().squeeze(0))

    cos = F.cosine_similarity(ref.flatten(), fused_unrot.flatten(), dim=0).item()
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  NO GQA seq_len={seq_len:>4}  cos={cos:.6f}  {status}")
