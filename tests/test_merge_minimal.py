"""Minimal merge debug: bdz=2, no GQA, seq_len=8.
Both tz groups have exactly 4 valid tokens. The merge must combine them.
Compare against bdz=1 (no merge, known correct).
"""
import torch, math, functools, os
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda"
HEAD_DIM = 128
NUM_KV_HEADS = 1
NUM_QO_HEADS = 1
PAGE_SIZE = 16

C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
       0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
       0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]

def _next_pow2(n): return 1 if n <= 1 else 1 << (n - 1).bit_length()
def pack_4bit(idx): return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)

def build_kernel(bdz_val):
    from torch.utils.cpp_extension import load
    csrc = Path(os.environ.get("TURBOQUANT_CSRC", os.path.expanduser("~/workdir/turboquant/csrc")))
    fi_inc = Path(os.environ.get("FLASHINFER_INCLUDE_DIR", os.path.expanduser("~/workdir/flashinfer/include")))
    src = Path(f"/tmp/fi_bdz{bdz_val}.cu")
    src.write_text(f"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "turboquant/flashinfer_decode_turbo.cuh"
#include "turboquant/page_turbo.cuh"
using namespace turboquant;

torch::Tensor run(torch::Tensor q, torch::Tensor kq, torch::Tensor vq,
    torch::Tensor kn, torch::Tensor vn, torch::Tensor ki, torch::Tensor kp,
    torch::Tensor kl, int nqo, int nkv, int ps, int hd, int pd, float sm) {{
    int bs = q.size(0);
    paged_kv_turbo_t<int32_t> paged_kv(nkv, ps, hd, pd, bs,
        kq.data_ptr<uint8_t>(), vq.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(kn.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(vn.data_ptr<at::Half>()),
        ki.data_ptr<int32_t>(), kp.data_ptr<int32_t>(), kl.data_ptr<int32_t>());
    auto o = torch::zeros({{bs, nqo, hd}}, torch::dtype(torch::kFloat16).device(q.device()));
    auto lse = torch::zeros({{bs, nqo}}, torch::dtype(torch::kFloat32).device(q.device()));
    std::vector<int32_t> ri(bs), ti(bs);
    for(int i=0;i<bs;i++) {{ ri[i]=i; ti[i]=0; }}
    auto rit = torch::tensor(ri, torch::dtype(torch::kInt32).device(q.device()));
    auto tit = torch::tensor(ti, torch::dtype(torch::kInt32).device(q.device()));
    int32_t cs = 1000000;
    auto cst = torch::tensor({{cs}}, torch::dtype(torch::kInt32).device(q.device()));
    constexpr uint32_t vec_size=8, bdx=8, bdy=1, bdz={bdz_val}, tsb=4, nss=1;
    constexpr uint32_t tt=tsb*bdy*bdz;
    constexpr uint32_t o_per_tx=4*vec_size;
    constexpr uint32_t mes=2+bdx*o_per_tx;
    uint32_t sb=max(2u*tt*64*2u, bdz*bdy*mes*4u);
    dim3 grid(bs,nkv), block(bdx,bdy,bdz);
    auto kernel=FlashInferDecodeWithTurboQuantKV<vec_size,bdx,bdy,bdz,tsb,nss,int32_t>;
    cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sb);
    kernel<<<grid,block,sb,c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        lse.data_ptr<float>(), paged_kv,
        rit.data_ptr<int32_t>(), tit.data_ptr<int32_t>(),
        nullptr, reinterpret_cast<const uint32_t*>(cst.data_ptr<int32_t>()),
        bs, nqo, nkv, false, sm);
    return o;
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("run", &run); }}
""")
    return load(name=f"fi_bdz{bdz_val}", sources=[str(src)],
        extra_include_paths=[str(csrc/"include"), str(fi_inc)],
        extra_cuda_cflags=["-std=c++17","-O2","--expt-relaxed-constexpr","-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False)

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

seq_len = 8
torch.manual_seed(42)
K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

num_pages = 1; dim_chunks = pd // 64
k_q = torch.zeros(1, NUM_KV_HEADS, PAGE_SIZE, dim_chunks*32, dtype=torch.uint8, device=DEVICE)
v_q = torch.zeros_like(k_q)
k_n = torch.zeros(1, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
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
ki=torch.tensor([0],dtype=torch.int32,device=DEVICE)
kp=torch.tensor([0,1],dtype=torch.int32,device=DEVICE)
kl=torch.tensor([seq_len],dtype=torch.int32,device=DEVICE)
sm = 1.0/math.sqrt(HEAD_DIM)

# Build bdz=1 (reference) and bdz=2 (merge test)
print("Building bdz=1...")
m1 = build_kernel(1)
print("Building bdz=2...")
m2 = build_kernel(2)

out1 = m1.run(RQ.half().unsqueeze(0), k_q.view(-1), v_q.view(-1),
    k_n.view(-1).view(torch.uint8).view(torch.float16),
    v_n.view(-1).view(torch.uint8).view(torch.float16),
    ki, kp, kl, 1, 1, PAGE_SIZE, HEAD_DIM, pd, sm)

out2 = m2.run(RQ.half().unsqueeze(0), k_q.view(-1), v_q.view(-1),
    k_n.view(-1).view(torch.uint8).view(torch.float16),
    v_n.view(-1).view(torch.uint8).view(torch.float16),
    ki, kp, kl, 1, 1, PAGE_SIZE, HEAD_DIM, pd, sm)

cos = F.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0)
print(f"\nbdz=1 vs bdz=2 cosine: {cos.item():.6f}")
print(f"bdz=1 norm: {out1.float().norm():.4f}")
print(f"bdz=2 norm: {out2.float().norm():.4f}")
print(f"bdz=1 [0:5]: {out1[0,0,:5].float().tolist()}")
print(f"bdz=2 [0:5]: {out2[0,0,:5].float().tolist()}")
print("PASS" if cos.item() > 0.99 else "FAIL")
