"""Test FlashInfer-style decode kernel with TurboQuant 4-bit KV.

Verifies correctness and benchmarks against PyTorch SDPA.
"""

import torch
import math
import functools
import os
from pathlib import Path

DEVICE = "cuda"
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16

C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
       0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
       0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def pack_4bit(idx):
    return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)


def unpack_4bit(p, n):
    pi = p.to(torch.int32)
    out = torch.zeros(*p.shape[:-1], n, dtype=torch.int32, device=p.device)
    out[..., 0::2] = (pi >> 4) & 0x0F
    out[..., 1::2] = pi & 0x0F
    return out.to(torch.uint8)


@functools.cache
def _build_module():
    """JIT-compile the FlashInfer-style TurboQuant kernel."""
    from torch.utils.cpp_extension import load

    csrc = Path(os.environ.get("TURBOQUANT_CSRC",
                os.path.expanduser("~/workdir/turboquant/csrc")))
    fi_inc = Path(os.environ.get("FLASHINFER_INCLUDE_DIR",
                  os.path.expanduser("~/workdir/flashinfer/include")))

    src = csrc / "src" / "flashinfer_turbo_binding.cu"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "turboquant/flashinfer_decode_turbo.cuh"
#include "turboquant/page_turbo.cuh"

using namespace turboquant;

torch::Tensor flashinfer_turbo_decode(
    torch::Tensor q,              // [batch, num_qo_heads, head_dim] fp16
    torch::Tensor k_quant,        // flat uint8
    torch::Tensor v_quant,        // flat uint8
    torch::Tensor k_norms,        // flat fp16
    torch::Tensor v_norms,        // flat fp16
    torch::Tensor kv_indices,     // [total_pages] int32
    torch::Tensor kv_indptr,      // [batch + 1] int32
    torch::Tensor kv_last_page_len, // [batch] int32
    torch::Tensor signs,          // [padded_dim] float32 (empty = no rotation)
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int padded_dim,
    float sm_scale
) {
    int batch_size = q.size(0);

    paged_kv_turbo_t<int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(k_norms.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_norms.data_ptr<at::Half>()),
        kv_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
        kv_last_page_len.data_ptr<int32_t>()
    );

    auto o = torch::zeros({batch_size, num_qo_heads, head_dim},
                           torch::dtype(torch::kFloat16).device(q.device()));
    auto lse = torch::zeros({batch_size, num_qo_heads},
                             torch::dtype(torch::kFloat32).device(q.device()));

    // Metadata for non-partitioned decode
    std::vector<int32_t> h_req_indices(batch_size), h_kv_tile_indices(batch_size);
    for (int i = 0; i < batch_size; i++) {
        h_req_indices[i] = i;
        h_kv_tile_indices[i] = 0;
    }
    auto req_indices = torch::tensor(h_req_indices, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_tile_indices = torch::tensor(h_kv_tile_indices, torch::dtype(torch::kInt32).device(q.device()));
    uint32_t kv_chunk_size_val = 1000000;  // large enough to cover all tokens
    auto kv_chunk_size = torch::tensor({(int32_t)kv_chunk_size_val}, torch::dtype(torch::kInt32).device(q.device()));

    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = 64 / vec_size;  // 8
    constexpr uint32_t bdz = 16;
    constexpr uint32_t tile_size_per_bdx = 4;
    constexpr uint32_t num_stages_smem = 1;  // not used for dequant path, but template param

    uint32_t gqa_ratio = num_qo_heads / num_kv_heads;

    #define LAUNCH(BDY) do { \
        constexpr uint32_t tile_tokens = tile_size_per_bdx * BDY * bdz; \
        constexpr uint32_t o_per_tx = 4 * vec_size; \
        constexpr uint32_t merge_entry_size = 2 + bdx * o_per_tx; \
        uint32_t merge_bytes = bdz * BDY * merge_entry_size * sizeof(float); \
        uint32_t tile_bytes = 2 * tile_tokens * 64 * sizeof(__half); \
        uint32_t smem_size = max(tile_bytes, merge_bytes); \
        dim3 grid(batch_size, num_kv_heads); \
        dim3 block(bdx, BDY, bdz); \
        auto kernel = FlashInferDecodeWithTurboQuantKV<vec_size, bdx, BDY, bdz, tile_size_per_bdx, num_stages_smem, int32_t>; \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
        const float* signs_ptr = signs.numel() > 0 ? signs.data_ptr<float>() : nullptr; \
        kernel<<<grid, block, smem_size, c10::cuda::getCurrentCUDAStream()>>>( \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), \
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), \
            lse.data_ptr<float>(), \
            signs_ptr, \
            paged_kv, \
            req_indices.data_ptr<int32_t>(), \
            kv_tile_indices.data_ptr<int32_t>(), \
            nullptr, /* block_valid_mask */ \
            reinterpret_cast<const uint32_t*>(kv_chunk_size.data_ptr<int32_t>()), \
            batch_size, num_qo_heads, num_kv_heads, false, sm_scale); \
    } while(0)

    if (gqa_ratio <= 1) { LAUNCH(1); }
    else if (gqa_ratio <= 2) { LAUNCH(2); }
    else if (gqa_ratio <= 4) { LAUNCH(4); }
    else { LAUNCH(8); }
    #undef LAUNCH

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &flashinfer_turbo_decode);
}
""")

    return load(
        name="flashinfer_turbo",
        sources=[str(src)],
        extra_include_paths=[str(csrc / "include"), str(fi_inc)],
        extra_cuda_cflags=[
            "-std=c++17", "-O2", "--expt-relaxed-constexpr", "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def setup_data(seq_len):
    pd = _next_pow2(HEAD_DIM)
    s = 1.0 / math.sqrt(pd)
    c4 = torch.tensor([c * s for c in C4], device=DEVICE, dtype=torch.float32)
    b4 = torch.tensor([b * s for b in B4], device=DEVICE, dtype=torch.float32)

    gen = torch.Generator(device="cpu"); gen.manual_seed(42)
    signs = torch.sign(torch.randn(pd, generator=gen)); signs[signs == 0] = 1.0
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
    def inv_rotate(y): return fwht(y) * signs

    torch.manual_seed(42)
    K = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    V = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, device=DEVICE)
    Q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, device=DEVICE)

    # Quantize to HND: [pages, heads, entries, bytes]
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    dim_chunks = pd // 64
    k_q = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks * 32, dtype=torch.uint8, device=DEVICE)
    v_q = torch.zeros_like(k_q)
    k_n = torch.zeros(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_n = torch.zeros_like(k_n)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        norms_fp16 = norms.squeeze(-1).to(torch.float16)
        rotated = rotate(xf / norms)
        qs = k_q if kv_idx == 0 else v_q
        ns = k_n if kv_idx == 0 else v_n
        for chunk in range(dim_chunks):
            ds = chunk * 64
            indices = torch.bucketize(rotated[..., ds:ds + 64].contiguous(), b4).to(torch.uint8)
            packed = pack_4bit(indices)
            bo = chunk * 32
            for t in range(seq_len):
                bid = t // PAGE_SIZE; boff = t % PAGE_SIZE
                qs[bid, :, boff, bo:bo + 32] = packed[t]
                ns[bid, :, boff, chunk] = norms_fp16[t]

    # Reference: rotated attention + un-rotate
    RQ = rotate(Q.float())
    deq_RK = torch.zeros(seq_len, NUM_KV_HEADS, pd, device=DEVICE)
    deq_RV = torch.zeros_like(deq_RK)
    for kv_idx, (qs, ns, deq) in enumerate([(k_q, k_n, deq_RK), (v_q, v_n, deq_RV)]):
        for t in range(seq_len):
            bid = t // PAGE_SIZE; boff = t % PAGE_SIZE
            for h in range(NUM_KV_HEADS):
                norm = ns[bid, h, boff, 0].float()
                for chunk in range(dim_chunks):
                    bo = chunk * 32
                    idx = unpack_4bit(qs[bid, h, boff, bo:bo + 32], 64)
                    deq[t, h, chunk * 64:chunk * 64 + 64] = c4[idx.long()] * norm

    gqa = NUM_QO_HEADS // NUM_KV_HEADS
    # Reference uses fp32 rotated Q. Kernel now takes unrotated Q and rotates internally.
    # For reference, rotate Q in fp32 for best precision.
    rq = RQ.transpose(0, 1).unsqueeze(0)
    rk = deq_RK.repeat_interleave(gqa, dim=1).transpose(0, 1).unsqueeze(0)
    rv = deq_RV.repeat_interleave(gqa, dim=1).transpose(0, 1).unsqueeze(0)
    ref = inv_rotate(torch.nn.functional.scaled_dot_product_attention(
        rq, rk, rv, scale=1.0 / math.sqrt(HEAD_DIM)
    ).squeeze(0).transpose(0, 1))

    return {
        "Q": Q, "RQ": RQ, "ref": ref,
        "k_q": k_q, "v_q": v_q, "k_n": k_n, "v_n": v_n,
        "signs": signs, "inv_rotate": inv_rotate,
    }


def main():
    print("=" * 60)
    print("FlashInfer-style TurboQuant Decode Kernel Test")
    print(f"hd={HEAD_DIM}, kv={NUM_KV_HEADS}, qo={NUM_QO_HEADS}")
    print("=" * 60)

    module = _build_module()
    print("Kernel compiled successfully")

    for seq_len in [1, 4, 16, 64]:
        data = setup_data(seq_len)
        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        ki = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
        kp = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
        kl = torch.tensor([seq_len - (num_pages - 1) * PAGE_SIZE], dtype=torch.int32, device=DEVICE)

        # Pass UNROTATED Q + signs — kernel handles rotation internally
        fused = module.decode(
            data["Q"].half().unsqueeze(0),
            data["k_q"].view(-1), data["v_q"].view(-1),
            data["k_n"].view(-1).view(torch.uint8).view(torch.float16),
            data["v_n"].view(-1).view(torch.uint8).view(torch.float16),
            ki, kp, kl,
            data["signs"],  # [padded_dim] float32
            NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 128,
            1.0 / math.sqrt(HEAD_DIM))

        # Output is already un-rotated by kernel
        fused_unrot = fused.float().squeeze(0)

        cos = torch.nn.functional.cosine_similarity(
            data["ref"].flatten(), fused_unrot.flatten(), dim=0).item()
        status = "PASS" if cos > 0.99 else "FAIL"
        print(f"  seq_len={seq_len:>4}  cos={cos:.6f}  {status}")

    # Benchmark
    print("\n--- Benchmark (seq_len=1024) ---")
    data = setup_data(1024)
    num_pages = (1024 + PAGE_SIZE - 1) // PAGE_SIZE
    ki = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kp = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kl = torch.tensor([1024 - (num_pages - 1) * PAGE_SIZE], dtype=torch.int32, device=DEVICE)

    def run_kernel():
        return module.decode(
            data["Q"].half().unsqueeze(0),
            data["k_q"].view(-1), data["v_q"].view(-1),
            data["k_n"].view(-1).view(torch.uint8).view(torch.float16),
            data["v_n"].view(-1).view(torch.uint8).view(torch.float16),
            ki, kp, kl,
            data["signs"],
            NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 128,
            1.0 / math.sqrt(HEAD_DIM))

    for _ in range(10): run_kernel()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(100): run_kernel()
    en.record(); torch.cuda.synchronize()
    tq_us = st.elapsed_time(en) / 100 * 1000

    # SDPA baseline
    q_sdpa = torch.randn(1, NUM_QO_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k_sdpa = torch.randn(1, NUM_QO_HEADS, 1024, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_sdpa = torch.randn(1, NUM_QO_HEADS, 1024, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=1.0/math.sqrt(HEAD_DIM))
    torch.cuda.synchronize()
    st.record()
    for _ in range(100):
        torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=1.0/math.sqrt(HEAD_DIM))
    en.record(); torch.cuda.synchronize()
    sdpa_us = st.elapsed_time(en) / 100 * 1000

    print(f"  SDPA baseline: {sdpa_us:.1f} μs")
    print(f"  TQ FlashInfer: {tq_us:.1f} μs")
    print(f"  Ratio: {tq_us/sdpa_us:.1f}x")


if __name__ == "__main__":
    main()
