"""Sweep bdz (thread parallelism) to measure its isolated effect.

Recompiles the kernel with different bdz values and benchmarks each.
"""

import torch
import math
import os
import functools
from pathlib import Path

DEVICE = "cuda"
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16
PAGE_SIZE = 16

_CSRC_DIR = Path(os.environ.get("TURBOQUANT_CSRC",
    os.path.expanduser("~/workdir/turboquant/csrc")))
_INCLUDE_DIR = _CSRC_DIR / "include"
_FLASHINFER_INCLUDE = Path(os.environ.get("FLASHINFER_INCLUDE_DIR",
    os.path.expanduser("~/workdir/flashinfer/include")))


def _build_kernel(bdz_val):
    """JIT-compile kernel with specific bdz."""
    from torch.utils.cpp_extension import load

    src_path = Path(f"/tmp/tq_bdz{bdz_val}_binding.cu")
    src_path.write_text(f"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "turboquant/decode_turboquant.cuh"
#include "turboquant/page_turbo.cuh"

using namespace turboquant;

torch::Tensor tq_decode_bdz{bdz_val}(
    torch::Tensor q, torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor kv_indices, torch::Tensor kv_indptr, torch::Tensor kv_last_page_len,
    int num_qo_heads, int num_kv_heads, int page_size,
    int head_dim, int padded_dim, float sm_scale
) {{
    int batch_size = q.size(0);
    paged_kv_turbo_t<int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(k_norms.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_norms.data_ptr<at::Half>()),
        kv_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
        kv_last_page_len.data_ptr<int32_t>()
    );
    auto o = torch::zeros({{batch_size, num_qo_heads, head_dim}},
                           torch::dtype(torch::kFloat16).device(q.device()));
    auto lse = torch::zeros({{batch_size, num_qo_heads}},
                             torch::dtype(torch::kFloat32).device(q.device()));

    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = 64 / vec_size;  // 8
    constexpr uint32_t bdz = {bdz_val};
    constexpr uint32_t tile_size_per_bdx = 4;

    uint32_t gqa_ratio = num_qo_heads / num_kv_heads;

    #define LAUNCH(BDY) do {{ \\
        constexpr uint32_t tile_tokens = tile_size_per_bdx * BDY * bdz; \\
        uint32_t smem_size = 2 * tile_tokens * 64 * sizeof(__half) + 2 * BDY * bdz * sizeof(float); \\
        dim3 grid(batch_size, num_kv_heads); \\
        dim3 block(bdx, BDY, bdz); \\
        auto kernel = BatchDecodeWithTurboQuantKVKernel<128, vec_size, bdx, BDY, bdz, tile_size_per_bdx, int32_t>; \\
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \\
        kernel<<<grid, block, smem_size, c10::cuda::getCurrentCUDAStream()>>>( \\
            paged_kv, reinterpret_cast<const __half*>(q.data_ptr<at::Half>()), \\
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), lse.data_ptr<float>(), \\
            num_qo_heads, sm_scale); \\
    }} while(0)

    if (gqa_ratio <= 1) {{ LAUNCH(1); }}
    else if (gqa_ratio <= 2) {{ LAUNCH(2); }}
    else if (gqa_ratio <= 4) {{ LAUNCH(4); }}
    else {{ LAUNCH(8); }}
    #undef LAUNCH

    return o;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("decode", &tq_decode_bdz{bdz_val});
}}
""")

    return load(
        name=f"tq_bdz{bdz_val}",
        sources=[str(src_path)],
        extra_include_paths=[str(_INCLUDE_DIR), str(_FLASHINFER_INCLUDE)],
        extra_cuda_cflags=[
            "-std=c++17", "-O2", "--expt-relaxed-constexpr", "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def setup_data(seq_len):
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    dim_chunks = 128 // 64
    k_q = torch.randint(0, 255, (num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks * 32),
                         dtype=torch.uint8, device=DEVICE)
    v_q = torch.randint_like(k_q, 0, 255)
    k_n = torch.ones(num_pages, NUM_KV_HEADS, PAGE_SIZE, dim_chunks, dtype=torch.float16, device=DEVICE)
    v_n = torch.ones_like(k_n)
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    ki = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    kp = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    kl = torch.tensor([seq_len - (num_pages - 1) * PAGE_SIZE], dtype=torch.int32, device=DEVICE)
    return q, k_q, v_q, k_n, v_n, ki, kp, kl


def bench(module, q, k_q, v_q, k_n, v_n, ki, kp, kl, warmup=10, iters=100):
    sm = 1.0 / math.sqrt(HEAD_DIM)
    for _ in range(warmup):
        module.decode(q, k_q.view(-1), v_q.view(-1),
            k_n.view(-1).view(torch.uint8).view(torch.float16),
            v_n.view(-1).view(torch.uint8).view(torch.float16),
            ki, kp, kl, NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 128, sm)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        module.decode(q, k_q.view(-1), v_q.view(-1),
            k_n.view(-1).view(torch.uint8).view(torch.float16),
            v_n.view(-1).view(torch.uint8).view(torch.float16),
            ki, kp, kl, NUM_QO_HEADS, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, 128, sm)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000


def bench_sdpa(seq_len, warmup=10, iters=100):
    q = torch.randn(1, NUM_QO_HEADS, 1, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    k = torch.randn(1, NUM_QO_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v = torch.randn(1, NUM_QO_HEADS, seq_len, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(HEAD_DIM))
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(HEAD_DIM))
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000


def main():
    seq_len = 1024
    print("=" * 65)
    print(f"bdz Sweep Benchmark (seq_len={seq_len})")
    print(f"hd={HEAD_DIM}, kv_heads={NUM_KV_HEADS}, qo_heads={NUM_QO_HEADS}")
    print("=" * 65)

    sdpa_us = bench_sdpa(seq_len)
    print(f"\nFP16 SDPA baseline: {sdpa_us:.1f} μs")

    q, k_q, v_q, k_n, v_n, ki, kp, kl = setup_data(seq_len)

    print(f"\n{'bdz':>4} {'threads/block':>14} {'μs':>8} {'vs SDPA':>10} {'speedup vs bdz=1':>18}")
    print("-" * 60)

    baseline_us = None
    for bdz in [1, 2, 4, 8, 16]:
        gqa = NUM_QO_HEADS // NUM_KV_HEADS  # 2
        total_threads = 8 * gqa * bdz
        print(f"  Compiling bdz={bdz} ({total_threads} threads)...", end=" ", flush=True)
        try:
            module = _build_kernel(bdz)
            us = bench(module, q, k_q, v_q, k_n, v_n, ki, kp, kl)
            if baseline_us is None:
                baseline_us = us
            speedup = baseline_us / us
            print(f"\r{bdz:>4} {total_threads:>14} {us:>8.1f} {us/sdpa_us:>9.1f}x {speedup:>17.2f}x")
        except Exception as e:
            print(f"\r{bdz:>4}   FAILED: {e}")

    print()
    print("Each bdz doubling should roughly halve latency (more threads = more KV parallelism)")


if __name__ == "__main__":
    main()
