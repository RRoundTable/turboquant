"""Python wrapper for the fused TurboQuant decode attention kernel.

JIT-compiles the CUDA kernel on first call and exposes a clean Python API.
Uses torch.utils.cpp_extension for JIT compilation.
"""

import functools
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

# Path to our CUDA sources
_CSRC_DIR = Path(__file__).parent.parent / "csrc"
_INCLUDE_DIR = _CSRC_DIR / "include"

# FlashInfer include path (for math.cuh, vec_dtypes.cuh, etc.)
_FLASHINFER_INCLUDE = Path(os.environ.get(
    "FLASHINFER_INCLUDE_DIR",
    os.path.expanduser("~/workdir/flashinfer/include"),
))


@functools.cache
def _get_module():
    """JIT-compile the TurboQuant decode kernel."""
    from torch.utils.cpp_extension import load

    # The binding source that wraps our kernel for PyTorch
    cuda_source = _CSRC_DIR / "src" / "decode_turboquant_binding.cu"

    if not cuda_source.exists():
        _write_binding_source(cuda_source)

    module = load(
        name="turboquant_decode",
        sources=[str(cuda_source)],
        extra_include_paths=[str(_INCLUDE_DIR), str(_FLASHINFER_INCLUDE)],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O2",
            "--expt-relaxed-constexpr",
            "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )
    return module


def _write_binding_source(path: Path):
    """Generate the PyTorch C++ binding source file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "turboquant/decode_turboquant.cuh"
#include "turboquant/page_turbo.cuh"

using namespace turboquant;

// PyTorch entry point for the fused decode kernel.
// Takes pre-quantized KV cache and query, returns attention output.
torch::Tensor turboquant_decode_attention(
    torch::Tensor q,              // [batch_size, num_qo_heads, head_dim] fp16
    torch::Tensor k_quant,        // [total_quant_bytes] uint8
    torch::Tensor v_quant,        // [total_quant_bytes] uint8
    torch::Tensor k_norms,        // [total_norm_entries] fp16
    torch::Tensor v_norms,        // [total_norm_entries] fp16
    torch::Tensor kv_indices,     // [total_pages] int32
    torch::Tensor kv_indptr,      // [batch_size + 1] int32
    torch::Tensor kv_last_page_len, // [batch_size] int32
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    int head_dim,
    int padded_dim,
    float sm_scale
) {
    int batch_size = q.size(0);

    // Build paged_kv struct on host
    paged_kv_turbo_t<int32_t> paged_kv(
        num_kv_heads, page_size, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(),
        v_quant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(k_norms.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_norms.data_ptr<at::Half>()),
        kv_indices.data_ptr<int32_t>(),
        kv_indptr.data_ptr<int32_t>(),
        kv_last_page_len.data_ptr<int32_t>()
    );

    // Allocate output
    auto o = torch::zeros({batch_size, num_qo_heads, head_dim},
                          torch::dtype(torch::kFloat16).device(q.device()));
    auto lse = torch::zeros({batch_size, num_qo_heads},
                            torch::dtype(torch::kFloat32).device(q.device()));

    // Kernel launch parameters (HEAD_DIM=64 specialization)
    constexpr uint32_t HEAD_DIM = 64;
    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = HEAD_DIM / vec_size;  // 8
    constexpr uint32_t bdy = 1;
    constexpr uint32_t bdz = 1;
    constexpr uint32_t tile_size_per_bdx = 4;
    constexpr uint32_t tile_tokens = tile_size_per_bdx * bdy * bdz;

    uint32_t smem_size = 2 * tile_tokens * 64 * sizeof(__half) + 2 * bdy * bdz * sizeof(float);

    dim3 grid(batch_size, num_kv_heads);
    dim3 block(bdx, bdy, bdz);

    auto kernel = BatchDecodeWithTurboQuantKVKernel<HEAD_DIM, vec_size, bdx, bdy, bdz, tile_size_per_bdx, int32_t>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<grid, block, smem_size, c10::cuda::getCurrentCUDAStream()>>>(
        paged_kv,
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        lse.data_ptr<float>(),
        num_qo_heads,
        sm_scale
    );

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attention", &turboquant_decode_attention,
          "TurboQuant fused decode attention with quantized KV cache");
}
""")


class TurboQuantDecoder:
    """Fused decode attention with TurboQuant quantized KV cache.

    Calls the CUDA kernel that loads quantized KV tiles, dequantizes
    inline in registers, and computes attention — all in one kernel.
    """

    def __init__(self, head_dim: int = 64, padded_dim: int = 64,
                 num_kv_heads: int = 1, page_size: int = 16):
        self.head_dim = head_dim
        self.padded_dim = padded_dim
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size

    @torch.no_grad()
    def decode(
        self,
        q: torch.Tensor,              # [batch, num_qo_heads, head_dim] fp16
        k_quant: torch.Tensor,        # flat uint8
        v_quant: torch.Tensor,        # flat uint8
        k_norms: torch.Tensor,        # flat fp16
        v_norms: torch.Tensor,        # flat fp16
        kv_indices: torch.Tensor,     # [total_pages] int32
        kv_indptr: torch.Tensor,      # [batch + 1] int32
        kv_last_page_len: torch.Tensor,  # [batch] int32
        num_qo_heads: int = 1,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run fused decode attention on quantized KV cache.

        Returns: [batch, num_qo_heads, head_dim] fp16 attention output.
        """
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(self.head_dim)

        module = _get_module()
        return module.decode_attention(
            q, k_quant, v_quant, k_norms, v_norms,
            kv_indices, kv_indptr, kv_last_page_len,
            num_qo_heads, self.num_kv_heads, self.page_size,
            self.head_dim, self.padded_dim, sm_scale,
        )
