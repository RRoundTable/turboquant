"""Test the warp_fwht_64 CUDA function against Python FWHT."""
import torch
import math
import functools
import os
from pathlib import Path

DEVICE = "cuda"

@functools.cache
def _build_fwht_test():
    from torch.utils.cpp_extension import load
    csrc = Path(os.environ.get("TURBOQUANT_CSRC", os.path.expanduser("~/workdir/turboquant/csrc")))
    fi_inc = Path(os.environ.get("FLASHINFER_INCLUDE_DIR", os.path.expanduser("~/workdir/flashinfer/include")))
    src = Path("/tmp/test_fwht.cu")
    src.write_text(r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "turboquant/flashinfer_decode_turbo.cuh"

using namespace turboquant;

// Test kernel: load 64 floats into smem, run warp_fwht_64, write back.
__global__ void test_fwht_kernel(const float* input, float* output) {
    constexpr uint32_t vec_size = 8;
    constexpr uint32_t bdx = 8;
    uint32_t tx = threadIdx.x;

    __shared__ float smem[64];

    // Load input to smem
    for (uint32_t i = 0; i < vec_size; i++) {
        smem[tx * vec_size + i] = input[tx * vec_size + i];
    }
    __syncwarp(0xFF);

    // Run FWHT
    warp_fwht_64<vec_size, bdx>(smem, tx, 0xFF);

    // Write output
    for (uint32_t i = 0; i < vec_size; i++) {
        output[tx * vec_size + i] = smem[tx * vec_size + i];
    }
}

torch::Tensor test_fwht(torch::Tensor input) {
    auto output = torch::zeros(64, torch::dtype(torch::kFloat32).device(input.device()));
    test_fwht_kernel<<<1, 8, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_fwht", &test_fwht);
}
""")
    return load(name="test_fwht_mod", sources=[str(src)],
        extra_include_paths=[str(csrc / "include"), str(fi_inc)],
        extra_cuda_cflags=["-std=c++17", "-O2", "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False)


def python_fwht(x):
    """Reference FWHT."""
    d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
        x[..., 0, :] = a + b; x[..., 1, :] = a - b
        x = x.view(shape); h *= 2
    return x * (1.0 / math.sqrt(d))


module = _build_fwht_test()
print("FWHT test kernel compiled")

# Test 1: known input [1, 0, 0, ..., 0]
x = torch.zeros(64, device=DEVICE)
x[0] = 1.0
cuda_out = module.test_fwht(x)
py_out = python_fwht(x.unsqueeze(0)).squeeze(0)
cos1 = torch.nn.functional.cosine_similarity(cuda_out.flatten(), py_out.flatten(), dim=0)
print(f"Test 1 (impulse): cos={cos1.item():.6f}")
print(f"  Python[0:4]: {py_out[:4].tolist()}")
print(f"  CUDA[0:4]:   {cuda_out[:4].tolist()}")

# Test 2: random input
torch.manual_seed(42)
x2 = torch.randn(64, device=DEVICE)
cuda_out2 = module.test_fwht(x2)
py_out2 = python_fwht(x2.unsqueeze(0)).squeeze(0)
cos2 = torch.nn.functional.cosine_similarity(cuda_out2.flatten(), py_out2.flatten(), dim=0)
print(f"\nTest 2 (random): cos={cos2.item():.6f}")
print(f"  Python norm: {py_out2.norm():.4f}")
print(f"  CUDA norm:   {cuda_out2.norm():.4f}")

# Test 3: involution (FWHT(FWHT(x)) = x)
cuda_out3 = module.test_fwht(cuda_out2)
cos3 = torch.nn.functional.cosine_similarity(x2.flatten(), cuda_out3.flatten(), dim=0)
print(f"\nTest 3 (involution): cos={cos3.item():.6f}")
print("PASS" if cos1.item() > 0.99 and cos2.item() > 0.99 and cos3.item() > 0.99 else "FAIL")
