"""Test full Hadamard rotation (signs * FWHT) in kernel vs Python."""
import torch, math, functools, os
from pathlib import Path

DEVICE = "cuda"

@functools.cache
def _build():
    from torch.utils.cpp_extension import load
    csrc = Path(os.environ.get("TURBOQUANT_CSRC", os.path.expanduser("~/workdir/turboquant/csrc")))
    fi_inc = Path(os.environ.get("FLASHINFER_INCLUDE_DIR", os.path.expanduser("~/workdir/flashinfer/include")))
    src = Path("/tmp/test_rotate.cu")
    src.write_text(r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "turboquant/flashinfer_decode_turbo.cuh"
using namespace turboquant;

// Test: rotate a 64-dim vector (signs * x → FWHT)
__global__ void test_rotate_kernel(const float* input, const float* signs, float* output) {
    constexpr uint32_t vec_size = 8, bdx = 8;
    uint32_t tx = threadIdx.x;
    __shared__ float smem[64];

    for (uint32_t i = 0; i < vec_size; i++)
        smem[tx * vec_size + i] = input[tx * vec_size + i];
    __syncwarp(0xFF);

    // Multiply by signs
    for (uint32_t i = 0; i < vec_size; i++)
        smem[tx * vec_size + i] *= signs[tx * vec_size + i];
    __syncwarp(0xFF);

    // FWHT
    warp_fwht_64<vec_size, bdx>(smem, tx, 0xFF);

    for (uint32_t i = 0; i < vec_size; i++)
        output[tx * vec_size + i] = smem[tx * vec_size + i];
}

// Test: inverse rotate (FWHT → signs * result)
__global__ void test_inv_rotate_kernel(const float* input, const float* signs, float* output) {
    constexpr uint32_t vec_size = 8, bdx = 8;
    uint32_t tx = threadIdx.x;
    __shared__ float smem[64];

    for (uint32_t i = 0; i < vec_size; i++)
        smem[tx * vec_size + i] = input[tx * vec_size + i];
    __syncwarp(0xFF);

    warp_fwht_64<vec_size, bdx>(smem, tx, 0xFF);

    for (uint32_t i = 0; i < vec_size; i++)
        smem[tx * vec_size + i] *= signs[tx * vec_size + i];
    __syncwarp(0xFF);

    for (uint32_t i = 0; i < vec_size; i++)
        output[tx * vec_size + i] = smem[tx * vec_size + i];
}

torch::Tensor test_rotate(torch::Tensor input, torch::Tensor signs) {
    auto output = torch::zeros(64, torch::dtype(torch::kFloat32).device(input.device()));
    test_rotate_kernel<<<1, 8, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), signs.data_ptr<float>(), output.data_ptr<float>());
    return output;
}
torch::Tensor test_inv_rotate(torch::Tensor input, torch::Tensor signs) {
    auto output = torch::zeros(64, torch::dtype(torch::kFloat32).device(input.device()));
    test_inv_rotate_kernel<<<1, 8, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), signs.data_ptr<float>(), output.data_ptr<float>());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotate", &test_rotate);
    m.def("inv_rotate", &test_inv_rotate);
}
""")
    return load(name="test_rotate_mod", sources=[str(src)],
        extra_include_paths=[str(csrc/"include"), str(fi_inc)],
        extra_cuda_cflags=["-std=c++17","-O2","--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_HALF2_OPERATORS__"],
        verbose=False)

def python_fwht(x):
    d=x.shape[-1]; x=x.clone(); shape=x.shape; h=1
    while h<d:
        x=x.view(*shape[:-1],d//(2*h),2,h); a=x[...,0,:].clone(); b=x[...,1,:].clone()
        x[...,0,:]=a+b; x[...,1,:]=a-b; x=x.view(shape); h*=2
    return x*(1.0/math.sqrt(d))

gen = torch.Generator(device="cpu"); gen.manual_seed(42)
signs = torch.sign(torch.randn(64, generator=gen)); signs[signs==0]=1.0
signs = signs.to(DEVICE)

module = _build()
print("Compiled")

# Test rotate
torch.manual_seed(99)
x = torch.randn(64, device=DEVICE)
cuda_rot = module.rotate(x, signs)
py_rot = python_fwht((x * signs).unsqueeze(0)).squeeze(0)
cos = torch.nn.functional.cosine_similarity(cuda_rot.flatten(), py_rot.flatten(), dim=0)
print(f"Rotate: cos={cos.item():.6f}")

# Test inverse rotate
cuda_inv = module.inv_rotate(cuda_rot, signs)
cos2 = torch.nn.functional.cosine_similarity(x.flatten(), cuda_inv.flatten(), dim=0)
print(f"Roundtrip (rotate then inverse): cos={cos2.item():.6f}")

# Test with 128-dim (2 chunks)
signs128 = torch.sign(torch.randn(128, generator=gen)); signs128[signs128==0]=1.0; signs128=signs128.to(DEVICE)
x128 = torch.randn(128, device=DEVICE)
# Rotate chunk by chunk
cuda_rot0 = module.rotate(x128[:64], signs128[:64])
cuda_rot1 = module.rotate(x128[64:], signs128[64:])
cuda_rot128 = torch.cat([cuda_rot0, cuda_rot1])

# Python reference (full 128)
def py_rotate_128(x, signs):
    return python_fwht((x * signs).unsqueeze(0)).squeeze(0)

# Note: Python FWHT on 128 dims is a 128-point transform.
# CUDA does two separate 64-point transforms.
# These are DIFFERENT — the 128-dim rotation is NOT the same as two 64-dim rotations!
# But that's what our kernel does (per-chunk FWHT).
# The reference should also do per-chunk FWHT.
py_rot0 = python_fwht((x128[:64] * signs128[:64]).unsqueeze(0)).squeeze(0)
py_rot1 = python_fwht((x128[64:] * signs128[64:]).unsqueeze(0)).squeeze(0)
py_rot128 = torch.cat([py_rot0, py_rot1])
cos3 = torch.nn.functional.cosine_similarity(cuda_rot128.flatten(), py_rot128.flatten(), dim=0)
print(f"128-dim (2 chunks): cos={cos3.item():.6f}")

print("PASS" if all(c > 0.99 for c in [cos.item(), cos2.item(), cos3.item()]) else "FAIL")
