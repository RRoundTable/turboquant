// PyTorch binding for the fused quantize-write kernel.
//
// Exposes quantize_write() which takes fp16 KV tensors and produces
// packed 4-bit quantized output + fp16 norms, matching the v4 contiguous
// decode kernel's expected layout.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "turboquant/quantize_write_kernel.cuh"

static uint32_t next_pow2(uint32_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// quantize_write: fused L2 normalize + codebook quantize + nibble pack + store.
//
// Args:
//   kv_in: [num_tokens, num_heads, head_dim] fp16
//   head_dim: original head dimension (e.g. 128)
//   padded_dim: next power of 2 >= head_dim (e.g. 128)
//
// Returns: tuple of:
//   quant_out: [num_tokens, num_heads, dim_chunks * 32] uint8
//   norms_out: [num_tokens, num_heads, dim_chunks] fp16
std::tuple<torch::Tensor, torch::Tensor> quantize_write(
    torch::Tensor kv_in,
    int head_dim,
    int padded_dim
) {
    TORCH_CHECK(kv_in.is_cuda(), "kv_in must be on CUDA");
    TORCH_CHECK(kv_in.dtype() == torch::kFloat16, "kv_in must be fp16");
    TORCH_CHECK(kv_in.dim() == 3, "kv_in must be [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(kv_in.size(2) == head_dim, "kv_in last dim must match head_dim");
    TORCH_CHECK(padded_dim >= head_dim, "padded_dim must be >= head_dim");
    TORCH_CHECK(padded_dim % 64 == 0, "padded_dim must be a multiple of 64");
    TORCH_CHECK(static_cast<uint32_t>(padded_dim) == next_pow2(head_dim),
                "padded_dim must be next_pow2(head_dim)");

    uint32_t num_tokens = kv_in.size(0);
    uint32_t num_heads  = kv_in.size(1);
    uint32_t dim_chunks = padded_dim / 64;
    uint32_t quant_bytes_per_head = dim_chunks * 32;

    auto quant_out = torch::zeros(
        {(int64_t)num_tokens, (int64_t)num_heads, (int64_t)quant_bytes_per_head},
        torch::dtype(torch::kUInt8).device(kv_in.device())
    );
    auto norms_out = torch::zeros(
        {(int64_t)num_tokens, (int64_t)num_heads, (int64_t)dim_chunks},
        torch::dtype(torch::kFloat16).device(kv_in.device())
    );

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaError_t err = turboquant::launch_quantize_write(
        reinterpret_cast<const __half*>(kv_in.data_ptr<at::Half>()),
        quant_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(norms_out.data_ptr<at::Half>()),
        num_tokens, num_heads, head_dim, padded_dim,
        stream
    );

    TORCH_CHECK(err == cudaSuccess,
                "quantize_write kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(quant_out, norms_out);
}

// quantize_write_hadamard: fused L2 norm + Hadamard rotate + quantize + pack.
// Takes UNROTATED fp16 KV and signs tensor. Does everything in one kernel.
std::tuple<torch::Tensor, torch::Tensor> quantize_write_hadamard(
    torch::Tensor kv_in,
    torch::Tensor signs,    // [padded_dim] float32
    int head_dim,
    int padded_dim
) {
    TORCH_CHECK(kv_in.is_cuda(), "kv_in must be on CUDA");
    TORCH_CHECK(kv_in.dtype() == torch::kFloat32, "kv_in must be float32");
    TORCH_CHECK(kv_in.dim() == 3, "kv_in must be [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(signs.numel() == padded_dim, "signs must have padded_dim elements");

    uint32_t num_tokens = kv_in.size(0);
    uint32_t num_heads  = kv_in.size(1);
    uint32_t dim_chunks = padded_dim / 64;
    uint32_t quant_bytes_per_head = dim_chunks * 32;

    auto quant_out = torch::zeros(
        {(int64_t)num_tokens, (int64_t)num_heads, (int64_t)quant_bytes_per_head},
        torch::dtype(torch::kUInt8).device(kv_in.device())
    );
    auto norms_out = torch::zeros(
        {(int64_t)num_tokens, (int64_t)num_heads, (int64_t)dim_chunks},
        torch::dtype(torch::kFloat16).device(kv_in.device())
    );

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaError_t err = turboquant::launch_quantize_write_hadamard(
        kv_in.data_ptr<float>(),
        quant_out.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(norms_out.data_ptr<at::Half>()),
        signs.data_ptr<float>(),
        num_tokens, num_heads, head_dim, padded_dim,
        stream
    );

    TORCH_CHECK(err == cudaSuccess,
                "quantize_write_hadamard failed: ", cudaGetErrorString(err));

    return std::make_tuple(quant_out, norms_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_write", &quantize_write,
          "Fused quantize-write: L2 normalize + 4-bit codebook quantize + nibble pack",
          py::arg("kv_in"), py::arg("head_dim"), py::arg("padded_dim"));
    m.def("quantize_write_hadamard", &quantize_write_hadamard,
          "Fused quantize-write with Hadamard: normalize + FWHT + quantize + pack",
          py::arg("kv_in"), py::arg("signs"), py::arg("head_dim"), py::arg("padded_dim"));
}
