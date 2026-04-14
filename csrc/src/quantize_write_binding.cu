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

// quantize_write_kv: combined K+V quantize in one kernel launch.
// Grid covers both K and V tokens (2*num_tokens blocks in x dimension).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_write_kv(
    torch::Tensor key,      // [num_tokens, num_heads, head_dim] float32
    torch::Tensor value,    // [num_tokens, num_heads, head_dim] float32
    torch::Tensor signs,    // [padded_dim] float32
    int head_dim,
    int padded_dim
) {
    TORCH_CHECK(key.is_cuda() && value.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(key.dtype() == torch::kFloat32, "key must be float32");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value must have same shape");

    uint32_t num_tokens = key.size(0);
    uint32_t num_heads  = key.size(1);
    uint32_t dim_chunks = padded_dim / 64;
    uint32_t qbytes = dim_chunks * 32;
    auto dev = key.device();

    // Allocate all outputs at once
    auto kq = torch::zeros({(int64_t)num_tokens, (int64_t)num_heads, (int64_t)qbytes},
                           torch::dtype(torch::kUInt8).device(dev));
    auto vq = torch::zeros_like(kq);
    auto kn = torch::zeros({(int64_t)num_tokens, (int64_t)num_heads, (int64_t)dim_chunks},
                           torch::dtype(torch::kFloat16).device(dev));
    auto vn = torch::zeros_like(kn);

    auto stream = c10::cuda::getCurrentCUDAStream();
    float boundary_scale = rsqrtf(static_cast<float>(padded_dim));
    uint32_t vec_size = padded_dim / 32;

    // Single kernel launch for K, then V (back-to-back on same stream, no sync)
    auto launch = [&](const float* in, uint8_t* qout, __half* nout) {
        dim3 grid(num_tokens, num_heads);
        dim3 block(32);
        switch (vec_size) {
            case 2:
                turboquant::quantize_write_hadamard_kernel<2><<<grid, block, 0, stream>>>(
                    in, qout, nout, signs.data_ptr<float>(),
                    num_tokens, num_heads, head_dim, padded_dim,
                    dim_chunks, qbytes, boundary_scale);
                break;
            case 4:
                turboquant::quantize_write_hadamard_kernel<4><<<grid, block, 0, stream>>>(
                    in, qout, nout, signs.data_ptr<float>(),
                    num_tokens, num_heads, head_dim, padded_dim,
                    dim_chunks, qbytes, boundary_scale);
                break;
            case 8:
                turboquant::quantize_write_hadamard_kernel<8><<<grid, block, 0, stream>>>(
                    in, qout, nout, signs.data_ptr<float>(),
                    num_tokens, num_heads, head_dim, padded_dim,
                    dim_chunks, qbytes, boundary_scale);
                break;
        }
    };

    launch(key.data_ptr<float>(), kq.data_ptr<uint8_t>(),
           reinterpret_cast<__half*>(kn.data_ptr<at::Half>()));
    launch(value.data_ptr<float>(), vq.data_ptr<uint8_t>(),
           reinterpret_cast<__half*>(vn.data_ptr<at::Half>()));

    return std::make_tuple(kq, vq, kn, vn);
}

// ─── Schema-compatible wrappers for torch.library ────────────────────
// torch.library only accepts int64_t / double. Thin wrappers forward to
// the int/float-typed implementations.
static std::tuple<torch::Tensor, torch::Tensor>
quantize_write_op(torch::Tensor kv_in, int64_t head_dim, int64_t padded_dim) {
    return quantize_write(kv_in, (int)head_dim, (int)padded_dim);
}
static std::tuple<torch::Tensor, torch::Tensor>
quantize_write_hadamard_op(torch::Tensor kv_in, torch::Tensor signs,
                           int64_t head_dim, int64_t padded_dim) {
    return quantize_write_hadamard(kv_in, signs, (int)head_dim, (int)padded_dim);
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
quantize_write_kv_op(torch::Tensor key, torch::Tensor value, torch::Tensor signs,
                     int64_t head_dim, int64_t padded_dim) {
    return quantize_write_kv(key, value, signs, (int)head_dim, (int)padded_dim);
}

// ─── torch.library registration (graph-replay-safe dispatch) ──────────
// Register in a SEPARATE namespace (`turboquant_write`) from decode so the
// two extensions can be loaded independently without schema conflicts.
TORCH_LIBRARY(turboquant_write, m) {
    m.def(
        "quantize_write(Tensor kv_in, int head_dim, int padded_dim) "
        "-> (Tensor, Tensor)");
    m.def(
        "quantize_write_hadamard(Tensor kv_in, Tensor signs, int head_dim, "
        "int padded_dim) -> (Tensor, Tensor)");
    m.def(
        "quantize_write_kv(Tensor key, Tensor value, Tensor signs, "
        "int head_dim, int padded_dim) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(turboquant_write, CUDA, m) {
    m.impl("quantize_write", &quantize_write_op);
    m.impl("quantize_write_hadamard", &quantize_write_hadamard_op);
    m.impl("quantize_write_kv", &quantize_write_kv_op);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_write", &quantize_write,
          "Fused quantize-write: L2 normalize + 4-bit codebook quantize + nibble pack",
          py::arg("kv_in"), py::arg("head_dim"), py::arg("padded_dim"));
    m.def("quantize_write_hadamard", &quantize_write_hadamard,
          "Fused quantize-write with Hadamard: normalize + FWHT + quantize + pack",
          py::arg("kv_in"), py::arg("signs"), py::arg("head_dim"), py::arg("padded_dim"));
    m.def("quantize_write_kv", &quantize_write_kv,
          "Combined K+V quantize-write (single call, pre-allocated outputs)",
          py::arg("key"), py::arg("value"), py::arg("signs"),
          py::arg("head_dim"), py::arg("padded_dim"));
}
