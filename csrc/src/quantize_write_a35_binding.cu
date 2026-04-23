// PyTorch binding for HYP-056 A_35_prime write kernel.
//
// Exposes quantize_write_a35() which takes fp16 K/V tensors and produces
// 60 B / 64 B-aligned tiles matching the Python reference in
// turboquant/outlier_aware_a35.py (bit-for-bit on the packed bytes).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "turboquant/quantize_write_kernel_a35.cuh"

// Signature:
//   kv_in:       [..., num_kv_heads, 128] fp16 — last two dims must be (H, 128)
//   outlier_idx: [num_kv_heads, 64] int32 — column indices of outlier dims
//   regular_idx: [num_kv_heads, 64] int32 — column indices of regular dims
//   signs_out:   [64] fp32 — ±1 signs for the outlier tier
//   signs_reg:   [64] fp32 — ±1 signs for the regular tier
//
// Returns:
//   tiles:       [..., num_kv_heads, 64] uint8
torch::Tensor quantize_write_a35(
    torch::Tensor kv_in,
    torch::Tensor outlier_idx,
    torch::Tensor regular_idx,
    torch::Tensor signs_out,
    torch::Tensor signs_reg
) {
    TORCH_CHECK(kv_in.is_cuda(), "kv_in must be on CUDA");
    TORCH_CHECK(kv_in.dtype() == torch::kFloat16, "kv_in must be fp16");
    TORCH_CHECK(kv_in.dim() >= 2, "kv_in must have at least 2 dims (H, D)");
    TORCH_CHECK(kv_in.size(-1) == 128, "A_35_prime assumes head_dim=128");

    TORCH_CHECK(outlier_idx.dtype() == torch::kInt32);
    TORCH_CHECK(regular_idx.dtype() == torch::kInt32);
    TORCH_CHECK(signs_out.dtype() == torch::kFloat32);
    TORCH_CHECK(signs_reg.dtype() == torch::kFloat32);
    TORCH_CHECK(outlier_idx.size(-1) == 64 && regular_idx.size(-1) == 64);
    TORCH_CHECK(signs_out.numel() == 64 && signs_reg.numel() == 64);

    int64_t num_kv_heads = kv_in.size(-2);
    TORCH_CHECK(outlier_idx.size(0) == num_kv_heads, "outlier_idx H mismatch");
    TORCH_CHECK(regular_idx.size(0) == num_kv_heads, "regular_idx H mismatch");

    // Flatten leading dims into num_tokens.
    auto kv_in_c = kv_in.contiguous();
    auto orig_shape = kv_in_c.sizes().vec();
    int64_t num_tokens = 1;
    for (int i = 0; i < (int)orig_shape.size() - 2; i++) num_tokens *= orig_shape[i];

    auto kv_flat = kv_in_c.view({num_tokens, num_kv_heads, 128});
    auto tiles = torch::zeros(
        {num_tokens, num_kv_heads, 64},
        torch::dtype(torch::kUInt8).device(kv_in.device())
    );

    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaError_t err = turboquant::launch_quantize_write_a35(
        reinterpret_cast<const __half*>(kv_flat.data_ptr<at::Half>()),
        tiles.data_ptr<uint8_t>(),
        outlier_idx.contiguous().data_ptr<int32_t>(),
        regular_idx.contiguous().data_ptr<int32_t>(),
        signs_out.contiguous().data_ptr<float>(),
        signs_reg.contiguous().data_ptr<float>(),
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_kv_heads),
        stream
    );
    TORCH_CHECK(err == cudaSuccess,
                "quantize_write_a35 kernel launch failed: ", cudaGetErrorString(err));

    // Restore original leading dims.
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_shape.push_back(64);
    return tiles.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_write_a35", &quantize_write_a35,
          "HYP-056 A_35_prime outlier-aware quantize-write kernel");
}
