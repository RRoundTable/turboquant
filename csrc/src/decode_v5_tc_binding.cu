// TurboQuant v5 tensor-core decode kernel binding (HYP-031).
//
// Exposes:
//   decode_v5_tc_contiguous(q, k_quant, v_quant, k_norms, v_norms,
//                           seq_len, num_kv_heads, head_dim, padded_dim, sm_scale)
//
//   decode_v5_tc_contiguous_splitkv(q, k_quant, v_quant, k_norms, v_norms,
//                                   seq_len, num_kv_heads, head_dim, padded_dim,
//                                   sm_scale, num_splits)
//
// Initial support: head_dim=128, bdy=4 (Qwen3-8B config).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v4_contiguous.cuh"  // ContiguousTurboQuantDecodeParams
#include "flashinfer_decode_turboquant_v5_tc.cuh"
#include "flashinfer_decode_turboquant_combine.cuh"

using namespace flashinfer;

// ---- Contiguous v5 tensor-core decode ----------------------------------------

torch::Tensor decode_v5_tc_contiguous(
    torch::Tensor q,           // [batch, num_qo_heads, padded_dim] fp16
    torch::Tensor k_quant,     // [batch, num_kv_heads, seq_len, quant_bytes] uint8
    torch::Tensor v_quant,     // same
    torch::Tensor k_norms,     // [batch, num_kv_heads, seq_len, dim_chunks] fp16
    torch::Tensor v_norms,     // same
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int padded_dim,
    float sm_scale
) {
    TORCH_CHECK(padded_dim == 128, "v5_tc currently only supports head_dim=128, got ", padded_dim);

    int bs = q.size(0);
    int nqo = q.size(1);
    int bdy = nqo / num_kv_heads;
    auto o = torch::zeros_like(q);
    auto stream = c10::cuda::getCurrentCUDAStream();

    using CP = ContiguousTurboQuantDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;

    CP p(
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        bs, num_kv_heads, seq_len, head_dim, padded_dim,
        (__half*)q.data_ptr<at::Half>(), (__half*)o.data_ptr<at::Half>(),
        nullptr, sm_scale, nqo
    );

    // Thread block: (32, 1, num_warps). bdy is a template param, not a thread dim.
    // 32 threads/warp * num_warps <= 1024 => num_warps <= 32.
    // Each warp needs its own smem slice, so more warps = more smem.
    // Start with num_warps=4 (128 threads) to keep smem reasonable.

    #define LAUNCH_V5TC(HD, BDY, NW) do { \
        constexpr uint32_t sm = calc_smem_v5_tc<HD, BDY, NW>(); \
        auto err = cudaFuncSetAttribute( \
            TurboQuantContiguousDecodeKernelV5TC<HD, BDY, NW, V, CP>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        TORCH_CHECK(err == cudaSuccess, "Failed to set smem size: ", cudaGetErrorString(err)); \
        TurboQuantContiguousDecodeKernelV5TC<HD, BDY, NW, V, CP> \
            <<<dim3(bs, num_kv_heads), dim3(32, 1, NW), sm, stream>>>(p); \
    } while(0)

    // Dispatch based on bdy. num_warps=4 keeps smem under 48KB default.
    switch (bdy) {
        case 1: LAUNCH_V5TC(128, 1, 4); break;
        case 2: LAUNCH_V5TC(128, 2, 4); break;
        case 4: LAUNCH_V5TC(128, 4, 4); break;
        case 8: LAUNCH_V5TC(128, 8, 4); break;
        default:
            TORCH_CHECK(false, "v5_tc: unsupported GQA ratio bdy=", bdy);
    }

    #undef LAUNCH_V5TC

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "v5_tc kernel launch failed: ", cudaGetErrorString(cuda_err));
    return o;
}

// ---- fill_split_indices (same as v4 binding) --------------------------------

__global__ void fill_split_indices_v5(
    int32_t* request_indices, int32_t* kv_tile_indices,
    int32_t* split_indptr,
    int batch_size, int actual_splits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * actual_splits;
    if (idx < total) {
        int b = idx / actual_splits;
        int s = idx % actual_splits;
        request_indices[idx] = b;
        kv_tile_indices[idx] = s;
    }
    if (idx <= batch_size) {
        split_indptr[idx] = idx * actual_splits;
    }
}

// ---- Contiguous v5 tensor-core with split-KV --------------------------------

torch::Tensor decode_v5_tc_contiguous_splitkv(
    torch::Tensor q,
    torch::Tensor k_quant,
    torch::Tensor v_quant,
    torch::Tensor k_norms,
    torch::Tensor v_norms,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int padded_dim,
    float sm_scale,
    int num_splits
) {
    TORCH_CHECK(padded_dim == 128, "v5_tc split-KV only supports head_dim=128");

    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    int bdy = num_qo_heads / num_kv_heads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto dev = q.device();

    int kv_chunk_size = (seq_len + num_splits - 1) / num_splits;
    int actual_splits = (seq_len + kv_chunk_size - 1) / kv_chunk_size;
    int padded_batch = batch_size * actual_splits;

    auto request_indices_t = torch::empty(padded_batch, torch::dtype(torch::kInt32).device(dev));
    auto kv_tile_indices_t = torch::empty(padded_batch, torch::dtype(torch::kInt32).device(dev));
    auto kv_chunk_size_t = torch::full({1}, kv_chunk_size, torch::dtype(torch::kInt32).device(dev));
    auto split_indptr_t = torch::empty(batch_size + 1, torch::dtype(torch::kInt32).device(dev));

    int fill_threads = max(padded_batch, batch_size + 1);
    fill_split_indices_v5<<<(fill_threads + 255) / 256, 256, 0, stream>>>(
        request_indices_t.data_ptr<int32_t>(),
        kv_tile_indices_t.data_ptr<int32_t>(),
        split_indptr_t.data_ptr<int32_t>(),
        batch_size, actual_splits);

    auto tmp_o = torch::zeros({padded_batch, num_qo_heads, (int64_t)padded_dim},
                              torch::dtype(torch::kFloat32).device(dev));
    auto tmp_lse = torch::full({padded_batch, num_qo_heads}, -1e30f,
                               torch::dtype(torch::kFloat32).device(dev));

    using CP = ContiguousTurboQuantDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;

    CP p(
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        batch_size, num_kv_heads, seq_len, head_dim, padded_dim,
        (__half*)q.data_ptr<at::Half>(), nullptr, nullptr,
        sm_scale, num_qo_heads
    );

    p.partition_kv = true;
    p.request_indices = request_indices_t.data_ptr<int32_t>();
    p.kv_tile_indices = kv_tile_indices_t.data_ptr<int32_t>();
    p.kv_chunk_size_ptr = kv_chunk_size_t.data_ptr<int32_t>();
    p.block_valid_mask = nullptr;
    p.partition_o = tmp_o.data_ptr<float>();
    p.partition_lse = tmp_lse.data_ptr<float>();

    #define LAUNCH_V5TC_SPLIT(HD, BDY, NW) do { \
        constexpr uint32_t sm = calc_smem_v5_tc<HD, BDY, NW>(); \
        auto err = cudaFuncSetAttribute( \
            TurboQuantContiguousDecodeKernelV5TC<HD, BDY, NW, V, CP>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        TORCH_CHECK(err == cudaSuccess, "Failed to set smem size: ", cudaGetErrorString(err)); \
        TurboQuantContiguousDecodeKernelV5TC<HD, BDY, NW, V, CP> \
            <<<dim3(padded_batch, num_kv_heads), dim3(32, 1, NW), sm, stream>>>(p); \
    } while(0)

    switch (bdy) {
        case 1: LAUNCH_V5TC_SPLIT(128, 1, 4); break;
        case 2: LAUNCH_V5TC_SPLIT(128, 2, 4); break;
        case 4: LAUNCH_V5TC_SPLIT(128, 4, 4); break;
        case 8: LAUNCH_V5TC_SPLIT(128, 8, 4); break;
        default:
            TORCH_CHECK(false, "v5_tc split-KV: unsupported GQA ratio bdy=", bdy);
    }
    #undef LAUNCH_V5TC_SPLIT

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess,
                "v5_tc split-KV kernel launch failed: ", cudaGetErrorString(cuda_err));

    // Combine
    auto o = torch::zeros({batch_size, num_qo_heads, (int64_t)padded_dim},
                          torch::dtype(torch::kFloat16).device(dev));

    SplitKVCombineKernel<__half><<<dim3(batch_size, num_qo_heads), padded_dim, 0, stream>>>(
        tmp_o.data_ptr<float>(),
        tmp_lse.data_ptr<float>(),
        (__half*)o.data_ptr<at::Half>(),
        nullptr,
        split_indptr_t.data_ptr<int32_t>(),
        num_qo_heads,
        padded_dim
    );

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v5_tc_contiguous", &decode_v5_tc_contiguous,
          "TurboQuant v5 tensor-core contiguous decode (HYP-031)");
    m.def("decode_v5_tc_contiguous_splitkv", &decode_v5_tc_contiguous_splitkv,
          "TurboQuant v5 tensor-core contiguous decode with split-KV (HYP-031)");
}
