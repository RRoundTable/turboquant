#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v2.cuh"
#include "flashinfer_decode_turboquant_v3.cuh"

using namespace flashinfer;
using namespace turboquant;

// Shared setup code
static auto make_tq_kv(
    torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    int num_kv_heads, int page_size_val, int head_dim, int padded_dim, int batch_size
) {
    return paged_kv_turbo_t<int32_t>(
        num_kv_heads, page_size_val, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(k_norms.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_norms.data_ptr<at::Half>()),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        last_page_len.data_ptr<int32_t>(), nullptr);
}

static auto make_params(
    torch::Tensor q, torch::Tensor o,
    paged_kv_turbo_t<int32_t>& tq_kv,
    torch::Tensor request_indices_t, torch::Tensor kv_tile_indices_t,
    torch::Tensor kv_chunk_size_t,
    int num_qo_heads, int head_dim, int batch_size, float sm_scale
) {
    using Params = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    Params params;
    params.q = reinterpret_cast<__half*>(q.data_ptr<at::Half>());
    params.tq_kv = tq_kv;
    params.o = reinterpret_cast<__half*>(o.data_ptr<at::Half>());
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = batch_size;
    params.num_qo_heads = num_qo_heads;
    params.q_stride_n = num_qo_heads * head_dim;
    params.q_stride_h = head_dim;
    params.window_left = -1;
    params.logits_soft_cap = 0.0f;
    params.sm_scale = sm_scale;
    params.rope_rcp_scale = 0.0f;
    params.rope_rcp_theta = 0.0f;
    params.partition_kv = false;
    params.request_indices = request_indices_t.data_ptr<int32_t>();
    params.kv_tile_indices = kv_tile_indices_t.data_ptr<int32_t>();
    params.kv_chunk_size_ptr = kv_chunk_size_t.data_ptr<int32_t>();
    params.block_valid_mask = nullptr;
    params.o_indptr = nullptr;
    return params;
}

// V2 decode (for comparison)
torch::Tensor turboquant_decode_v2(
    torch::Tensor q, torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    int num_kv_heads, int page_size_val, int head_dim, int padded_dim, float sm_scale
) {
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    auto o = torch::zeros_like(q);

    auto tq_kv = make_tq_kv(k_quant, v_quant, k_norms, v_norms,
                             indices, indptr, last_page_len,
                             num_kv_heads, page_size_val, head_dim, padded_dim, batch_size);

    auto req_idx = torch::arange(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_tile_idx = torch::zeros(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_cs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));

    using Params = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using Variant = DefaultAttention<false, false, false, false>;
    auto params = make_params(q, o, tq_kv, req_idx, kv_tile_idx, kv_cs,
                              num_qo_heads, head_dim, batch_size, sm_scale);

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = num_qo_heads / num_kv_heads;

    #define LAUNCH_V2(HD, BDX, BDY, BDZ) do { \
        constexpr uint32_t t = 4; \
        constexpr uint32_t smem = 2 * t * BDY * BDZ * HD * sizeof(__half) + BDY * BDZ * 2 * sizeof(float); \
        TurboQuantPagedDecodeKernel<PosEncodingMode::kNone, t, 8, BDX, BDY, BDZ, Variant, Params> \
            <<<dim3(batch_size, num_kv_heads), dim3(BDX, BDY, BDZ), smem, stream>>>(params); \
    } while(0)

    if (head_dim <= 64) {
        if (bdy <= 1) LAUNCH_V2(64, 8, 1, 4);
        else if (bdy <= 2) LAUNCH_V2(64, 8, 2, 4);
        else if (bdy <= 4) LAUNCH_V2(64, 8, 4, 4);
        else LAUNCH_V2(64, 8, 8, 2);
    } else {
        if (bdy <= 1) LAUNCH_V2(128, 16, 1, 4);
        else if (bdy <= 2) LAUNCH_V2(128, 16, 2, 4);
        else if (bdy <= 4) LAUNCH_V2(128, 16, 4, 4);
        else LAUNCH_V2(128, 16, 8, 2);
    }
    #undef LAUNCH_V2
    return o;
}

// V3 decode (cp_async staged pipeline)
torch::Tensor turboquant_decode_v3(
    torch::Tensor q, torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    int num_kv_heads, int page_size_val, int head_dim, int padded_dim, float sm_scale
) {
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    auto o = torch::zeros_like(q);

    auto tq_kv = make_tq_kv(k_quant, v_quant, k_norms, v_norms,
                             indices, indptr, last_page_len,
                             num_kv_heads, page_size_val, head_dim, padded_dim, batch_size);

    auto req_idx = torch::arange(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_tile_idx = torch::zeros(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_cs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));

    using Params = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using Variant = DefaultAttention<false, false, false, false>;
    auto params = make_params(q, o, tq_kv, req_idx, kv_tile_idx, kv_cs,
                              num_qo_heads, head_dim, batch_size, sm_scale);

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = num_qo_heads / num_kv_heads;

    // V3 smem: k_fp16 + v_fp16 + staging + smem_md
    // staging = tile_tokens * dim_chunks * 32, aligned to 16
    #define LAUNCH_V3(HD, BDX, BDY, BDZ) do { \
        constexpr uint32_t t = 4; \
        constexpr uint32_t tt = t * BDY * BDZ; \
        constexpr uint32_t dc = HD / 64; \
        constexpr uint32_t fp16_bytes = tt * HD * sizeof(__half); \
        constexpr uint32_t stg_bytes = ((tt * dc * 32 + 15) / 16) * 16; \
        constexpr uint32_t smem = 2 * fp16_bytes + stg_bytes + 16 + BDY * BDZ * 2 * sizeof(float); \
        TurboQuantPagedDecodeKernelV3<PosEncodingMode::kNone, t, 8, BDX, BDY, BDZ, Variant, Params> \
            <<<dim3(batch_size, num_kv_heads), dim3(BDX, BDY, BDZ), smem, stream>>>(params); \
    } while(0)

    if (head_dim <= 64) {
        if (bdy <= 1) LAUNCH_V3(64, 8, 1, 4);
        else if (bdy <= 2) LAUNCH_V3(64, 8, 2, 4);
        else if (bdy <= 4) LAUNCH_V3(64, 8, 4, 4);
        else LAUNCH_V3(64, 8, 8, 2);
    } else {
        if (bdy <= 1) LAUNCH_V3(128, 16, 1, 4);
        else if (bdy <= 2) LAUNCH_V3(128, 16, 2, 4);
        else if (bdy <= 4) LAUNCH_V3(128, 16, 4, 4);
        else LAUNCH_V3(128, 16, 8, 2);
    }
    #undef LAUNCH_V3
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v2", &turboquant_decode_v2);
    m.def("decode_v3", &turboquant_decode_v3);
}
