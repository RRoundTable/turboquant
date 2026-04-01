#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v2.cuh"

using namespace flashinfer;
using namespace turboquant;

// Launch v2 kernel for a single batch decode
torch::Tensor turboquant_decode_v2(
    torch::Tensor q,           // [batch, num_qo_heads, head_dim] fp16
    torch::Tensor k_quant,     // flat uint8 quantized K
    torch::Tensor v_quant,     // flat uint8 quantized V
    torch::Tensor k_norms,     // flat fp16 norms for K
    torch::Tensor v_norms,     // flat fp16 norms for V
    torch::Tensor indices,     // page table indices
    torch::Tensor indptr,      // page table indptr
    torch::Tensor last_page_len,
    int num_kv_heads,
    int page_size_val,
    int head_dim,
    int padded_dim,
    float sm_scale
) {
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);

    auto o = torch::zeros_like(q);

    // Build paged_kv_turbo_t
    paged_kv_turbo_t<int32_t> tq_kv(
        num_kv_heads, page_size_val, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(k_norms.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_norms.data_ptr<at::Half>()),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        last_page_len.data_ptr<int32_t>(), nullptr
    );

    // Build params
    using Params = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using Variant = DefaultAttention<false, false, false, false>;
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

    // No partitioning: 1 block per batch, kv_tile_idx=0
    auto request_indices_t = torch::arange(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_tile_indices_t = torch::zeros(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kv_chunk_size_t = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));

    params.request_indices = request_indices_t.data_ptr<int32_t>();
    params.kv_tile_indices = kv_tile_indices_t.data_ptr<int32_t>();
    params.kv_chunk_size_ptr = kv_chunk_size_t.data_ptr<int32_t>();
    params.block_valid_mask = nullptr;
    params.o_indptr = nullptr;

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = num_qo_heads / num_kv_heads;

    // Dispatch on head_dim and bdy
    #define LAUNCH(HD, BDX, BDY, BDZ) do { \
        constexpr uint32_t tile_per_bdx = 4; \
        constexpr uint32_t smem_size = 2 * 1 * tile_per_bdx * BDY * BDZ * HD * sizeof(__half) \
                                     + BDY * BDZ * 2 * sizeof(float); \
        dim3 grid(batch_size, num_kv_heads); \
        dim3 block(BDX, BDY, BDZ); \
        TurboQuantPagedDecodeKernel<PosEncodingMode::kNone, tile_per_bdx, 8, BDX, BDY, BDZ, \
                                    Variant, Params> \
            <<<grid, block, smem_size, stream>>>(params); \
    } while(0)

    if (head_dim <= 64) {
        if (bdy == 1) LAUNCH(64, 8, 1, 4);
        else if (bdy == 2) LAUNCH(64, 8, 2, 4);
        else if (bdy == 4) LAUNCH(64, 8, 4, 4);
        else LAUNCH(64, 8, 8, 4);
    } else {
        if (bdy == 1) LAUNCH(128, 16, 1, 4);
        else if (bdy == 2) LAUNCH(128, 16, 2, 4);
        else if (bdy == 4) LAUNCH(128, 16, 4, 4);
        else LAUNCH(128, 16, 8, 4);
    }
    #undef LAUNCH

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v2", &turboquant_decode_v2);
}
