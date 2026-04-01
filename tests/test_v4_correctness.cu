#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v2.cuh"
#include "flashinfer_decode_turboquant_v4.cuh"

using namespace flashinfer;
using namespace turboquant;

// V2 decode (reference)
torch::Tensor decode_v2(
    torch::Tensor q, torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    int num_kv_heads, int page_size_val, int head_dim, int padded_dim, float sm_scale
) {
    int bs = q.size(0), nqo = q.size(1);
    auto o = torch::zeros_like(q);
    auto tq = paged_kv_turbo_t<int32_t>(num_kv_heads, page_size_val, head_dim, padded_dim, bs,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        last_page_len.data_ptr<int32_t>(), nullptr);

    using P = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;
    P p; p.q = (__half*)q.data_ptr<at::Half>(); p.tq_kv = tq;
    p.o = (__half*)o.data_ptr<at::Half>(); p.lse = nullptr; p.maybe_alibi_slopes = nullptr;
    p.padded_batch_size = bs; p.num_qo_heads = nqo;
    p.q_stride_n = nqo * head_dim; p.q_stride_h = head_dim;
    p.window_left = -1; p.logits_soft_cap = 0; p.sm_scale = sm_scale;
    p.rope_rcp_scale = 0; p.rope_rcp_theta = 0; p.partition_kv = false;
    auto ri = torch::arange(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kti = torch::zeros(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kcs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));
    p.request_indices = ri.data_ptr<int32_t>();
    p.kv_tile_indices = kti.data_ptr<int32_t>();
    p.kv_chunk_size_ptr = kcs.data_ptr<int32_t>();
    p.block_valid_mask = nullptr; p.o_indptr = nullptr;

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = nqo / num_kv_heads;
    #define L2(HD,BDX,BDY,BDZ) do { \
        constexpr uint32_t t=4, sm=2*t*BDY*BDZ*HD*sizeof(__half)+BDY*BDZ*2*sizeof(float); \
        TurboQuantPagedDecodeKernel<PosEncodingMode::kNone,t,8,BDX,BDY,BDZ,V,P> \
            <<<dim3(bs,num_kv_heads),dim3(BDX,BDY,BDZ),sm,stream>>>(p); \
    } while(0)
    if (head_dim<=64) { if(bdy<=1) L2(64,8,1,4); else if(bdy<=2) L2(64,8,2,4); else L2(64,8,4,4); }
    else { if(bdy<=1) L2(128,16,1,4); else if(bdy<=2) L2(128,16,2,4); else L2(128,16,4,4); }
    #undef L2
    return o;
}

// V4 decode (fused inline dequant)
torch::Tensor decode_v4(
    torch::Tensor q, torch::Tensor k_quant, torch::Tensor v_quant,
    torch::Tensor k_norms, torch::Tensor v_norms,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    int num_kv_heads, int page_size_val, int head_dim, int padded_dim, float sm_scale
) {
    int bs = q.size(0), nqo = q.size(1);
    auto o = torch::zeros_like(q);
    auto tq = paged_kv_turbo_t<int32_t>(num_kv_heads, page_size_val, head_dim, padded_dim, bs,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        last_page_len.data_ptr<int32_t>(), nullptr);

    using P = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;
    P p; p.q = (__half*)q.data_ptr<at::Half>(); p.tq_kv = tq;
    p.o = (__half*)o.data_ptr<at::Half>(); p.lse = nullptr; p.maybe_alibi_slopes = nullptr;
    p.padded_batch_size = bs; p.num_qo_heads = nqo;
    p.q_stride_n = nqo * head_dim; p.q_stride_h = head_dim;
    p.window_left = -1; p.logits_soft_cap = 0; p.sm_scale = sm_scale;
    p.rope_rcp_scale = 0; p.rope_rcp_theta = 0; p.partition_kv = false;
    auto ri = torch::arange(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kti = torch::zeros(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kcs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));
    p.request_indices = ri.data_ptr<int32_t>();
    p.kv_tile_indices = kti.data_ptr<int32_t>();
    p.kv_chunk_size_ptr = kcs.data_ptr<int32_t>();
    p.block_valid_mask = nullptr; p.o_indptr = nullptr;

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = nqo / num_kv_heads;

    // V4 smem: max(staging_layout, sync_state_need) + smem_md
    #define L4(HD,BDX,BDY,BDZ) do { \
        constexpr uint32_t t=4, tt=t*BDY*BDZ, dc=HD/64; \
        constexpr uint32_t stg = ((tt * dc * 32 + 15) / 16) * 16; \
        constexpr uint32_t nrm = tt * dc * sizeof(float); \
        constexpr uint32_t off = tt * dc * sizeof(size_t); \
        constexpr uint32_t main_bytes = stg + nrm + off; \
        constexpr uint32_t sync_bytes = BDZ * BDY * (HD + 2) * sizeof(float); \
        constexpr uint32_t md = BDY * BDZ * 2 * sizeof(float); \
        constexpr uint32_t base = (main_bytes > sync_bytes) ? main_bytes : sync_bytes; \
        constexpr uint32_t sm = base + md + 64; \
        TurboQuantPagedDecodeKernelV4<PosEncodingMode::kNone,t,8,BDX,BDY,BDZ,V,P> \
            <<<dim3(bs,num_kv_heads),dim3(BDX,BDY,BDZ),sm,stream>>>(p); \
    } while(0)
    // Maximize bdz for occupancy: bdx * bdy * bdz <= 1024
    if (head_dim<=64) {
        if(bdy<=1) L4(64,8,1,16);       // 128 threads
        else if(bdy<=2) L4(64,8,2,8);   // 128 threads
        else L4(64,8,4,4);              // 128 threads
    } else {
        if(bdy<=1) L4(128,16,1,16);     // 256 threads
        else if(bdy<=2) L4(128,16,2,8); // 256 threads
        else L4(128,16,4,4);            // 256 threads
    }
    #undef L4
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v2", &decode_v2);
    m.def("decode_v4", &decode_v4);
}
