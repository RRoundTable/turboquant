// TurboQuant v4 contiguous vs paged binding for benchmark comparison.
//
// Exposes:
//   decode_v4_paged(q, k_quant, v_quant, k_norms, v_norms,
//                   indices, indptr, last_page_len,
//                   num_kv_heads, page_size, head_dim, padded_dim, sm_scale)
//
//   decode_v4_contiguous(q, k_quant, v_quant, k_norms, v_norms,
//                        seq_len, num_kv_heads, head_dim, padded_dim, sm_scale)

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v2.cuh"  // TurboQuantBatchDecodeParams
#include "flashinfer_decode_turboquant_v4.cuh"
#include "flashinfer_decode_turboquant_v4_contiguous.cuh"

using namespace flashinfer;
using namespace turboquant;

// ---- Smem calculation for paged v4 (same as decode_v4_binding.cu) ----------
template <uint32_t HD, uint32_t BDX, uint32_t BDY, uint32_t BDZ>
constexpr uint32_t calc_smem_paged() {
    constexpr uint32_t t = 4;
    constexpr uint32_t tt = t * BDY * BDZ;
    constexpr uint32_t dc = HD / 64;
    constexpr uint32_t stg = ((tt * dc * 32 + 15) / 16) * 16;
    constexpr uint32_t nrm = tt * dc * sizeof(float);
    constexpr uint32_t off = tt * dc * sizeof(size_t);
    constexpr uint32_t main_bytes = stg + nrm + off;
    constexpr uint32_t sync_bytes = BDZ * BDY * (HD + 2) * sizeof(float);
    constexpr uint32_t base = (main_bytes > sync_bytes) ? main_bytes : sync_bytes;
    constexpr uint32_t md = BDY * BDZ * 2 * sizeof(float);
    return base + md + 64;
}

// ---- Smem calculation for contiguous v4 (no quant_offsets) -----------------
template <uint32_t HD, uint32_t BDX, uint32_t BDY, uint32_t BDZ>
constexpr uint32_t calc_smem_contiguous() {
    constexpr uint32_t t = 4;
    constexpr uint32_t tt = t * BDY * BDZ;
    constexpr uint32_t dc = HD / 64;
    constexpr uint32_t stg = ((tt * dc * 32 + 15) / 16) * 16;
    constexpr uint32_t nrm = tt * dc * sizeof(float);
    // No quant_offsets array needed for contiguous
    constexpr uint32_t main_bytes = stg + nrm;
    constexpr uint32_t sync_bytes = BDZ * BDY * (HD + 2) * sizeof(float);
    constexpr uint32_t base = (main_bytes > sync_bytes) ? main_bytes : sync_bytes;
    constexpr uint32_t md = BDY * BDZ * 2 * sizeof(float);
    return base + md + 64;
}

// ---- Paged v4 decode -------------------------------------------------------
torch::Tensor decode_v4_paged(
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
    P p;
    p.q = (__half*)q.data_ptr<at::Half>();
    p.tq_kv = tq;
    p.o = (__half*)o.data_ptr<at::Half>();
    p.lse = nullptr;
    p.maybe_alibi_slopes = nullptr;
    p.padded_batch_size = bs;
    p.num_qo_heads = nqo;
    p.q_stride_n = nqo * padded_dim;
    p.q_stride_h = padded_dim;
    p.window_left = -1;
    p.logits_soft_cap = 0;
    p.sm_scale = sm_scale;
    p.rope_rcp_scale = 0;
    p.rope_rcp_theta = 0;
    p.partition_kv = false;
    p.partition_o = nullptr;
    p.partition_lse = nullptr;
    auto ri = torch::arange(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kti = torch::zeros(bs, torch::dtype(torch::kInt32).device(q.device()));
    auto kcs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));
    p.request_indices = ri.data_ptr<int32_t>();
    p.kv_tile_indices = kti.data_ptr<int32_t>();
    p.kv_chunk_size_ptr = kcs.data_ptr<int32_t>();
    p.block_valid_mask = nullptr;
    p.o_indptr = nullptr;

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = nqo / num_kv_heads;

    #define LAUNCH_PAGED(HD,BDX,BDY,BDZ) do { \
        constexpr uint32_t sm = calc_smem_paged<HD,BDX,BDY,BDZ>(); \
        TurboQuantPagedDecodeKernelV4<PosEncodingMode::kNone,4,8,BDX,BDY,BDZ,V,P> \
            <<<dim3(bs,num_kv_heads),dim3(BDX,BDY,BDZ),sm,stream>>>(p); \
    } while(0)

    if (padded_dim<=64) {
        if(bdy<=1) LAUNCH_PAGED(64,8,1,16);
        else if(bdy<=2) LAUNCH_PAGED(64,8,2,8);
        else LAUNCH_PAGED(64,8,4,4);
    } else {
        if(bdy<=1) LAUNCH_PAGED(128,16,1,16);
        else if(bdy<=2) LAUNCH_PAGED(128,16,2,16);
        else LAUNCH_PAGED(128,16,4,4);
    }
    #undef LAUNCH_PAGED
    return o;
}

// ---- Contiguous v4 decode --------------------------------------------------
torch::Tensor decode_v4_contiguous(
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
    int bs = q.size(0);
    int nqo = q.size(1);
    auto o = torch::zeros_like(q);

    using CP = ContiguousTurboQuantDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;

    CP p(
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        bs, num_kv_heads, seq_len, head_dim, padded_dim,
        (__half*)q.data_ptr<at::Half>(), (__half*)o.data_ptr<at::Half>(),
        nullptr, sm_scale, nqo
    );

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = nqo / num_kv_heads;

    #define LAUNCH_CONTIG(HD,BDX,BDY,BDZ) do { \
        constexpr uint32_t sm = calc_smem_contiguous<HD,BDX,BDY,BDZ>(); \
        TurboQuantContiguousDecodeKernelV4<PosEncodingMode::kNone,4,8,BDX,BDY,BDZ,V,CP> \
            <<<dim3(bs,num_kv_heads),dim3(BDX,BDY,BDZ),sm,stream>>>(p); \
    } while(0)

    if (padded_dim<=64) {
        if(bdy<=1) LAUNCH_CONTIG(64,8,1,16);
        else if(bdy<=2) LAUNCH_CONTIG(64,8,2,8);
        else LAUNCH_CONTIG(64,8,4,4);
    } else {
        if(bdy<=1) LAUNCH_CONTIG(128,16,1,16);
        else if(bdy<=2) LAUNCH_CONTIG(128,16,2,16);
        else LAUNCH_CONTIG(128,16,4,4);
    }
    #undef LAUNCH_CONTIG
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v4_paged", &decode_v4_paged,
          "TurboQuant v4 paged decode (with page table)");
    m.def("decode_v4_contiguous", &decode_v4_contiguous,
          "TurboQuant v4 contiguous decode (no paging overhead)");
}
