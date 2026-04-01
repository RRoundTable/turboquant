// TurboQuant v4 decode kernel binding — generalized dispatch.
//
// Supports any (head_dim, bdy, bdz) combination via compile-time dispatch.
// head_dim ∈ {64, 128, 256}, bdy ∈ {1..8}, bdz auto-maximized.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer_decode_turboquant_v2.cuh"  // for TurboQuantBatchDecodeParams
#include "flashinfer_decode_turboquant_v4.cuh"

using namespace flashinfer;
using namespace turboquant;

// ─── Smem calculation ───────────────────────────────────────────────
template <uint32_t HD, uint32_t BDX, uint32_t BDY, uint32_t BDZ>
constexpr uint32_t calc_smem() {
    constexpr uint32_t t = 4;  // tile_size_per_bdx
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

// ─── Launch helper ──────────────────────────────────────────────────
template <uint32_t HD, uint32_t BDX, uint32_t BDY, uint32_t BDZ>
void launch_v4(
    TurboQuantBatchDecodeParams<__half, __half, int32_t>& params,
    int batch_size, int num_kv_heads, cudaStream_t stream
) {
    using P = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;
    constexpr uint32_t sm = calc_smem<HD, BDX, BDY, BDZ>();
    TurboQuantPagedDecodeKernelV4<PosEncodingMode::kNone, 4, 8, BDX, BDY, BDZ, V, P>
        <<<dim3(batch_size, num_kv_heads), dim3(BDX, BDY, BDZ), sm, stream>>>(params);
}

// ─── bdz dispatch ───────────────────────────────────────────────────
// Given HD and BDY, maximize bdz within 1024 threads, cap at 64.
template <uint32_t HD, uint32_t BDX, uint32_t BDY>
void dispatch_bdz(
    TurboQuantBatchDecodeParams<__half, __half, int32_t>& params,
    int batch_size, int num_kv_heads, int bdz, cudaStream_t stream
) {
    // Dispatch to the closest compiled bdz
    if      (bdz >= 64) launch_v4<HD, BDX, BDY, 64>(params, batch_size, num_kv_heads, stream);
    else if (bdz >= 32) launch_v4<HD, BDX, BDY, 32>(params, batch_size, num_kv_heads, stream);
    else if (bdz >= 16) launch_v4<HD, BDX, BDY, 16>(params, batch_size, num_kv_heads, stream);
    else if (bdz >= 12) launch_v4<HD, BDX, BDY, 12>(params, batch_size, num_kv_heads, stream);
    else if (bdz >=  9) launch_v4<HD, BDX, BDY,  9>(params, batch_size, num_kv_heads, stream);
    else if (bdz >=  8) launch_v4<HD, BDX, BDY,  8>(params, batch_size, num_kv_heads, stream);
    else if (bdz >=  4) launch_v4<HD, BDX, BDY,  4>(params, batch_size, num_kv_heads, stream);
    else if (bdz >=  2) launch_v4<HD, BDX, BDY,  2>(params, batch_size, num_kv_heads, stream);
    else                launch_v4<HD, BDX, BDY,  1>(params, batch_size, num_kv_heads, stream);
}

// ─── Main entry point ───────────────────────────────────────────────
torch::Tensor turboquant_decode_v4(
    torch::Tensor q,              // [batch, num_qo_heads, head_dim] fp16
    torch::Tensor k_quant,        // flat uint8
    torch::Tensor v_quant,        // flat uint8
    torch::Tensor k_norms,        // flat fp16
    torch::Tensor v_norms,        // flat fp16
    torch::Tensor indices,        // page table indices
    torch::Tensor indptr,         // page table indptr
    torch::Tensor last_page_len,  // last page lengths
    int num_kv_heads,
    int page_size,
    int head_dim,
    int padded_dim,
    float sm_scale
) {
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    auto o = torch::zeros_like(q);

    auto tq_kv = paged_kv_turbo_t<int32_t>(
        num_kv_heads, page_size, head_dim, padded_dim, batch_size,
        k_quant.data_ptr<uint8_t>(), v_quant.data_ptr<uint8_t>(),
        (__half*)k_norms.data_ptr<at::Half>(), (__half*)v_norms.data_ptr<at::Half>(),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        last_page_len.data_ptr<int32_t>(), nullptr);

    using P = TurboQuantBatchDecodeParams<__half, __half, int32_t>;
    P params;
    params.q = (__half*)q.data_ptr<at::Half>();
    params.tq_kv = tq_kv;
    params.o = (__half*)o.data_ptr<at::Half>();
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.padded_batch_size = batch_size;
    params.num_qo_heads = num_qo_heads;
    params.q_stride_n = num_qo_heads * padded_dim;
    params.q_stride_h = padded_dim;
    params.window_left = -1;
    params.logits_soft_cap = 0;
    params.sm_scale = sm_scale;
    params.rope_rcp_scale = 0;
    params.rope_rcp_theta = 0;
    params.partition_kv = false;

    auto ri = torch::arange(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kti = torch::zeros(batch_size, torch::dtype(torch::kInt32).device(q.device()));
    auto kcs = torch::zeros(1, torch::dtype(torch::kInt32).device(q.device()));
    params.request_indices = ri.data_ptr<int32_t>();
    params.kv_tile_indices = kti.data_ptr<int32_t>();
    params.kv_chunk_size_ptr = kcs.data_ptr<int32_t>();
    params.block_valid_mask = nullptr;
    params.o_indptr = nullptr;

    auto stream = c10::cuda::getCurrentCUDAStream();
    int bdy = num_qo_heads / num_kv_heads;
    constexpr int VEC = 8;

    // Compute optimal bdz: maximize within 1024 threads, cap at 64
    // Cap at 768 threads to avoid register pressure (v4 uses ~64 regs/thread,
    // A100 has 65536 regs/SM, 768*64=49152 leaves headroom)
    auto compute_bdz = [](int bdx, int bdy) -> int {
        int max_bdz = 768 / (bdx * bdy);
        return (max_bdz > 64) ? 64 : (max_bdz < 1 ? 1 : max_bdz);
    };

    // Dispatch on padded_dim → bdx, then bdy, then bdz
    if (padded_dim <= 64) {
        constexpr int BDX = 64 / VEC;  // 8
        int bdz = compute_bdz(BDX, bdy);
        switch (bdy) {
            case 1: dispatch_bdz<64, BDX, 1>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 2: dispatch_bdz<64, BDX, 2>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 4: dispatch_bdz<64, BDX, 4>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 8: dispatch_bdz<64, BDX, 8>(params, batch_size, num_kv_heads, bdz, stream); break;
            default: TORCH_CHECK(false, "Unsupported GQA ratio ", bdy, " for head_dim=64");
        }
    } else if (padded_dim <= 128) {
        constexpr int BDX = 128 / VEC;  // 16
        int bdz = compute_bdz(BDX, bdy);
        switch (bdy) {
            case 1: dispatch_bdz<128, BDX, 1>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 2: dispatch_bdz<128, BDX, 2>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 3: dispatch_bdz<128, BDX, 3>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 4: dispatch_bdz<128, BDX, 4>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 5: dispatch_bdz<128, BDX, 5>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 6: dispatch_bdz<128, BDX, 6>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 7: dispatch_bdz<128, BDX, 7>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 8: dispatch_bdz<128, BDX, 8>(params, batch_size, num_kv_heads, bdz, stream); break;
            default: TORCH_CHECK(false, "Unsupported GQA ratio ", bdy, " for head_dim=128");
        }
    } else if (padded_dim <= 256) {
        constexpr int BDX = 256 / VEC;  // 32
        int bdz = compute_bdz(BDX, bdy);
        switch (bdy) {
            case 1: dispatch_bdz<256, BDX, 1>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 2: dispatch_bdz<256, BDX, 2>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 4: dispatch_bdz<256, BDX, 4>(params, batch_size, num_kv_heads, bdz, stream); break;
            case 8: dispatch_bdz<256, BDX, 8>(params, batch_size, num_kv_heads, bdz, stream); break;
            default: TORCH_CHECK(false, "Unsupported GQA ratio ", bdy, " for head_dim=256");
        }
    } else {
        TORCH_CHECK(false, "Unsupported padded_dim ", padded_dim);
    }

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v4", &turboquant_decode_v4,
          "TurboQuant v4 fused decode with inline dequant (generalized dispatch)");
}
