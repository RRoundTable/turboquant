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

// ---- Hadamard rotation kernel for Q -------------------------------------------
// Apply signs × FWHT × scale to each Q head independently.
// Grid: (batch, num_qo_heads), Block: (32) = one warp per head.
// padded_dim must be 128 → vec_size = 4 per thread.
__global__ void hadamard_rotate_q_kernel(
    __half* q,                          // [batch, num_qo_heads, padded_dim] fp16 (in-place)
    const float* __restrict__ signs,    // [padded_dim] float
    int padded_dim,
    float hadamard_scale                // 1/sqrt(padded_dim)
) {
    constexpr int VEC = 4;  // 128 / 32 threads
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tx = threadIdx.x;

    __half* head_ptr = q + (size_t)batch_idx * gridDim.y * padded_dim
                         + (size_t)head_idx * padded_dim;

    float v[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        int d = tx * VEC + i;
        v[i] = __half2float(head_ptr[d]) * signs[d];
    }

    // Local FWHT stages (within-thread butterfly)
    #pragma unroll
    for (int h = 1; h < VEC; h <<= 1) {
        #pragma unroll
        for (int i = 0; i < VEC; i += 2 * h) {
            #pragma unroll
            for (int k = 0; k < h; k++) {
                float a = v[i + k], b = v[i + k + h];
                v[i + k] = a + b;
                v[i + k + h] = a - b;
            }
        }
    }

    // Cross-thread FWHT stages (warp shuffle butterfly)
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
        #pragma unroll
        for (int i = 0; i < VEC; i++) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, v[i], delta);
            if (tx & delta)
                v[i] = partner - v[i];
            else
                v[i] = v[i] + partner;
        }
    }

    // Scale and store
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        int d = tx * VEC + i;
        head_ptr[d] = __float2half(v[i] * hadamard_scale);
    }
}

// ---- Gather kernel: paged NHD → contiguous HND --------------------------------
// Copies quant bytes + norms from paged kv_cache into flat contiguous buffers
// matching ContiguousTurboQuantDecodeParams layout.
__global__ void gather_paged_to_contiguous(
    const uint8_t* __restrict__ kv_slab,  // [num_blocks, block_size, num_heads, ebs]
    uint8_t* __restrict__ cont_quant,     // [batch, num_heads, max_len, qbytes]
    __half* __restrict__ cont_norms,      // [batch, num_heads, max_len, dim_chunks]
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ indptr,
    const int32_t* __restrict__ seq_lens,
    int block_size, int num_heads, int qbytes, int dim_chunks,
    int ebs, int max_len
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_len = seq_lens[batch_idx];
    int base_page = indptr[batch_idx];

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        int page_iter = base_page + t / block_size;
        int entry = t % block_size;
        int page_idx = indices[page_iter];

        // Source in paged NHD: [page_idx, entry, head_idx, :]
        const uint8_t* src = kv_slab
            + (size_t)page_idx * block_size * num_heads * ebs
            + (size_t)entry * num_heads * ebs
            + (size_t)head_idx * ebs;

        // Dest in contiguous HND: [batch, head, t, :]
        size_t dst_off = (size_t)batch_idx * num_heads * max_len * qbytes
                       + (size_t)head_idx * max_len * qbytes
                       + (size_t)t * qbytes;
        uint8_t* dst = cont_quant + dst_off;

        // Copy quant bytes (vectorized as uint4 = 16 bytes)
        for (int b = 0; b < qbytes; b += 16) {
            *reinterpret_cast<uint4*>(dst + b) =
                *reinterpret_cast<const uint4*>(src + b);
        }

        // Copy norms (dim_chunks fp16 = dim_chunks*2 bytes)
        size_t norm_dst_off = (size_t)batch_idx * num_heads * max_len * dim_chunks
                            + (size_t)head_idx * max_len * dim_chunks
                            + (size_t)t * dim_chunks;
        const __half* norm_src = (const __half*)(src + qbytes);
        for (int c = 0; c < dim_chunks; c++) {
            cont_norms[norm_dst_off + c] = norm_src[c];
        }
    }
}

// ---- decode_v5_from_cache: graph-safe paged decode via gather + v5 contiguous --
torch::Tensor decode_v5_from_cache(
    torch::Tensor q,              // [batch, num_qo_heads, padded_dim] fp16
    torch::Tensor kv_cache,       // [2, num_blocks, block_size, num_heads, ebs] uint8
    torch::Tensor indices,
    torch::Tensor indptr,
    torch::Tensor last_page_len,
    torch::Tensor seq_lens,       // [batch] int32
    int num_kv_heads,
    int page_size,
    int head_dim,
    int padded_dim,
    float sm_scale,
    torch::Tensor hadamard_signs,
    int qbytes,
    int nbytes
) {
    TORCH_CHECK(padded_dim == 128, "v5 from_cache only supports head_dim=128");
    int ebs = static_cast<int>(kv_cache.size(-1));
    int block_size = static_cast<int>(kv_cache.size(2));
    int batch_size = q.size(0);
    int num_qo_heads = q.size(1);
    int bdy = num_qo_heads / num_kv_heads;
    int dim_chunks = padded_dim / 64;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto dev = q.device();

    // Max seq_len for contiguous buffer sizing (use max from seq_lens)
    int max_len = seq_lens.max().item<int>();

    // Zero-init: tokens beyond seq_lens[b] read as zero quant + zero norms,
    // which dequant to 0 → zero QK contribution → no output corruption.
    auto k_quant_c = torch::zeros({batch_size, num_kv_heads, max_len, qbytes},
                                   torch::dtype(torch::kUInt8).device(dev));
    auto v_quant_c = torch::zeros_like(k_quant_c);
    auto k_norms_c = torch::zeros({batch_size, num_kv_heads, max_len, dim_chunks},
                                   torch::dtype(torch::kFloat16).device(dev));
    auto v_norms_c = torch::zeros_like(k_norms_c);

    auto k_half = kv_cache.select(0, 0);
    auto v_half = kv_cache.select(0, 1);

    // Gather K: paged NHD → contiguous HND
    gather_paged_to_contiguous<<<dim3(batch_size, num_kv_heads), 256, 0, stream>>>(
        (uint8_t*)k_half.data_ptr(),
        k_quant_c.data_ptr<uint8_t>(),
        (__half*)k_norms_c.data_ptr<at::Half>(),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        block_size, num_kv_heads, qbytes, dim_chunks, ebs, max_len);

    // Gather V
    gather_paged_to_contiguous<<<dim3(batch_size, num_kv_heads), 256, 0, stream>>>(
        (uint8_t*)v_half.data_ptr(),
        v_quant_c.data_ptr<uint8_t>(),
        (__half*)v_norms_c.data_ptr<at::Half>(),
        indices.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        block_size, num_kv_heads, qbytes, dim_chunks, ebs, max_len);

    // Build params with Hadamard signs — the v5 kernel applies the
    // rotation to Q internally (fused into the Q-load phase).
    auto o = torch::zeros_like(q);

    using CP = ContiguousTurboQuantDecodeParams<__half, __half, int32_t>;
    using V = DefaultAttention<false, false, false, false>;

    CP p(
        k_quant_c.data_ptr<uint8_t>(), v_quant_c.data_ptr<uint8_t>(),
        (__half*)k_norms_c.data_ptr<at::Half>(), (__half*)v_norms_c.data_ptr<at::Half>(),
        batch_size, num_kv_heads, max_len, head_dim, padded_dim,
        (__half*)q.data_ptr<at::Half>(), (__half*)o.data_ptr<at::Half>(),
        nullptr, sm_scale, num_qo_heads
    );
    if (hadamard_signs.numel() > 0) {
        p.hadamard_signs = hadamard_signs.data_ptr<float>();
        p.hadamard_scale = rsqrtf(static_cast<float>(padded_dim));
    }

    #define LAUNCH_V5FC(BDY, NW) do { \
        constexpr uint32_t sm = calc_smem_v5_tc<128, BDY, NW>(); \
        auto err = cudaFuncSetAttribute( \
            TurboQuantContiguousDecodeKernelV5TC<128, BDY, NW, V, CP>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
        TORCH_CHECK(err == cudaSuccess, "smem: ", cudaGetErrorString(err)); \
        TurboQuantContiguousDecodeKernelV5TC<128, BDY, NW, V, CP> \
            <<<dim3(batch_size, num_kv_heads), dim3(32, 1, NW), sm, stream>>>(p); \
    } while(0)

    switch (bdy) {
        case 1: LAUNCH_V5FC(1, 4); break;
        case 2: LAUNCH_V5FC(2, 4); break;
        case 4: LAUNCH_V5FC(4, 4); break;
        case 8: LAUNCH_V5FC(8, 4); break;
        default: TORCH_CHECK(false, "v5 from_cache: unsupported bdy=", bdy);
    }
    #undef LAUNCH_V5FC

    auto cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "v5 from_cache: ", cudaGetErrorString(cuda_err));
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_v5_tc_contiguous", &decode_v5_tc_contiguous,
          "TurboQuant v5 tensor-core contiguous decode (HYP-031)");
    m.def("decode_v5_tc_contiguous_splitkv", &decode_v5_tc_contiguous_splitkv,
          "TurboQuant v5 tensor-core contiguous decode with split-KV (HYP-031)");
    m.def("decode_v5_from_cache", &decode_v5_from_cache,
          "TurboQuant v5 tensor-core decode from paged kv_cache (HYP-031)");
}
