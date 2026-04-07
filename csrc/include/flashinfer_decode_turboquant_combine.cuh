#pragma once
// Split-KV combine kernel: merge partial attention results across splits.
// Uses online softmax merge: same math as sync_state cross-warp merge.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "flashinfer/math.cuh"

namespace flashinfer {

// Combine partial results from split-KV decode.
// Each split produced: tmp_o[split, qo_head, head_dim] and lse[split, qo_head].
// This kernel merges them into: o[batch, qo_head, head_dim].
//
// Grid: (num_requests, num_qo_heads)
// Block: (head_dim,) — one thread per dimension
template <typename DTypeO>
__global__ void SplitKVCombineKernel(
    const float* __restrict__ tmp_o,  // [total_splits_padded, num_qo_heads, head_dim]
    const float* __restrict__ tmp_lse,// [total_splits_padded, num_qo_heads]
    DTypeO* __restrict__ o,           // [num_requests, num_qo_heads, head_dim]
    float* __restrict__ lse,          // [num_requests, num_qo_heads] (optional)
    const int32_t* __restrict__ split_indptr, // [num_requests + 1] — CSR offsets into splits
    int num_qo_heads,
    int head_dim
) {
    int req = blockIdx.x;
    int qo_head = blockIdx.y;
    int d = threadIdx.x;  // dimension index

    if (d >= head_dim) return;

    int split_start = split_indptr[req];
    int split_end = split_indptr[req + 1];
    int num_splits = split_end - split_start;

    if (num_splits == 0) {
        o[(req * num_qo_heads + qo_head) * head_dim + d] = DTypeO(0);
        return;
    }

    // Online softmax merge across splits
    float m_global = -1e30f;
    float d_global = 0.f;
    float o_acc = 0.f;

    for (int s = split_start; s < split_end; s++) {
        float m_s = tmp_lse[s * num_qo_heads + qo_head];
        float o_s = tmp_o[(s * num_qo_heads + qo_head) * head_dim + d];

        if (m_s <= -1e29f) continue;  // empty split

        if (m_global <= -1e29f) {
            // First valid split
            m_global = m_s;
            d_global = 1.0f;
            o_acc = o_s;
            continue;
        }

        float m_new = fmaxf(m_global, m_s);
        float scale_old = expf(m_global - m_new);
        float scale_new = expf(m_s - m_new);

        o_acc = o_acc * scale_old + o_s * scale_new;
        d_global = d_global * scale_old + scale_new;
        m_global = m_new;
    }

    // Normalize
    if (d_global > 0.f) {
        o_acc /= d_global;
    }

    o[(req * num_qo_heads + qo_head) * head_dim + d] = DTypeO(o_acc);
    if (lse != nullptr && d == 0) {
        lse[req * num_qo_heads + qo_head] = m_global + logf(d_global);
    }
}

}  // namespace flashinfer
