"""HYP-019: INT4/INT8 tensor core QK dot product benchmark.

Compares three QK dot product approaches:
  1. Scalar FMA (PyTorch reference — codebook dequant + fp32 dot)
  2. INT8 tensor core via Triton tl.dot (unpack 4-bit to INT8, then MMA)
  3. FP16 tensor core via Triton tl.dot (codebook dequant to fp16, then MMA)

The Triton INT8 path tests the HYP-019 hypothesis: using integer tensor cores
for the QK dot product by operating on raw 4-bit indices instead of dequantized
float values.

Mathematical basis (uniform quantization):
  For uniform codebook: centroid[i] = scale * (i - offset)
  sum(q * centroid[k_idx]) = scale * (sum(q * k_idx) - offset * sum(q))
  The integer matmul gives sum(q_int * k_idx), and post-matmul scaling corrects.

For Lloyd-Max codebook: this is approximate because centroids are non-uniform.
The INT8 dot product uses raw index arithmetic, not true centroid values.

Run:
  uv run python tests/bench_int4_tc.py
"""

import math
import torch
import triton
import triton.language as tl

DEVICE = "cuda"

# Config matching Qwen3-1.7B
HEAD_DIM = 128
PADDED_DIM = 128   # next_power_of_2(128) = 128
NUM_KV_HEADS = 8
NUM_QO_HEADS = 16

# Lloyd-Max 4-bit codebook centroids for N(0,1)
CODEBOOK_4BIT = [
    -2.7326368, -2.0693470, -1.6180345, -1.2562491,
    -0.9423684, -0.6567591, -0.3880823, -0.1284154,
     0.1284154,  0.3880823,  0.6567591,  0.9423684,
     1.2562491,  1.6180345,  2.0693470,  2.7326368,
]

BOUNDARIES_4BIT = [
    (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0 for i in range(15)
]


# ────────────────────────────────────────────────────────────────────
# Data preparation
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prepare_data(seq_len: int):
    """Generate Q (fp16/fp32) and K (4-bit packed + norms) for benchmarking.

    Returns:
        q_fp32: [1, HEAD_DIM] float32 query
        k_packed: [seq_len, HEAD_DIM//2] uint8 nibble-packed codebook indices
        k_norms: [seq_len] float32 L2 norms
        k_indices: [seq_len, HEAD_DIM] uint8 raw codebook indices (unpacked)
        k_deq: [seq_len, HEAD_DIM] float32 dequantized K (for reference)
    """
    codebook_scale = 1.0 / math.sqrt(PADDED_DIM)
    codebook = torch.tensor(CODEBOOK_4BIT, dtype=torch.float32, device=DEVICE)
    boundaries = torch.tensor(
        [b * codebook_scale for b in BOUNDARIES_4BIT],
        dtype=torch.float32, device=DEVICE,
    )

    torch.manual_seed(42)

    # Generate random Q and K
    q_fp32 = torch.randn(1, HEAD_DIM, device=DEVICE, dtype=torch.float32)
    k_raw = torch.randn(seq_len, HEAD_DIM, device=DEVICE, dtype=torch.float32)

    # Compute K norms and normalize
    k_norms = k_raw.norm(dim=-1).clamp(min=1e-8)  # [seq_len]
    k_normalized = k_raw / k_norms.unsqueeze(-1)    # [seq_len, HEAD_DIM]

    # Quantize K: find codebook indices via bucketize
    k_indices = torch.bucketize(k_normalized, boundaries).to(torch.uint8)  # [seq_len, HEAD_DIM]

    # Dequantize K for reference: codebook_scale * codebook[idx] * norm
    k_deq = (codebook[k_indices.long()] * codebook_scale) * k_norms.unsqueeze(-1)

    # Nibble-pack: high nibble = even dim, low nibble = odd dim
    even = k_indices[:, 0::2]   # [seq_len, HEAD_DIM//2]
    odd = k_indices[:, 1::2]
    k_packed = ((even << 4) | odd).to(torch.uint8)  # [seq_len, HEAD_DIM//2]

    return q_fp32, k_packed, k_norms, k_indices, k_deq


# ────────────────────────────────────────────────────────────────────
# Approach 1: Scalar FMA reference (PyTorch, matches v4 codebook lookup)
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def scalar_qk(q_fp32, k_deq):
    """Reference QK: fp32 dot product with dequantized K.

    Args:
        q_fp32: [1, HEAD_DIM] float32
        k_deq: [seq_len, HEAD_DIM] float32

    Returns:
        scores: [seq_len] float32
    """
    return (k_deq @ q_fp32.squeeze(0))  # [seq_len]


# ────────────────────────────────────────────────────────────────────
# Approach 2: INT8 tensor core QK via Triton
# ────────────────────────────────────────────────────────────────────

@triton.jit
def _int8_tc_qk_kernel(
    Q_int8,         # [1, HEAD_DIM] int8 (uniformly quantized Q)
    K_int8,         # [seq_len, HEAD_DIM] int8 (indices shifted to signed)
    output,         # [seq_len] int32 (raw dot products)
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """INT8 tensor core QK dot product.

    Computes raw integer dot product: sum(q_int8[d] * k_int8[d]) for each token.
    Post-matmul scaling converts to true QK score.
    """
    pid = tl.program_id(0)
    seq_start = pid * BLOCK_SEQ

    seq_offsets = tl.arange(0, BLOCK_SEQ)
    seq_mask = (seq_start + seq_offsets) < seq_len

    # Load Q: [1, HEAD_DIM] int8 -> broadcast to [BLOCK_SEQ, HEAD_DIM] for tl.dot
    # tl.dot requires [M, K] x [K, N] -> [M, N]
    # We want: Q [1, HEAD_DIM] x K^T [HEAD_DIM, BLOCK_SEQ] -> scores [1, BLOCK_SEQ]
    # But tl.dot needs both matrices in 2D tile form.

    # Strategy: load K tile [BLOCK_SEQ, HEAD_DIM], compute QK via tl.dot
    # Q: [HEAD_DIM] broadcast -> we'll reshape to [1, HEAD_DIM]
    # K: [BLOCK_SEQ, HEAD_DIM] -> K^T = [HEAD_DIM, BLOCK_SEQ]

    # Load Q row
    dim_offsets = tl.arange(0, HEAD_DIM)
    q_row = tl.load(Q_int8 + dim_offsets)  # [HEAD_DIM] int8

    # Load K tile: [BLOCK_SEQ, HEAD_DIM]
    k_ptrs = K_int8 + (seq_start + seq_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :]
    k_tile = tl.load(k_ptrs, mask=seq_mask[:, None], other=0)  # [BLOCK_SEQ, HEAD_DIM] int8

    # Compute QK dot products via matrix multiply:
    # scores = K_tile @ Q^T = [BLOCK_SEQ, HEAD_DIM] x [HEAD_DIM, 1] -> [BLOCK_SEQ, 1]
    # Using tl.dot: need [BLOCK_SEQ, HEAD_DIM] x [HEAD_DIM, 1]
    # But tl.dot requires N >= 16 for the second dim.
    #
    # Instead: use element-wise multiply + sum
    # This still uses tensor cores when the shapes are compatible.
    #
    # For proper tl.dot usage, we'd need to batch multiple queries.
    # For single-query decode, use the reduction approach:
    q_broadcast = q_row[None, :]  # [1, HEAD_DIM]
    products = (k_tile.to(tl.int32) * q_broadcast.to(tl.int32))  # [BLOCK_SEQ, HEAD_DIM]
    scores = tl.sum(products, axis=1)  # [BLOCK_SEQ]

    # Store raw integer scores
    out_ptrs = output + seq_start + seq_offsets
    tl.store(out_ptrs, scores, mask=seq_mask)


@torch.no_grad()
def int8_tc_qk(q_fp32, k_indices, k_norms, seq_len):
    """INT8 tensor core QK using Triton.

    Quantizes Q to INT8 uniformly, converts K indices to signed INT8,
    then computes raw integer dot product. Post-matmul scaling corrects.

    Args:
        q_fp32: [1, HEAD_DIM] float32
        k_indices: [seq_len, HEAD_DIM] uint8 codebook indices
        k_norms: [seq_len] float32
        seq_len: int

    Returns:
        scores: [seq_len] float32
    """
    codebook_scale = 1.0 / math.sqrt(PADDED_DIM)

    # Quantize Q to INT8: uniform quantization
    q_absmax = q_fp32.abs().max().item()
    q_scale = q_absmax / 7.0  # map to [-7, 7] (INT4 range, but stored in INT8)
    q_inv_scale = 1.0 / q_scale if q_scale > 0 else 0.0

    q_int8 = (q_fp32 * q_inv_scale).round().clamp(-8, 7).to(torch.int8)  # [1, HEAD_DIM]

    # Convert K indices to signed: index - 8 (maps [0,15] to [-8,7])
    k_int8 = (k_indices.to(torch.int16) - 8).to(torch.int8)  # [seq_len, HEAD_DIM]

    # Allocate output (raw integer dot products)
    raw_output = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)

    BLOCK_SEQ = 128
    grid = ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ,)

    _int8_tc_qk_kernel[grid](
        q_int8.contiguous().view(-1),
        k_int8.contiguous(),
        raw_output,
        seq_len,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SEQ=BLOCK_SEQ,
    )

    # Post-matmul scaling:
    # True QK = q_scale * codebook_scale * k_norm * raw_dot
    # (approximate for Lloyd-Max codebook)
    scores = q_scale * codebook_scale * k_norms * raw_output.float()

    return scores


# ────────────────────────────────────────────────────────────────────
# Approach 3: FP16 tensor core QK via Triton (dequant then fp16 matmul)
# ────────────────────────────────────────────────────────────────────

@triton.jit
def _fp16_tc_qk_kernel(
    Q_fp16,         # [1, HEAD_DIM] fp16
    K_deq_fp16,     # [seq_len, HEAD_DIM] fp16 (dequantized K)
    output,         # [seq_len] fp32 (dot products)
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """FP16 tensor core QK dot product (dequantized K)."""
    pid = tl.program_id(0)
    seq_start = pid * BLOCK_SEQ

    seq_offsets = tl.arange(0, BLOCK_SEQ)
    seq_mask = (seq_start + seq_offsets) < seq_len
    dim_offsets = tl.arange(0, HEAD_DIM)

    # Load Q
    q_row = tl.load(Q_fp16 + dim_offsets).to(tl.float32)

    # Load K tile
    k_ptrs = K_deq_fp16 + (seq_start + seq_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :]
    k_tile = tl.load(k_ptrs, mask=seq_mask[:, None], other=0.0).to(tl.float32)

    # Dot product
    products = k_tile * q_row[None, :]
    scores = tl.sum(products, axis=1)

    out_ptrs = output + seq_start + seq_offsets
    tl.store(out_ptrs, scores, mask=seq_mask)


@torch.no_grad()
def fp16_tc_qk(q_fp32, k_deq, seq_len):
    """FP16 QK using Triton (for comparison with INT8 approach).

    This represents the BitDecoding approach: dequant to fp16, then tensor core.
    """
    q_fp16 = q_fp32.half().contiguous().view(-1)
    k_deq_fp16 = k_deq.half().contiguous()

    output = torch.zeros(seq_len, dtype=torch.float32, device=DEVICE)

    BLOCK_SEQ = 128
    grid = ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ,)

    _fp16_tc_qk_kernel[grid](
        q_fp16, k_deq_fp16, output, seq_len,
        HEAD_DIM=HEAD_DIM, BLOCK_SEQ=BLOCK_SEQ,
    )

    return output


# ────────────────────────────────────────────────────────────────────
# Approach 4: Triton tl.dot with INT8 (batch-style, uses tensor cores)
# ────────────────────────────────────────────────────────────────────

@triton.jit
def _int8_dot_qk_kernel(
    Q_int8,         # [BLOCK_M, HEAD_DIM] int8 (Q replicated to BLOCK_M rows)
    K_int8,         # [seq_len, HEAD_DIM] int8
    output,         # [BLOCK_M, seq_len] int32
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 tl.dot-based QK for tensor core utilization.

    Pads Q to BLOCK_M rows (only row 0 is real) and uses tl.dot
    to trigger INT8 MMA instructions (m16n8k32 on A100).
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    # Accumulator: [BLOCK_M, BLOCK_N] int32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = (n_start + n_offsets) < seq_len

    for k_start in range(0, HEAD_DIM, BLOCK_K):
        k_offsets = tl.arange(0, BLOCK_K)

        # Load A (Q): [BLOCK_M, BLOCK_K]
        a_ptrs = Q_int8 + m_offsets[:, None] * HEAD_DIM + (k_start + k_offsets[None, :])
        a = tl.load(a_ptrs)  # [BLOCK_M, BLOCK_K] int8

        # Load B (K^T): [BLOCK_K, BLOCK_N]
        # K is [seq_len, HEAD_DIM] stored row-major.
        # K^T[k, n] = K[n, k], so:
        b_ptrs = K_int8 + (n_start + n_offsets[None, :]) * HEAD_DIM + (k_start + k_offsets[:, None])
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0)  # [BLOCK_K, BLOCK_N] int8

        # MMA: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)

    # Store row 0 of accumulator (the real query's scores)
    out_ptrs = output + n_start + n_offsets
    tl.store(out_ptrs, acc[0, :], mask=n_mask)


@torch.no_grad()
def int8_dot_qk(q_fp32, k_indices, k_norms, seq_len):
    """INT8 tensor core QK using Triton tl.dot.

    Pads Q to a matrix (16 rows) and uses tl.dot to trigger INT8 MMA.
    This should use actual tensor cores (mma.m16n8k32.s8.s8.s32).
    """
    codebook_scale = 1.0 / math.sqrt(PADDED_DIM)

    # Quantize Q
    q_absmax = q_fp32.abs().max().item()
    q_scale = q_absmax / 7.0
    q_inv_scale = 1.0 / q_scale if q_scale > 0 else 0.0
    q_int8_row = (q_fp32 * q_inv_scale).round().clamp(-8, 7).to(torch.int8)  # [1, HEAD_DIM]

    # Replicate Q to BLOCK_M rows (pad with zeros)
    BLOCK_M = 16
    q_padded = torch.zeros(BLOCK_M, HEAD_DIM, dtype=torch.int8, device=DEVICE)
    q_padded[0, :] = q_int8_row.squeeze(0)

    # Convert K indices to signed INT8
    k_int8 = (k_indices.to(torch.int16) - 8).to(torch.int8)  # [seq_len, HEAD_DIM]

    # Output
    raw_output = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)

    BLOCK_N = 16
    BLOCK_K = 32  # INT8 MMA processes 32 K-elements per instruction
    grid = ((seq_len + BLOCK_N - 1) // BLOCK_N,)

    _int8_dot_qk_kernel[grid](
        q_padded.contiguous(),
        k_int8.contiguous(),
        raw_output,
        seq_len,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Post-matmul scaling
    scores = q_scale * codebook_scale * k_norms * raw_output.float()
    return scores


# ────────────────────────────────────────────────────────────────────
# Benchmark harness
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_one(fn, *args, warmup=10, iters=100):
    """Time a function using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # microseconds


@torch.no_grad()
def cosine_similarity(a, b):
    """Cosine similarity between two 1D tensors."""
    a = a.float()
    b = b.float()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


@torch.no_grad()
def run_seq_len(seq_len, warmup=10, iters=100):
    """Run all approaches for one sequence length."""
    q_fp32, k_packed, k_norms, k_indices, k_deq = prepare_data(seq_len)

    # Reference: scalar FMA
    ref_scores = scalar_qk(q_fp32, k_deq)

    # INT8 elementwise (no true tl.dot)
    int8_scores = int8_tc_qk(q_fp32, k_indices, k_norms, seq_len)

    # INT8 tl.dot (true tensor core MMA)
    int8_dot_scores = int8_dot_qk(q_fp32, k_indices, k_norms, seq_len)

    # FP16 dequant + Triton
    fp16_scores = fp16_tc_qk(q_fp32, k_deq, seq_len)

    # Correctness
    cos_int8 = cosine_similarity(int8_scores, ref_scores).item()
    cos_int8_dot = cosine_similarity(int8_dot_scores, ref_scores).item()
    cos_fp16 = cosine_similarity(fp16_scores, ref_scores).item()

    # Timing
    # Scalar: use PyTorch matmul as proxy
    t_scalar = benchmark_one(scalar_qk, q_fp32, k_deq, warmup=warmup, iters=iters)
    t_int8 = benchmark_one(int8_tc_qk, q_fp32, k_indices, k_norms, seq_len,
                           warmup=warmup, iters=iters)
    t_int8_dot = benchmark_one(int8_dot_qk, q_fp32, k_indices, k_norms, seq_len,
                               warmup=warmup, iters=iters)
    t_fp16 = benchmark_one(fp16_tc_qk, q_fp32, k_deq, seq_len,
                           warmup=warmup, iters=iters)

    return {
        "seq_len": seq_len,
        "scalar_us": t_scalar,
        "int8_us": t_int8,
        "int8_dot_us": t_int8_dot,
        "fp16_us": t_fp16,
        "cos_int8": cos_int8,
        "cos_int8_dot": cos_int8_dot,
        "cos_fp16": cos_fp16,
    }


def main():
    print("=" * 90)
    print("HYP-019: INT4/INT8 Tensor Core QK Benchmark (Triton)")
    print(f"Config: HEAD_DIM={HEAD_DIM}, PADDED_DIM={PADDED_DIM}")
    print("=" * 90)

    # Check GPU
    assert torch.cuda.is_available(), "CUDA required"
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (SM {prop.major}.{prop.minor})")
    print()

    print("Approaches:")
    print("  1. Scalar FMA:    PyTorch matmul with codebook-dequantized K (reference)")
    print("  2. INT8 elemwise: Triton element-wise int32 multiply + sum")
    print("  3. INT8 tl.dot:   Triton tl.dot with INT8 inputs (tensor core MMA)")
    print("  4. FP16 Triton:   Triton with fp16 dequantized K (BitDecoding approach)")
    print()
    print("Note: INT8 approaches use raw index arithmetic, not codebook lookup.")
    print("      Lower cosine = quantization error from index-based dot product.")
    print("      This error is fundamental and would require switching from Lloyd-Max")
    print("      to uniform quantization to eliminate.")
    print()

    header = (
        f"{'seq':>6}  "
        f"{'Scalar(us)':>11}  "
        f"{'INT8-ew(us)':>12}  "
        f"{'INT8-dot(us)':>13}  "
        f"{'FP16(us)':>10}  "
        f"{'INT8-dot spdup':>14}  "
        f"{'cos(INT8-ew)':>12}  "
        f"{'cos(INT8-dot)':>13}  "
        f"{'cos(FP16)':>10}"
    )
    print(header)
    print("-" * len(header))

    for seq_len in [128, 512, 1024, 4096]:
        r = run_seq_len(seq_len, warmup=10, iters=100)
        speedup_int8_dot = r["scalar_us"] / r["int8_dot_us"] if r["int8_dot_us"] > 0 else 0
        print(
            f"{r['seq_len']:>6}  "
            f"{r['scalar_us']:>11.1f}  "
            f"{r['int8_us']:>12.1f}  "
            f"{r['int8_dot_us']:>13.1f}  "
            f"{r['fp16_us']:>10.1f}  "
            f"{speedup_int8_dot:>13.2f}x  "
            f"{r['cos_int8']:>12.6f}  "
            f"{r['cos_int8_dot']:>13.6f}  "
            f"{r['cos_fp16']:>10.6f}"
        )

    print()
    print("Analysis:")
    print("  - 'Scalar' is the reference (codebook dequant + fp32 matmul)")
    print("  - 'INT8-dot' uses tl.dot which maps to INT8 tensor cores on A100")
    print("  - If INT8-dot speedup > 1x AND cos > 0.99, HYP-019 is viable")
    print("  - If cos < 0.95, Lloyd-Max codebook is too non-uniform for index arithmetic")
    print("    and uniform quantization would be required (higher MSE but exact math)")
    print()
    print("Theoretical throughput (A100):")
    print("  Scalar FMA:        ~20 TFLOPS (CUDA core FP32)")
    print("  FP16 tensor core:  312 TFLOPS")
    print("  INT8 tensor core:  624 TOPS")
    print("  INT4 tensor core:  1248 TOPS (native PTX, not available via Triton)")


if __name__ == "__main__":
    main()
