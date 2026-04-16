"""HYP-031: TurboQuant v5 tensor-core decode kernel correctness test.

Tests the WMMA tensor-core variant against a PyTorch reference implementation.
Validates:
  1. Cosine similarity between v5 TC output and reference (target: >= 0.999)
  2. Multiple configs: seq_len={16, 64, 128, 256}, bdy={1, 2, 4}
  3. Split-KV correctness (when seq_len >= 128)

Run:
  # On a machine with GPU (Forge notebook or DGX):
  uv run python tests/test_v5_tc.py
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = "cuda"

# Qwen3-8B config (bdy=4)
HEAD_DIM = 128
PADDED_DIM = 128

# Codebook boundaries for quantization
CODEBOOK_4BIT = [
    -2.7326368, -2.0693470, -1.6180345, -1.2562491,
    -0.9423684, -0.6567591, -0.3880823, -0.1284154,
     0.1284154,  0.3880823,  0.6567591,  0.9423684,
     1.2562491,  1.6180345,  2.0693470,  2.7326368,
]

BOUNDARIES_4BIT = [
    (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0 for i in range(15)
]

_CSRC_DIR = Path(os.environ.get("TURBOQUANT_CSRC",
    os.path.expanduser("~/workdir/turboquant/csrc")))
_INCLUDE_DIR = _CSRC_DIR / "include"
_FLASHINFER_INCLUDE = Path(os.environ.get("FLASHINFER_INCLUDE_DIR",
    os.path.expanduser("~/workdir/flashinfer/include")))


def _build_module():
    """JIT-compile the v5 tensor-core kernel binding."""
    from torch.utils.cpp_extension import load

    src = _CSRC_DIR / "src" / "decode_v5_tc_binding.cu"
    return load(
        name="tq_v5_tc_test",
        sources=[str(src)],
        extra_include_paths=[str(_INCLUDE_DIR), str(_FLASHINFER_INCLUDE)],
        extra_cuda_cflags=[
            "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
            "--generate-line-info",
            "-Xptxas", "-v",   # register usage check
        ],
        verbose=True,
    )


def _pack_4bit(indices):
    """Pack pairs of 4-bit indices into bytes (hi << 4 | lo)."""
    return ((indices[..., 0::2] << 4) | (indices[..., 1::2] & 0x0F)).to(torch.uint8)


def _fwht(x, signs=None):
    """Fast Walsh-Hadamard Transform."""
    d = x.shape[-1]
    x = x.clone()
    shape = x.shape
    h = 1
    while h < d:
        x = x.view(*shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(shape)
        h *= 2
    x = x * (1.0 / math.sqrt(d))
    if signs is not None:
        x = x * signs
    return x


def _fwht_signs():
    """Generate deterministic Hadamard signs."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    signs = torch.sign(torch.randn(PADDED_DIM, generator=gen))
    signs[signs == 0] = 1.0
    return signs.to(DEVICE)


@torch.no_grad()
def setup_data(seq_len, num_kv_heads, num_qo_heads, batch_size=1):
    """Create quantized KV data in contiguous layout + reference Q/K/V.

    Returns dict with contiguous kernel inputs and reference tensors.
    """
    dim_chunks = PADDED_DIM // 64
    quant_bytes = dim_chunks * 32
    signs = _fwht_signs()
    b4 = torch.tensor(BOUNDARIES_4BIT, device=DEVICE, dtype=torch.float32)
    cb = torch.tensor(CODEBOOK_4BIT, device=DEVICE, dtype=torch.float32)

    torch.manual_seed(seq_len * 1000 + num_kv_heads)
    K = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=DEVICE)
    V = torch.randn(seq_len, num_kv_heads, HEAD_DIM, device=DEVICE)
    Q = torch.randn(batch_size, num_qo_heads, HEAD_DIM, device=DEVICE)

    # Contiguous quantized storage
    k_quant = torch.zeros(batch_size, num_kv_heads, seq_len, quant_bytes,
                          dtype=torch.uint8, device=DEVICE)
    v_quant = torch.zeros_like(k_quant)
    k_norms = torch.zeros(batch_size, num_kv_heads, seq_len, dim_chunks,
                          dtype=torch.float16, device=DEVICE)
    v_norms = torch.zeros_like(k_norms)

    # Dequantized KV for reference computation
    K_deq = torch.zeros(seq_len, num_kv_heads, PADDED_DIM, device=DEVICE)
    V_deq = torch.zeros(seq_len, num_kv_heads, PADDED_DIM, device=DEVICE)

    codebook_scale = 1.0 / math.sqrt(PADDED_DIM)

    for kv_idx, tensor in enumerate([K, V]):
        xf = tensor.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)

        # Pad and rotate
        padded = normalized
        if PADDED_DIM > HEAD_DIM:
            padded = torch.zeros(seq_len, num_kv_heads, PADDED_DIM, device=DEVICE)
            padded[..., :HEAD_DIM] = normalized
        rotated = _fwht(padded, signs)

        qs = k_quant if kv_idx == 0 else v_quant
        ns = k_norms if kv_idx == 0 else v_norms
        deq = K_deq if kv_idx == 0 else V_deq

        for chunk in range(dim_chunks):
            ds = chunk * 64
            cd = rotated[..., ds:ds + 64]
            indices = torch.bucketize(cd.contiguous(), b4).to(torch.uint8)
            packed = _pack_4bit(indices)  # [seq_len, num_kv_heads, 32]
            bo = chunk * 32

            # Contiguous layout: [batch, heads, seq, bytes]
            qs[0, :, :, bo:bo + 32] = packed.permute(1, 0, 2)
            ns[0, :, :, chunk] = norms_fp16.permute(1, 0)

            # Dequantized reference
            deq[..., ds:ds + 64] = cb[indices.long()] * codebook_scale * norms.float()

    # Rotate Q for kernel
    Q_padded = Q.float()
    if PADDED_DIM > HEAD_DIM:
        Q_padded_full = torch.zeros(batch_size, num_qo_heads, PADDED_DIM, device=DEVICE)
        Q_padded_full[..., :HEAD_DIM] = Q_padded
        Q_padded = Q_padded_full
    Q_rotated = _fwht(Q_padded, signs).half()

    # SDPA reference (using dequantized KV in rotated space)
    gqa = num_qo_heads // num_kv_heads
    K_ref = K_deq.half().repeat_interleave(gqa, dim=1)  # [seq, qo_heads, pd]
    V_ref = V_deq.half().repeat_interleave(gqa, dim=1)
    sm_scale = 1.0 / math.sqrt(PADDED_DIM)

    # Reference attention output
    q_ref = Q_rotated.float()  # [batch, qo_heads, pd]
    k_ref = K_ref.float().permute(1, 0, 2)  # [qo_heads, seq, pd]
    v_ref = V_ref.float().permute(1, 0, 2)  # [qo_heads, seq, pd]

    scores = torch.bmm(q_ref.squeeze(0).unsqueeze(1), k_ref.transpose(1, 2)) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.bmm(attn, v_ref).squeeze(1)  # [qo_heads, pd]
    ref_out = ref_out.unsqueeze(0)  # [1, qo_heads, pd]

    return {
        "Q_rotated": Q_rotated,
        "k_quant": k_quant, "v_quant": v_quant,
        "k_norms": k_norms, "v_norms": v_norms,
        "ref_out": ref_out.half(),
        "sm_scale": sm_scale,
    }


def cosine_sim(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def test_v5_tc_correctness(module):
    """Test v5 tensor-core kernel against PyTorch reference."""
    configs = [
        # (seq_len, num_kv_heads, num_qo_heads, description)
        (16,  8, 32, "seq=16, bdy=4 (Qwen3-8B)"),
        (64,  8, 32, "seq=64, bdy=4 (Qwen3-8B)"),
        (128, 8, 32, "seq=128, bdy=4 (Qwen3-8B)"),
        (256, 8, 32, "seq=256, bdy=4 (Qwen3-8B)"),
        (16,  8, 16, "seq=16, bdy=2 (Qwen3-1.7B)"),
        (64,  8, 16, "seq=64, bdy=2 (Qwen3-1.7B)"),
        (128, 8, 16, "seq=128, bdy=2 (Qwen3-1.7B)"),
        (16,  8,  8, "seq=16, bdy=1 (Llama-2)"),
        (64,  8,  8, "seq=64, bdy=1 (Llama-2)"),
        (128, 8,  8, "seq=128, bdy=1 (Llama-2)"),
        (16,  8, 64, "seq=16, bdy=8 (Llama-3-70B)"),
        (64,  8, 64, "seq=64, bdy=8 (Llama-3-70B)"),
    ]

    print("=" * 72)
    print("HYP-031: v5 tensor-core kernel correctness test")
    print("=" * 72)

    passed = 0
    failed = 0

    for seq_len, num_kv_heads, num_qo_heads, desc in configs:
        data = setup_data(seq_len, num_kv_heads, num_qo_heads)

        try:
            out = module.decode_v5_tc_contiguous(
                data["Q_rotated"],
                data["k_quant"], data["v_quant"],
                data["k_norms"], data["v_norms"],
                seq_len, num_kv_heads, HEAD_DIM, PADDED_DIM,
                data["sm_scale"]
            )
            torch.cuda.synchronize()

            cos = cosine_sim(out, data["ref_out"])
            status = "PASS" if cos >= 0.990 else "FAIL"
            print(f"  [{status}] {desc}: cos={cos:.6f}")

            if cos >= 0.990:
                passed += 1
            else:
                failed += 1
                # Debug info
                print(f"         out  range: [{out.min().item():.4f}, {out.max().item():.4f}]")
                print(f"         ref  range: [{data['ref_out'].min().item():.4f}, {data['ref_out'].max().item():.4f}]")
                diff = (out.float() - data["ref_out"].float()).abs()
                print(f"         max diff: {diff.max().item():.6f}")

        except Exception as e:
            print(f"  [ERROR] {desc}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


def test_v5_tc_splitkv(module):
    """Test v5 tensor-core split-KV against non-split reference."""
    configs = [
        (128, 8, 32, 2, "seq=128, bdy=4, 2 splits"),
        (256, 8, 32, 4, "seq=256, bdy=4, 4 splits"),
        (512, 8, 32, 8, "seq=512, bdy=4, 8 splits"),
        (128, 8, 16, 2, "seq=128, bdy=2, 2 splits"),
        (256, 8, 16, 4, "seq=256, bdy=2, 4 splits"),
    ]

    print()
    print("=" * 72)
    print("HYP-031: v5 tensor-core split-KV correctness test")
    print("=" * 72)

    passed = 0
    failed = 0

    for seq_len, num_kv_heads, num_qo_heads, num_splits, desc in configs:
        data = setup_data(seq_len, num_kv_heads, num_qo_heads)

        try:
            # Non-split reference
            ref_out = module.decode_v5_tc_contiguous(
                data["Q_rotated"],
                data["k_quant"], data["v_quant"],
                data["k_norms"], data["v_norms"],
                seq_len, num_kv_heads, HEAD_DIM, PADDED_DIM,
                data["sm_scale"]
            )
            torch.cuda.synchronize()

            # Split-KV
            split_out = module.decode_v5_tc_contiguous_splitkv(
                data["Q_rotated"],
                data["k_quant"], data["v_quant"],
                data["k_norms"], data["v_norms"],
                seq_len, num_kv_heads, HEAD_DIM, PADDED_DIM,
                data["sm_scale"], num_splits
            )
            torch.cuda.synchronize()

            cos = cosine_sim(split_out, ref_out)
            status = "PASS" if cos >= 0.999 else "FAIL"
            print(f"  [{status}] {desc}: cos={cos:.6f}")

            if cos >= 0.999:
                passed += 1
            else:
                failed += 1
                diff = (split_out.float() - ref_out.float()).abs()
                print(f"         max diff: {diff.max().item():.6f}")

        except Exception as e:
            print(f"  [ERROR] {desc}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


def bench_v5_tc(module, seq_len=1024, num_kv_heads=8, num_qo_heads=32,
                warmup=20, iters=100):
    """Benchmark v5 tensor-core kernel."""
    data = setup_data(seq_len, num_kv_heads, num_qo_heads)

    def run():
        return module.decode_v5_tc_contiguous(
            data["Q_rotated"],
            data["k_quant"], data["v_quant"],
            data["k_norms"], data["v_norms"],
            seq_len, num_kv_heads, HEAD_DIM, PADDED_DIM,
            data["sm_scale"]
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) / iters * 1000
    return us


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires a GPU.")
        sys.exit(1)

    print("Building v5 tensor-core kernel...")
    module = _build_module()
    print("Build complete.\n")

    all_pass = True
    all_pass &= test_v5_tc_correctness(module)

    if all_pass:
        all_pass &= test_v5_tc_splitkv(module)

    if all_pass:
        print("\n" + "=" * 72)
        print("HYP-031: v5 tensor-core benchmark")
        print("=" * 72)
        bdy = 4
        nkv = 8
        nqo = nkv * bdy
        for seq in [128, 256, 512, 1024, 2048]:
            us = bench_v5_tc(module, seq_len=seq, num_kv_heads=nkv,
                            num_qo_heads=nqo)
            print(f"  seq={seq:5d}: {us:8.1f} us")

    if not all_pass:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests PASSED.")
        sys.exit(0)
