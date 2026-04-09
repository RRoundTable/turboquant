"""Benchmark: CUDA fused write kernel vs Python vectorized quantize vs FP16 memcpy.

Measures prefill quantize latency at various token counts.
Config: Qwen3-1.7B (28 layers, 8 KV heads, head_dim=128).

Three paths compared:
  1. FP16 memcpy baseline — raw CUDA memcpy of fp16 KV, no quantization
  2. Python vectorized — torch.bucketize + pack, runs on GPU but Python loop overhead
  3. CUDA write kernel — fused normalize + bucketize + pack in a single kernel launch

Also verifies correctness: CUDA output vs Python output must match (bit-exact on indices).
"""

import math
import os
import time
from pathlib import Path

import torch

DEVICE = "cuda"

# Qwen3-1.7B config
NUM_LAYERS = 28
HEAD_DIM = 128
NUM_KV_HEADS = 8
PADDED_DIM = 128  # next_pow2(128) = 128
DIM_CHUNKS = PADDED_DIM // 64  # 2
QUANT_BYTES_PER_HEAD = DIM_CHUNKS * 32  # 64

# Codebook boundaries for 4-bit N(0,1) Lloyd-Max
B4 = [-2.4009919259139040, -1.8436907609368955, -1.4371417912897455,
      -1.0993087223329937, -0.7995637494657970, -0.5224207238480534,
      -0.2582488430577470,  0.0000000000000000,  0.2582488430577470,
       0.5224207238480534,  0.7995637494657970,  1.0993087223329937,
       1.4371417912897455,  1.8436907609368955,  2.4009919259139040]

_CSRC_DIR = Path(os.environ.get("TURBOQUANT_CSRC",
    os.path.expanduser("~/workdir/turboquant/csrc")))
_INCLUDE_DIR = _CSRC_DIR / "include"


def _build_write_kernel():
    """JIT-compile the fused quantize-write kernel."""
    from torch.utils.cpp_extension import load

    binding_src = _CSRC_DIR / "src" / "quantize_write_binding.cu"
    return load(
        name="turboquant_quantize_write",
        sources=[str(binding_src)],
        extra_include_paths=[str(_INCLUDE_DIR)],
        extra_cuda_cflags=[
            "-std=c++17", "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        verbose=False,
    )


def _pack_4bit(indices):
    """Pack pairs of 4-bit indices into bytes: high nibble = even, low nibble = odd."""
    return ((indices[..., 0::2] << 4) | (indices[..., 1::2] & 0x0F)).to(torch.uint8)


@torch.no_grad()
def python_vectorized_quantize(kv_in):
    """Python vectorized quantize: L2 normalize + bucketize + pack.

    Input:  kv_in [num_tokens, num_heads, head_dim] fp16
    Output: (quant [num_tokens, num_heads, quant_bytes], norms [num_tokens, num_heads, dim_chunks])
    """
    scale = 1.0 / math.sqrt(PADDED_DIM)
    b4_scaled = torch.tensor([b * scale for b in B4], device=DEVICE, dtype=torch.float32)

    xf = kv_in.float()
    norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, H, 1]
    normalized = xf / norms
    norms_fp16 = norms.squeeze(-1).to(torch.float16)  # [T, H]

    num_tokens = kv_in.shape[0]
    num_heads = kv_in.shape[1]

    quant_out = torch.zeros(num_tokens, num_heads, QUANT_BYTES_PER_HEAD,
                            dtype=torch.uint8, device=DEVICE)
    norms_out = torch.zeros(num_tokens, num_heads, DIM_CHUNKS,
                            dtype=torch.float16, device=DEVICE)

    for chunk in range(DIM_CHUNKS):
        ds = chunk * 64
        chunk_data = normalized[..., ds:ds + 64]
        indices = torch.bucketize(chunk_data.contiguous(), b4_scaled).to(torch.uint8)
        packed = _pack_4bit(indices)  # [T, H, 32]
        bo = chunk * 32
        quant_out[..., bo:bo + 32] = packed
        norms_out[..., chunk] = norms_fp16

    return quant_out, norms_out


def benchmark_fn(fn, warmup=10, repeats=100):
    """Benchmark a function, return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]  # median


def verify_correctness(mod):
    """Compare CUDA kernel output vs Python reference (bit-exact on indices)."""
    print("=== Correctness Verification ===")

    for num_tokens in [1, 16, 128, 1024]:
        torch.manual_seed(42)
        kv = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM,
                         device=DEVICE, dtype=torch.float16)

        # Python reference
        py_quant, py_norms = python_vectorized_quantize(kv)

        # CUDA kernel
        cuda_quant, cuda_norms = mod.quantize_write(kv, HEAD_DIM, PADDED_DIM)

        # Compare quant bytes (should be bit-exact)
        quant_match = torch.equal(py_quant, cuda_quant)
        quant_diff = (py_quant != cuda_quant).sum().item()

        # Compare norms (allow 1 ULP fp16 difference)
        norm_diff_abs = (py_norms.float() - cuda_norms.float()).abs()
        # fp16 ULP at typical norm values (~1.0) is ~0.001
        norm_ok = (norm_diff_abs < 0.002).all().item()

        status = "PASS" if quant_match and norm_ok else "FAIL"
        print(f"  tokens={num_tokens:5d}: quant {'exact' if quant_match else f'{quant_diff} diffs'}, "
              f"norms {'ok' if norm_ok else f'max_diff={norm_diff_abs.max():.6f}'} -> {status}")

    print()


def run_benchmarks(mod):
    """Run latency benchmarks at various token counts."""
    print("=== Write Kernel Benchmark ===")
    print(f"Config: head_dim={HEAD_DIM}, num_kv_heads={NUM_KV_HEADS}, "
          f"padded_dim={PADDED_DIM}, dim_chunks={DIM_CHUNKS}")
    print(f"{'tokens':>8s}  {'memcpy_us':>10s}  {'python_us':>10s}  {'cuda_us':>10s}  "
          f"{'py/memcpy':>10s}  {'cuda/memcpy':>12s}  {'py/cuda':>8s}")
    print("-" * 85)

    for num_tokens in [32, 128, 512, 1024, 2048]:
        torch.manual_seed(num_tokens)
        kv = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM,
                         device=DEVICE, dtype=torch.float16)

        # Destination for memcpy baseline
        dst = torch.empty_like(kv)

        # 1. FP16 memcpy baseline
        def memcpy_fn():
            dst.copy_(kv)
        memcpy_us = benchmark_fn(memcpy_fn)

        # 2. Python vectorized quantize
        def python_fn():
            python_vectorized_quantize(kv)
        python_us = benchmark_fn(python_fn)

        # 3. CUDA write kernel
        def cuda_fn():
            mod.quantize_write(kv, HEAD_DIM, PADDED_DIM)
        cuda_us = benchmark_fn(cuda_fn)

        py_over_memcpy = python_us / memcpy_us
        cuda_over_memcpy = cuda_us / memcpy_us
        py_over_cuda = python_us / cuda_us

        print(f"{num_tokens:8d}  {memcpy_us:10.1f}  {python_us:10.1f}  {cuda_us:10.1f}  "
              f"{py_over_memcpy:10.1f}x  {cuda_over_memcpy:12.2f}x  {py_over_cuda:8.1f}x")

    print()

    # Extrapolate full model prefill (28 layers, K+V)
    print("=== Full Model Prefill Estimate (28 layers, K+V) ===")
    print(f"{'tokens':>8s}  {'memcpy_us':>12s}  {'python_us':>12s}  {'cuda_us':>12s}")
    print("-" * 55)

    for num_tokens in [512, 1024, 2048]:
        torch.manual_seed(num_tokens)
        kv = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM,
                         device=DEVICE, dtype=torch.float16)
        dst = torch.empty_like(kv)

        def memcpy_fn():
            dst.copy_(kv)
        memcpy_us = benchmark_fn(memcpy_fn)

        def python_fn():
            python_vectorized_quantize(kv)
        python_us = benchmark_fn(python_fn)

        def cuda_fn():
            mod.quantize_write(kv, HEAD_DIM, PADDED_DIM)
        cuda_us = benchmark_fn(cuda_fn)

        # 28 layers * 2 (K+V)
        factor = NUM_LAYERS * 2
        print(f"{num_tokens:8d}  {memcpy_us * factor:12.0f}  "
              f"{python_us * factor:12.0f}  {cuda_us * factor:12.0f}")


if __name__ == "__main__":
    print("Building CUDA write kernel...")
    mod = _build_write_kernel()
    print("Build complete.\n")

    verify_correctness(mod)
    run_benchmarks(mod)
