# Architecture

## Purpose

TurboQuant KV cache quantization for LLM inference. Four-phase development: memory layout (C++) → write kernel (CUDA) → standalone fused decode kernel (CUDA, uses FlashInfer headers) → vLLM integration (eager simulation, kernel fusion not yet connected).

## Tech Stack

| Layer | Choice |
|-------|--------|
| Tile layout / packing | C++17 |
| Kernels | CUDA (standalone, uses FlashInfer headers for math/types) |
| Algorithm PoC | Python 3.10+, PyTorch >= 2.1.0 |
| Attention fusion | Standalone CUDA kernel (FlashInfer NOT modified) |
| Serving integration | vLLM v0.18.0 |
| Testing | GoogleTest (C++), pytest (Python) |
| Build | CMake (C++), setuptools (Python) |
| Device | CUDA only (DGX Spark, SM121) |

## Directory Structure

```
turboquant/
├── csrc/                           # C++/CUDA (tile layout, kernels)
│   ├── include/turboquant/
│   │   ├── tile.h                  # TurboQuantTile struct (480B, 16-byte aligned)
│   │   ├── pack.h                  # 3-bit / 4-bit bit packing
│   │   ├── quantize.h              # quantize_tile / dequantize_tile
│   │   ├── fp16_utils.h            # FP16 conversion utilities
│   │   └── test_framework.h
│   ├── src/
│   │   ├── pack.cpp                # GGML-compatible packing
│   │   └── quantize.cpp            # Signed quantization (max-abs scaling)
│   ├── tests/
│   │   ├── main.cpp
│   │   ├── test_pack.cpp           # Bit-exact pack/unpack roundtrip
│   │   ├── test_tile.cpp           # Struct size/alignment checks
│   │   ├── test_roundtrip.cpp      # Full quantize → dequantize roundtrip
│   │   └── test_fp16.cpp           # FP16 conversion correctness
│   ├── CMakeLists.txt
│   └── Makefile
├── turboquant/                     # Python package (algorithm PoC)
│   ├── __init__.py
│   ├── quantizer.py                # TurboQuantMSE, TurboQuantProd
│   ├── codebook.py                 # Lloyd-Max codebook
│   ├── hadamard.py                 # FWHT + RandomHadamardRotation
│   └── qjl.py                      # QJL 1-bit transform
├── tests/                          # Python tests
│   ├── test_algorithm.py           # Algorithm correctness (GPU)
│   └── eval_accuracy.py            # Distortion / quality evaluation
├── docs/
│   ├── GOAL.md
│   ├── ROADMAP.md
│   ├── SPEC.md
│   ├── ARCHITECTURE.md
│   └── reference/
│       ├── memory-layout.md        # TurboQuantTile layout specification
│       ├── core-implementation.md  # Algorithm implementation details
│       ├── flashinfer-analysis.md
│       └── vllm-kv-cache-analysis.md
├── pyproject.toml
└── CLAUDE.md
```

## Tile Layout (Phase 1)

`TurboQuantTile`: 16 tokens × 64 dims, 3.5-bit average, 480 bytes, 16-byte aligned.

- 32 outlier dims at 4-bit signed (MSB sign + 3-bit magnitude)
- 32 normal dims at 3-bit signed (MSB sign + 2-bit magnitude)
- FP16 norm per row (per token)
- 4.27× compression vs FP16

See `docs/reference/memory-layout.md` for full specification.

## Module Boundaries

### C++ (`csrc/`)

```
quantize → {pack, tile, fp16_utils}
```

### Python (`turboquant/`)

```
quantizer → {codebook, hadamard, qjl}
```

No cross-language imports. C++ and Python are independently testable. Phase 2+ will add Python/C++ bindings.

## Constraints

- `TurboQuantTile` must be exactly 480 bytes, 16-byte aligned (`static_assert`)
- All CUDA kernels target SM121 (DGX Spark GB10), must use `backend='fa2'` for FlashInfer
- All quantization operations run under `@torch.no_grad()` (inference only)
- All Python tensors created with `device="cuda"` directly
- Random seeds must be deterministic for reproducibility
