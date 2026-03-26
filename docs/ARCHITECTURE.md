# Architecture

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| ML framework | PyTorch >= 2.1.0 |
| Serving | vLLM >= 0.6.0 |
| Attention kernels | FlashInfer |
| Performance extensions | Rust or C++ (if needed for custom kernels) |
| Testing | pytest >= 7.0 |
| Build | setuptools |

## System Overview

```
┌─────────────────────────────────────────────┐
│              User / Application              │
├──────────────┬──────────────┬───────────────┤
│  vLLM        │  SGLang      │  HuggingFace  │
│  Integration │  Integration │  (direct)     │
├──────────────┴──────────────┴───────────────┤
│           FlashInfer Kernels                 │
│        (quantized KV attention)              │
├─────────────────────────────────────────────┤
│              TurboQuant Core                 │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Quantizer│ │ KV Cache │ │  Codebook   │ │
│  │ MSE/Prod │ │          │ │  + Hadamard │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
turboquant/
├── turboquant/               # Core library
│   ├── __init__.py           # Public API exports
│   ├── quantizer.py          # TurboQuantMSE, TurboQuantProd
│   ├── kv_cache.py           # TurboQuantCache (HF-compatible)
│   ├── hadamard.py           # Fast Walsh-Hadamard transform, random rotation
│   ├── codebook.py           # Lloyd-Max codebook, scalar quantization
│   ├── qjl.py                # Quantized Johnson-Lindenstrauss (1-bit)
│   ├── vllm_integration.py   # vLLM attention patching
│   └── sglang_integration.py # SGLang attention patching
├── tests/
│   ├── test_algorithm.py     # Algorithm correctness tests
│   └── bench_kv_cache.py     # Performance benchmarks
├── examples/
│   ├── vllm_example.py
│   └── sglang_example.py
├── pyproject.toml
└── docs/
```

## Module Boundaries

### Import Rules

- `turboquant.quantizer` imports from `codebook`, `hadamard`, `qjl` (core primitives)
- `turboquant.kv_cache` imports from `quantizer` only
- `turboquant.vllm_integration` imports from `quantizer` only
- `turboquant.sglang_integration` imports from `quantizer` only
- Integration modules (`vllm_integration`, `sglang_integration`) never import from each other
- `codebook`, `hadamard`, `qjl` are leaf modules with no internal imports

### Dependency Direction

```
integrations (vllm, sglang) → kv_cache → quantizer → {codebook, hadamard, qjl}
```

No reverse dependencies. No circular imports.

## Constraints

- All quantization operations must work with `@torch.no_grad()` (inference only)
- Codebook centroids are precomputed constants, not learned parameters
- Random seeds must be deterministic for reproducibility
- All tensor operations must support arbitrary batch dimensions (`[..., d]` pattern)
