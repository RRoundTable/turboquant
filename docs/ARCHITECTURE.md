# Architecture

## Purpose

Proof-of-concept GPU-native TurboQuant core. Once validated, the algorithm will be re-implemented directly inside vLLM.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| ML framework | PyTorch >= 2.1.0 |
| Testing | pytest >= 7.0 |
| Build | setuptools |
| Device | CUDA only (DGX Spark, SM121) |

## Design Decisions

- **Classes** for all modules (Codebook, RandomHadamardRotation, QJL, TurboQuantMSE, TurboQuantProd)
- **GPU-only** — all tensors initialized directly on CUDA, no CPU fallback
- **Bit-packed storage** — 3-bit values packed into bytes for true 5x compression vs fp16
- **Proof of concept** — will be re-implemented inside vLLM, so keep it simple

## Directory Structure

```
turboquant/
├── turboquant/               # Core library
│   ├── __init__.py           # Public API exports
│   ├── quantizer.py          # TurboQuantMSE, TurboQuantProd
│   ├── hadamard.py           # FWHT + RandomHadamardRotation
│   ├── codebook.py           # Lloyd-Max codebook, scalar quantization
│   └── qjl.py                # QJL 1-bit transform
├── tests/
│   └── test_algorithm.py     # Algorithm correctness tests (GPU)
├── pyproject.toml
└── docs/
```

## Module Boundaries

### Dependency Direction

```
quantizer → {codebook, hadamard, qjl}
```

- `codebook`, `hadamard`, `qjl` are leaf modules with no internal imports
- No integration modules — vLLM integration lives in the vLLM repo

## Constraints

- All quantization operations run under `@torch.no_grad()` (inference only)
- Codebook centroids are precomputed constants, not learned parameters
- All tensors created with `device="cuda"` directly
- All tensor operations support arbitrary batch dimensions (`[..., d]` pattern)
- Random seeds must be deterministic for reproducibility
