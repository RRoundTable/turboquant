# TurboQuant

Near-optimal KV cache quantization for LLM inference (arXiv:2504.19874).

## Build & Test

```bash
pip install -e ".[dev]"        # Install with dev dependencies
pytest tests/ -v               # Run all tests
python tests/test_algorithm.py # Run standalone validation
```

## Architecture

Read `docs/ARCHITECTURE.md` for module boundaries and import rules.

**Dependency direction:** integrations → kv_cache → quantizer → {codebook, hadamard, qjl}

## Governance

- Read `docs/GOAL.md` before starting any task
- Read `docs/ROADMAP.md` to know what to work on (only "Now" items)
- Read `docs/SPEC.md` for behavioral requirements
- ADRs are stored as git commits with `adr/*` tags
- Spec records are stored as git commits with `spec/*` tags

## Code Standards

- No `TODO`, `FIXME`, `HACK`, `XXX`, or `WORKAROUND` in committed code
- All quantization ops run under `@torch.no_grad()`
- Tensor operations must support arbitrary batch dimensions (`[..., d]` pattern)
- Integration modules never import from each other
