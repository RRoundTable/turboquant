# TurboQuant

Near-optimal KV cache quantization for LLM inference (arXiv:2504.19874).

## Environment

- **GPU node**: `ssh -q mlsys-dgx-spark` (DGX Spark, NVIDIA GB10, aarch64)
- **Workdir on node**: `~/workdir/turboquant`
- **Python**: 3.12, managed with `uv`
- **GPU only** — no CPU device support

## Setup

```bash
# On DGX Spark node
cd ~/workdir/turboquant
uv venv
uv pip install -e ".[dev]"
uv pip install flashinfer vllm
```

## Build & Test

```bash
uv run pytest tests/ -v               # Run all tests
uv run python tests/test_algorithm.py  # Standalone validation
```

## Remote Workflow

```bash
# From local machine
ssh -q mlsys-dgx-spark "cd ~/workdir/turboquant && uv run pytest tests/ -v"
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
- GPU only — never initialize tensors on CPU; use `device="cuda"` directly
