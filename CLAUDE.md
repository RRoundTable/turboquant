# TurboQuant

Near-optimal KV cache quantization for LLM inference (arXiv:2504.19874).

## Environment

- **Forge cluster**: 4 nodes, 32x A100-SXM4-40GB (8 per node), team quota 8 GPUs
- **DGX Spark** (legacy): `ssh -q mlsys-dgx-spark` (NVIDIA GB10, aarch64)
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

## Forge GPU Workflow

Use Forge for all GPU work. Two modes:

### Notebook (interactive) — for iterative compile/debug

Use when you need repeated compile-test cycles (e.g., CUDA kernel development).

```bash
# Check available GPUs
forge quota my

# Create SSH notebook with 1 GPU (max 24h)
forge notebook create --name tq-debug --gpu 1 --type ssh --shared-nfs --duration 4

# Run commands inside notebook
forge notebook exec <id> -c "cd /workspace && git clone <repo> && cd turboquant && pip install -e '.[dev]'"
forge notebook exec <id> -c "cd /workspace/turboquant && pytest tests/ -v"

# Get SSH connection info (for manual access)
forge notebook ssh <id>

# When done: commit results, then clean up
forge notebook delete <id>
```

### Job (batch) — for benchmarks and parallel runs

Use when you need reproducible runs, parameter sweeps, or multiple configs in parallel.
Can build custom Docker images with all dependencies baked in.

```bash
# Build image with dependencies (once)
forge image build --name tq-kernel --pip "torch,flashinfer,pytest" --build-timeout 4 --build-memory 32Gi

# Submit job (prefer --entrypoint-file for complex commands)
cat > /tmp/entry.sh <<'SCRIPT'
cd /workspace/turboquant && pytest tests/ -v
SCRIPT
forge job dry-run --name tq-test --entrypoint-file /tmp/entry.sh --gpu 1 --image tq-kernel --shared-nfs
forge job submit  --name tq-test --entrypoint-file /tmp/entry.sh --gpu 1 --image tq-kernel --shared-nfs

# Monitor
forge job logs <id> --follow
forge job get <id>

# Cancel if needed
forge job cancel <id>
```

### Key rules

- Always `forge quota my` before creating notebooks/jobs
- Always `forge job dry-run` before `forge job submit`
- Always `forge notebook delete` after interactive work is done
- Shared NFS is at `/workspace/shared/` inside containers
- Max 8 GPUs per node, max 24h for notebooks, max 168h for jobs
- Use `--shared-nfs` for data and code that needs to persist

### Legacy: DGX Spark (SSH)

```bash
# From local machine (fallback if Forge is unavailable)
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
