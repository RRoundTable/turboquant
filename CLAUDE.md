# TurboQuant

Near-optimal KV cache quantization for LLM inference (arXiv:2504.19874).

## Environment

- **Forge cluster**: 4 nodes, 32x A100-SXM4-40GB (8 per node), team quota 8 GPUs
- **Python**: 3.12, managed with `uv`
- **GPU only** — no CPU device support

## Setup

```bash
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

### Persistent disks (staging code/artifacts)

For code, tarballs, or artifacts that the job must read, use a persistent disk
or the shared NFS — not the entrypoint script itself.

```bash
# List / create / delete disks
forge disk list
forge disk create --name tq-staging --size 10          # size in GiB
forge disk delete tq-staging

# Mount into a job or notebook
forge job submit --name foo --gpu 1 --disk-mount tq-staging:/mnt/tq ...
```

**Staging pattern** (use for any input > ~64 KB):

1. Create a small SSH notebook with `--gpu 0 --shared-nfs`.
2. `scp -o ProxyJump=bastion@forge.krafton-ml.net:30022 <local file> root@<notebook-host>:/workspace/shared/<file>` to stage the artifact.
3. `forge notebook delete <id>` immediately.
4. Submit the bench / compute job with `--shared-nfs`; its entrypoint reads `/workspace/shared/<file>`.

Do NOT embed large (>~128 KB) base64 blobs in `--entrypoint-file` — the
entire entrypoint is passed as a single argv and will hit
`exec: argument list too long` (ARG_MAX).

### Key rules

- Prefer `forge job submit` over `forge notebook create` for any non-interactive work. Notebooks burn quota while idle; jobs record the entrypoint and stop on their own. Only use notebooks when you genuinely need an SSH session or a compile-debug loop.
- Always `forge quota my` before creating notebooks/jobs
- Always `forge job dry-run` before `forge job submit`
- Always `forge notebook delete` after interactive work is done
- Shared NFS is at `/workspace/shared/` inside containers
- Max 8 GPUs per node, max 24h for notebooks, max 168h for jobs
- Use `--shared-nfs` for data and code that needs to persist, or `--disk-mount <name>:<path>` for a job-private persistent disk

### Parallel execution (jobs + notebooks)

Team quota is 8 GPUs. Run independent work concurrently — don't serialize.

**Rule of thumb:**
- **Parallel jobs** for independent sweeps (seq lengths, batch sizes, kernel variants, ablations). Each job is self-contained and records its own logs.
- **One notebook + parallel jobs** when you need a compile-debug loop *and* large sweeps at the same time. Keep the notebook on 1 GPU; fan out jobs for the sweep.
- **Never** allocate multiple SSH notebooks for the same task — idle notebooks burn quota.

```bash
# Fan out a sweep: submit N jobs in one shell, each with its own config
for SEQ in 512 1024 2048 4096; do
  cat > /tmp/entry-$SEQ.sh <<SCRIPT
cd /workspace/turboquant && uv run python bench.py --seq $SEQ
SCRIPT
  forge job submit --name tq-bench-$SEQ --entrypoint-file /tmp/entry-$SEQ.sh \
    --gpu 1 --image tq-kernel --shared-nfs
done

# Watch all at once
forge job list
forge job logs <id> --follow   # per job

# Compile-debug in a notebook, benchmark in jobs (in parallel)
forge notebook create --name tq-debug --gpu 1 --type ssh --shared-nfs --duration 4
forge job submit --name tq-sweep-a --entrypoint-file /tmp/a.sh --gpu 1 --image tq-kernel --shared-nfs
forge job submit --name tq-sweep-b --entrypoint-file /tmp/b.sh --gpu 1 --image tq-kernel --shared-nfs
```

**Claude tool usage:** when dispatching multiple independent Forge commands, emit them as parallel tool calls in a single message — don't wait for one `forge job submit` before issuing the next.

**Quota discipline:**
- `forge quota my` before fanning out — confirm headroom for N concurrent GPUs
- Size each job to 1 GPU unless the workload needs more (most benches do not)
- Cancel stragglers promptly (`forge job cancel <id>`) once you have enough data points

## Project Goal

Match FlashInfer's decode latency with TurboQuant's 3.76× memory efficiency.
The kernel must be **as fast as FlashInfer** and **as memory-efficient as TurboQuant**.
Use profiling data to guide every optimization decision — never guess.

## GPU Profiling

Profile before optimizing. Every kernel change must be justified by profiling data.

### Tool hierarchy (use in order)

1. **nsys** (system-level) — is the kernel the bottleneck, or Python/launch overhead?
2. **ncu** (kernel-level) — why is the kernel slow? Compute, memory, or latency-bound?
3. **ncu warp stalls** — what are warps waiting on? This directly tells you what to fix.

### Key commands

```bash
# Step 1: System timeline — confirm kernel is the bottleneck
nsys profile -o trace python bench.py
nsys stats trace.nsys-rep

# Step 2: SpeedOfLight — classify as compute/memory/latency bound
ncu --section SpeedOfLight --section Occupancy --kernel-name "TurboQuant" \
    -o profile.ncu-rep python bench.py

# Step 3: Warp stall reasons — find dominant bottleneck
ncu --section WarpStateStatistics --kernel-name "TurboQuant" python bench.py
# Key stall metrics:
#   long_scoreboard  → waiting for HBM load (need more ILP or prefetch)
#   short_scoreboard → smem bank conflicts (add padding)
#   math_pipe_throttle → compute-bound (use tensor cores)
#   wait             → __syncthreads() imbalance (reduce barriers)

# Step 4: Memory analysis — L2 hit rate, coalescing, bandwidth
ncu --section MemoryWorkloadAnalysis --kernel-name "TurboQuant" python bench.py

# Step 5: Roofline chart — visualize headroom
ncu --section SpeedOfLight_RooflineChart --kernel-name "TurboQuant" \
    -o roofline.ncu-rep python bench.py

# Step 6: Source-level hotspot (compile with -lineinfo)
ncu --section SourceCounters --kernel-name "TurboQuant" -o source.ncu-rep python bench.py
```

### Remote profiling (Forge notebook)

**Forge containers lack GPU perf counter permissions (`ERR_NVGPUCTRPERM`).**
ncu and CUPTI profiler metrics are blocked. Available alternatives:

```bash
# nsys WORKS — kernel timeline, durations, CUDA API calls
NSYS=/opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64/nsys
$NSYS profile --stats=true -t cuda,nvtx -o /tmp/trace python bench.py

# cuobjdump WORKS — SASS instruction disassembly (no GPU needed)
cuobjdump --dump-sass /path/to/compiled.so

# clock64() instrumentation WORKS — per-phase cycle counts inside kernel

# torch.profiler WORKS (timing only, NOT CUPTI metrics)
# CUPTI metrics silently return no data on Forge

# nvidia-smi dmon WORKS — coarse SM utilization at 1s granularity
nvidia-smi dmon -s u -d 1
```

For full ncu profiling, request Forge admin to set
`RmProfilingAdminOnly=0` on cluster nodes.

### Compile flags for profiling

```bash
# Always include for ncu source-level analysis
extra_cuda_cflags=[..., "--generate-line-info"]

# Register/spill check at compile time
extra_cuda_cflags=[..., "-Xptxas", "-v"]
# Look for: "0 bytes spill stores, 0 bytes spill loads" (good)
# Any spill > 0 is a critical performance bug
```

### Profiling decision tree

```
Is the kernel the bottleneck? (nsys)
  No  → fix Python/launch overhead first
  Yes → What does SpeedOfLight say? (ncu)
    High compute%, low memory% → compute-bound → use tensor cores
    Low compute%, high memory% → memory-bound → improve coalescing/L2
    Both low                   → latency-bound → check warp stalls:
      long_scoreboard dominant → HBM latency → increase occupancy/ILP
      wait dominant            → sync overhead → reduce __syncthreads()
      short_scoreboard dominant → smem conflicts → fix bank conflicts
```

### FlashInfer comparison profiling

When comparing TQ kernel vs FlashInfer, profile BOTH with the same ncu metrics.
The delta in warp stall distribution tells you exactly where FlashInfer wins.

```bash
# Profile both kernels in same run
ncu --section SpeedOfLight --section WarpStateStatistics \
    --kernel-name regex:"TurboQuant|BatchDecode" \
    -o compare.ncu-rep python bench_both.py
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

## Experiment-Driven Development

All kernel optimization work follows a hypothesis → experiment → record cycle.

### Workflow

1. **Hypothesize**: Before writing code, write a hypothesis doc at `docs/hypotheses/HYP-NNN-short-name.md`:
   ```markdown
   # HYP-NNN: Short descriptive title

   ## Hypothesis
   What you believe will happen and why.

   ## Prediction
   Specific, measurable outcome (e.g., "v4 will be 2-3× faster than v2 at seq=1024").

   ## Method
   What code changes, what benchmark, what configs.

   ## Status: pending | confirmed | rejected
   ```

2. **Experiment**: Run the experiment in a git worktree (`isolation: worktree`) or Forge notebook. Keep the main branch clean — experimental code lives in the worktree until results are in.

3. **Record**: Update the hypothesis doc with actual results, then set status:
   - **confirmed** — prediction matched. Merge the code to main, commit the hypothesis doc.
   - **rejected** — prediction did not match. Do NOT merge code. Commit the hypothesis doc with the negative result and analysis of why.
   Both outcomes are valuable. Never delete a rejected hypothesis — it prevents re-trying the same idea.

4. **Review history**: Before proposing a new optimization, read ALL existing hypotheses in `docs/hypotheses/`. Check if the idea (or a similar one) was already tried. If so, explain what's different this time.

5. **Search literature**: When stuck or proposing a new direction, search papers related to:
   - KV cache quantization (KIVI, GEAR, KVQuant, QuIP, SqueezeLLM)
   - GPU kernel optimization (FlashAttention, FlashDecoding, PagedAttention)
   - Dequantization on GPU (mixed-precision matmul, lookup table quantization)
   - Tensor core utilization with non-standard data formats

### Rules

- Never skip the hypothesis doc — even for "obvious" improvements
- Every experiment must have a predicted outcome BEFORE running
- Rejected experiments are as important as confirmed ones
- Check `docs/hypotheses/` before proposing any new optimization
- Include paper references when the idea comes from literature

## Code Standards

- No `TODO`, `FIXME`, `HACK`, `XXX`, or `WORKAROUND` in committed code
- All quantization ops run under `@torch.no_grad()`
- Tensor operations must support arbitrary batch dimensions (`[..., d]` pattern)
- Integration modules never import from each other
- GPU only — never initialize tensors on CPU; use `device="cuda"` directly
