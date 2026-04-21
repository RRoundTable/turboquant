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

### Model cache on persistent disk (HuggingFace models)

Do NOT re-download large HuggingFace models (Qwen3-8B ≈ 16 GB) into
`/workspace/shared/hf_cache` or into the container's ephemeral disk on
every job. Use a **dedicated persistent disk** so the cache survives
job exit and shared-NFS cleanup cycles.

One-time setup:

```bash
# Size: 100 GiB holds Qwen3-1.7B + 8B + a couple of comparable models.
# Bump if you start pulling Qwen3-32B or similar.
forge disk create --name tq-models --size 100
```

Standard job pattern (reuses the cache across every run):

```bash
forge job submit --name tq-hyp-XXX --gpu 1 \
    --disk-mount tq-models:/mnt/models \
    --shared-nfs \
    --entrypoint-file /tmp/entry.sh
```

Inside the entrypoint:

```bash
export HF_HOME=/mnt/models/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"
# First job downloads Qwen3-8B; every subsequent job reads from disk.
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-8B')"
```

Rules:

- `HF_HOME` lives on the disk mount, not `/workspace/shared/` or `/tmp`.
- If a job fails mid-download, `HF_HOME` may have partial files —
  `rm -rf $HF_HOME/hub/<broken-model>` before retrying.
- Disk mounts are team-scoped — don't store credentials or one-off
  experiment artifacts here; those go on `--shared-nfs` or a job-local
  path. `tq-models` is **read-heavy, write-rare**.

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

**Decompose before dispatching.** Don't throw one monolithic job at the cluster — break the problem into independent sub-problems and run them in parallel.

1. **Profile first** (see "GPU Profiling" below). A single nsys/ncu run tells you which phase is the bottleneck. Without it, parallel work just parallelizes guessing.
2. **Split along an independent axis.** Good axes: seq length, batch size, kernel variant, layer index, tile size, head count. Bad axes: anything where sub-results depend on each other.
3. **One sub-problem per job.** Each job owns one config, writes its own log/artifact under `/workspace/shared/`, and exits. No shared mutable state between jobs.
4. **Join at the end.** A final step (local or a tiny job) reads all per-job artifacts and aggregates — this is the only serial step.

Example: "why is decode slow?" → profile → decompose into `{quantize, hadamard, dequant, attention}` phase benchmarks → submit 4 parallel jobs → aggregate → attack the worst phase.

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

**Monitor until terminal.** After `forge job submit`, Claude must stay attached until every job reaches a terminal state (`Succeeded` or `Failed`) — never submit-and-forget. Catch the exact moment each job finishes and react immediately. Watch jobs as a group — don't tail one log at a time.

```bash
# Snapshot: all jobs, states, GPUs used
forge job list
forge job list --name-prefix tq-bench-      # filter this sweep

# Per-job detail — state, start time, entrypoint, exit code
forge job get <id>

# Follow a job until it exits — use run_in_background=true so the harness
# notifies on terminal state instead of blocking the conversation
forge job logs <id> --follow
forge job logs <id> --tail 200              # last 200 lines, non-blocking

# Aggregate: one status line per job of the sweep
for id in $(forge job list --name-prefix tq-bench- -o json | jq -r '.[].id'); do
  printf "%s  %-20s  %s\n" "$id" "$(forge job get $id -o json | jq -r .status)" \
    "$(forge job get $id -o json | jq -r .name)"
done
```

**Monitoring checklist after submit:**
1. `forge job list` — confirm all N jobs are `Pending`/`Running`, not `Failed` at launch (image pull, quota, mount errors surface here).
2. `forge job logs <id> --tail 50` on one job after ~30s — verify the entrypoint actually started (catches silent `entry.sh` bugs before all N jobs burn time).
3. For each job, run `forge job logs <id> --follow` with `run_in_background=true` — the harness notifies on terminal state. Do NOT sleep-poll in a loop.
4. The moment a job **fails**, read its log tail before anything else and diagnose — a failed sweep entry that sits unread wastes the whole fan-out.
5. The moment a job **succeeds**, fetch its artifact from `/workspace/shared/` and start folding it into the aggregation step — don't wait for the slowest job to finish before analyzing the fast ones.
6. If a job stays `Pending` after quota is clearly available, investigate — don't assume it will schedule.
7. For long runs (>15 min) without a stream, use the `loop` skill to poll on an interval instead of blocking.

**Fail-fast rule:** if the first finished job crashed with a bug that applies to all configs (wrong import, missing env var, OOM at batch=1), cancel the remaining siblings immediately — don't let the whole sweep burn quota reproducing the same failure.

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

### Remote profiling (Forge)

Forge `--security-profile profiling-debug` unlocks GPU perf counters
(ncu, CUPTI). Without it, containers get `ERR_NVGPUCTRPERM`. nsys
works without the profile.

```bash
# Notebook with ncu access
forge notebook create --name ncu-debug --gpu 1 --type ssh \
    --shared-nfs --duration 4 --security-profile profiling-debug

# Batch job with ncu access
forge job submit --name ncu-job --gpu 1 --shared-nfs \
    --security-profile profiling-debug --entrypoint-file /tmp/profile.sh
```

**Profiling plan (use in order):**

```bash
# ── Tier 1: nsys (no security profile needed) ──────────────────────
# Q: Is the kernel the bottleneck, or Python/launch overhead?
nsys profile --stats=true -t cuda,nvtx -o /tmp/trace python bench.py
nsys stats /tmp/trace.nsys-rep

# ── Tier 2: ncu (needs --security-profile profiling-debug) ────────
# Q: Why is the kernel slow — compute, memory, or latency-bound?

# Step 1: SpeedOfLight — classify bottleneck
ncu --section SpeedOfLight --section Occupancy \
    --kernel-name "TurboQuant" -o profile.ncu-rep python bench.py

# Step 2: Warp stall reasons — what are warps waiting on?
ncu --section WarpStateStatistics --kernel-name "TurboQuant" python bench.py
#   long_scoreboard  → HBM latency → more ILP / occupancy
#   short_scoreboard → smem bank conflicts → add padding
#   math_pipe_throttle → compute-bound → use tensor cores
#   wait             → sync overhead → reduce __syncthreads()

# Step 3: Memory analysis — L2 hit rate, coalescing, bandwidth
ncu --section MemoryWorkloadAnalysis --kernel-name "TurboQuant" python bench.py

# Step 4: Roofline chart
ncu --section SpeedOfLight_RooflineChart --kernel-name "TurboQuant" \
    -o roofline.ncu-rep python bench.py

# Step 5: Source-level hotspot (compile with -lineinfo)
ncu --section SourceCounters --kernel-name "TurboQuant" \
    -o source.ncu-rep python bench.py

# Step 6: Compare TQ vs FlashInfer in same run
ncu --section SpeedOfLight --section WarpStateStatistics \
    --kernel-name regex:"TurboQuant|BatchDecode" \
    -o compare.ncu-rep python bench_both.py

# ── Other tools (always available) ─────────────────────────────────
# SASS disassembly (no GPU needed)
cuobjdump --dump-sass /path/to/compiled.so

# Per-phase cycle counts (instrument kernel with clock64())

# Coarse SM utilization
nvidia-smi dmon -s u -d 1
```

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
