# HYP-012: Profile TQ v4 vs FlashAttention to identify dominant stall

## Hypothesis
torch.profiler shows TQ v4 at 90.6μs vs SDPA (FlashAttention split-KV) at 20.0μs
per call at seq=1024, Qwen3-1.7B config on A100. The 4.5× gap must come from one of:
- Warp stalls on HBM loads (paged KV scatter)
- Low FMA throughput (scalar vs tensor cores)
- Sync barrier overhead (4 syncs per tile iteration)

## Profiling results (torch.profiler, A100)

| Kernel | CUDA Time/call | Calls | % Total |
|--------|---------------|-------|---------|
| TurboQuantPagedDecodeKernelV4 | **90.6 μs** | 5 | 76.7% |
| flash_fwd_splitkv_kernel | **13.6 μs** | 5 | 11.5% |
| flash_fwd_splitkv_combine_kernel | 6.4 μs | 5 | 5.4% |
| elementwise_kernel (zeros) | ~2.8 μs | 10 | 4.7% |

SDPA total = 13.6 + 6.4 = **20.0 μs** (split-KV + combine).

## Profiling tool availability on Forge A100 containers

| Tool | Works? | Data obtained |
|------|--------|---------------|
| **nsys** | YES | Kernel durations, NVTX ranges, CUDA API times, memory copies |
| **ncu** (all sections) | **NO** | `ERR_NVGPUCTRPERM` — GPU perf counters blocked |
| **CUPTI metrics** (torch.profiler) | **NO** | API succeeds but returns no data silently |
| **nvidia-smi dmon** | YES | SM/mem util % at 1s granularity (too coarse) |
| **ptxas -v** | YES | Registers (96), spills (0), smem per block |
| **cuobjdump --dump-sass** | YES | Full SASS disassembly (no GPU needed) |
| **torch.profiler** (timing) | YES | Per-kernel CUDA time |
| **clock64() instrumentation** | YES | Per-phase cycle counts inside kernel |

**Root cause**: Forge containers run without `--privileged` and cannot write to
`/proc/driver/nvidia/params/RmProfilingAdminOnly`. This blocks ALL hardware
performance counters (ncu, CUPTI range profiling, warp stall metrics).

**Workarounds available:**
1. SASS disassembly → instruction mix (FMA/LDG/BAR counts) without GPU
2. clock64() → per-phase cycle counts inside kernel, no permissions needed
3. Request Forge admin to set `RmProfilingAdminOnly=0` on cluster nodes
4. Use DGX Spark (may have host-level ncu access)

## nsys results (TQ v4 vs SDPA, seq=1024, Qwen3-1.7B, A100)

| Kernel | Per call | Calls |
|--------|----------|-------|
| TurboQuantPagedDecodeKernelV4 | **90.9 μs** | 15 |
| flash_fwd_splitkv_kernel (SDPA) | **13.6 μs** | 15 |
| flash_fwd_splitkv_combine_kernel | **6.5 μs** | 15 |
| **SDPA total** | **20.1 μs** | — |
| **TQ / SDPA ratio** | **4.5×** | — |

## ptxas results (compile-time)

| Variant | Registers | Spills | Implication |
|---------|-----------|--------|-------------|
| v4 generalized binding | **96** | 0 | High — limits to 1 block/SM |
| v4 test file (bdz=16) | 64 | 0 | Better — 2 blocks/SM possible |

The generalized binding compiles many template instantiations, inflating register
pressure from 64 → 96. This cuts occupancy from ~50% to ~25%.

## Computed analysis (no GPU counters needed)

### SM utilization
- Grid: 8 blocks (1 batch × 8 KV heads)
- A100 SMs: 108
- **Only 7% of SMs active** — 100 SMs completely idle
- FlashAttention uses split-KV → many more blocks → full SM utilization

### Bandwidth utilization
- KV data read: ~1.1 MB per decode
- Min time at 2 TB/s: 0.5 μs
- Actual: 90.6 μs → **0.6% bandwidth utilization**
- Kernel is NOT bandwidth-bound

### Compute utilization
- Total FLOPs: ~17 MFLOP
- Achieved: 0.19 TFLOPS at 90.6 μs
- FP32 peak: 19.5 TFLOPS → **1.0% compute utilization**
- Kernel is NOT compute-bound either

### Conclusion: latency-bound
Both compute and bandwidth utilization are <2%. The kernel is **latency-bound**:
- SM underutilization (92% SMs idle)
- Low occupancy (25% warps per active SM)
- Instruction latency not hidden by warp switching

## Bottleneck hierarchy (updated)

| # | Issue | Impact | Evidence |
|---|-------|--------|----------|
| **1** | **SM underutilization** — 8 blocks / 108 SMs | 92% SMs idle | Grid config: batch×kv_heads |
| **2** | **Low occupancy** — 96 regs → 1 block/SM | 25% warps | ptxas -v |
| **3** | **Page table latency** | 32μs per decode | HYP-008 paged vs contiguous |
| **4** | **Instruction latency** | Hidden by 1+2 | <2% compute+BW utilization |

**#1 fix: Split-KV parallelism** (FlashDecoding) — partition sequence across blocks.
**#2 fix: Reduce registers** (`--maxrregcount=64`) or simplify generalized binding.

## Status: confirmed
Profiling data obtained via nsys + computed analysis. Warp stall data blocked
by container permissions. Key finding: **latency-bound due to SM underutilization
(92% idle) and low occupancy (25%)**, not compute or bandwidth.
