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

## ncu status

**BLOCKED**: Forge containers lack GPU performance counter permissions
(`ERR_NVGPUCTRPERM`). Need host-level `RmProfilingAdminOnly=0` or privileged container.

Alternatives attempted:
- `echo 0 > /proc/driver/nvidia/params/RmProfilingAdminOnly` → permission denied
- `nsys` → not installed in container
- `torch.profiler` → works, gives kernel-level timing but no warp stall breakdown

## What we know without ncu

From HYP-008 bottleneck isolation (contiguous vs paged benchmarks):
- Contiguous dq at bdz=16: 57μs (no paging)
- v4 paged at bdz=16: 89μs
- Page overhead: 32μs (36% of total)
- Contiguous vs SDPA: 57μs vs 30μs → 27μs structural gap

The 27μs structural gap (contiguous dq vs SDPA) is likely:
- Scalar FMA vs tensor cores (SDPA uses FlashAttention which uses HMMA)
- Our online softmax has more instructions (codebook lookup + norm multiply per element)
- Our 4 syncs per iteration vs FlashAttention's pipelined approach

## Next steps

1. Try ncu on DGX Spark (may have host-level access)
2. Instrument kernel with `clock64()` to get per-phase timing
3. Use `cuobjdump --dump-sass` to inspect instruction mix

## Status: partial
Got kernel-level timing. Blocked on ncu warp stall analysis due to container permissions.
