# HYP-016: KernelAgent for systematic kernel optimization

## Hypothesis
KernelAgent (Meta's LLM-guided Triton kernel optimizer) can systematically discover
optimizations we missed through manual experimentation.

## Method
1. Ported v4 CUDA kernel to Triton — 5/5 correctness tests pass (cos=1.0)
2. Created KernelAgent-compatible task files (kernel.py, problem.py, test.py)
3. Patched KernelAgent to handle missing ncu (injected hardcoded bottleneck analysis)
4. Ran greedy strategy on Forge A100

## Results

**Pipeline works end-to-end on Forge:**
- Initial kernel benchmarked: 0.277ms (277μs) — matches our previous measurements
- PyTorch reference baseline: 116.6ms (slow, pure Python dequant+attention)
- PyTorch compile baseline: 5.7ms
- LLM generates optimized Triton kernels (~45s per generation via Claude Sonnet)
- Correctness verification runs (~3.5min per check)

**But: all generated kernels fail correctness (4/4 rounds).**

The LLM rewrites the entire kernel structure instead of making targeted changes.
Without real NCU metrics (warp stalls, cache hits, bandwidth), the LLM doesn't
know WHERE the bottleneck is — it guesses and makes broad structural changes that
break the complex paged-KV addressing and codebook dequant logic.

## Analysis

**Why it fails without ncu:**
- KernelAgent's strength is the NCU → bottleneck → specific fix → verify loop
- Without NCU, the LLM gets only our hardcoded bottleneck description
- The kernel is too complex (paged KV, codebook LUT, online softmax) for the
  LLM to rewrite correctly without iterative feedback from actual profiling

**What would work:**
- Running with ncu access (DGX Spark or Forge with admin permissions)
- Simplifying the kernel to reduce LLM rewrite scope
- Providing more detailed bottleneck information in the fallback

## Forge limitations confirmed
- ncu: BLOCKED (ERR_NVGPUCTRPERM)
- CUPTI metrics: BLOCKED (silently returns no data)
- KernelAgent noop profiler: causes "No analysis available, skipping round"
- Patched fallback: LLM generates but correctness fails

## Status: inconclusive
Pipeline verified end-to-end. LLM optimization blocked by lack of ncu profiling
data on Forge. Need ncu access (DGX Spark or Forge admin) for effective use.
