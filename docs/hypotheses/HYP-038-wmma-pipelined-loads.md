# HYP-038: Software-pipelined WMMA loads

## Hypothesis

Phase probe isolated the actual WMMA_QK bottleneck: the **load→mma data
dependency**. Numbers from the interleaved probe (load_matrix_sync + mma_sync
mirroring real kernel):

```
5a load_only (16 loads, no mma)      163 cyc  (116 ns)   — loads alone
5b mma_only N=1 (preloaded frags)    193 cyc  (137 ns)   — mma alone
5g interleave N=1 (real kernel)     1044 cyc  (740 ns)   — the actual phase
```

Real kernel is 1044 cycles but load+mma in isolation sum to 356. The **688
cycle gap is stall time**: each mma_sync waits for fresh a_frag/b_frag to
arrive from smem before it can issue. load_matrix_sync → ldmatrix has ~90
cycles latency; 8 serial kt iterations × ~86 cycles stall = 688 cycles.

Fix: classical software pipelining. Issue the *next* tile's loads while the
*current* tile's mma runs:

```
// Prologue: preload tile 0
load a[0]; load b[0];
for (kt = 0; kt < k_tiles - 1; kt++) {
    // Issue next tile's loads early
    load a[(kt+1) % 2]; load b[(kt+1) % 2];
    // Compute current tile while next loads are in flight
    mma(c, a[kt % 2], b[kt % 2], c);
}
// Epilogue: last mma
mma(c, a[(k_tiles-1) % 2], b[(k_tiles-1) % 2], c);
```

2 pairs of a/b fragments double-buffered. Load for tile N+1 overlaps with
mma for tile N. If load latency < mma latency, both hide completely.

## Prediction

A100, same rig as prior probes (bs=1, kv_heads=8, bdy=4, head_dim=128):

**Phase 5 cycle reduction:**
- Baseline interleave N=1: 1044 cycles (what we have)
- Expected pipelined: **~400-500 cycles**
  - Prologue: ~90 cycles (1 pair of loads)
  - Steady state: max(load_latency, mma_latency) × 7 tiles ≈ 90 × 7 = 630 cycles (if load-limited)
  - Epilogue: 193 cycles (final mma without next load to hide under)
  - Total: ~900 cycles worst case. More likely loads+mmas overlap better → ~500 cycles.

**End-to-end speedup at seq=4096 (chunk=128):**
- Phase 5 saves: 1044 → 500 = 544 cycles = 386 ns/tile
- 128 tokens / tile_n=16 = 8 tiles per chunk
- 32 splits, each split processes 8 tiles → 256 Phase 5 invocations per call
- Assuming this scales linearly with seq: ~386 ns × 8 tiles × per-split savings
- Rough estimate: **76 μs → ~55-60 μs** at seq=4096

**Correctness:** fp32 accumulations in same order (just issuing loads earlier).
cos ≥ 0.9999 expected.

**Risk: register pressure.** 2× fragment pairs = 32 fp16 regs (~1 reg/lane).
Marginal. Watch `-Xptxas -v` for spills.

## Method

### 1. Probe-first discipline

Before touching production kernel, extend the phase probe with a new
`5h_interleave_pipelined_N=1` variant that explicitly double-buffers the
fragments in an unrolled loop. If the probe shows speedup on the real
interleaved pattern (not mma-only), proceed to production. If flat,
reject before touching the kernel.

```cpp
// Probe variant: pipelined load+mma, N=1 c_frag
::nvcuda::wmma::fragment<matrix_a,...> a0, a1;
::nvcuda::wmma::fragment<matrix_b,...> b0, b1;
::nvcuda::wmma::fragment<accumulator,...> c;
::nvcuda::wmma::fill_fragment(c, 0.0f);

// Prologue
::nvcuda::wmma::load_matrix_sync(a0, q_smem, head_dim);
::nvcuda::wmma::load_matrix_sync(b0, kv_fp16, head_dim);

#pragma unroll
for (uint32_t kt = 0; kt < k_tiles - 1; kt++) {
    // Issue next tile loads
    if (kt % 2 == 0) {
        ::nvcuda::wmma::load_matrix_sync(a1, q_smem + (kt+1)*16, head_dim);
        ::nvcuda::wmma::load_matrix_sync(b1, kv_fp16 + (kt+1)*16, head_dim);
        ::nvcuda::wmma::mma_sync(c, a0, b0, c);
    } else {
        ::nvcuda::wmma::load_matrix_sync(a0, q_smem + (kt+1)*16, head_dim);
        ::nvcuda::wmma::load_matrix_sync(b0, kv_fp16 + (kt+1)*16, head_dim);
        ::nvcuda::wmma::mma_sync(c, a1, b1, c);
    }
}
// Epilogue: last tile's mma (odd/even branch)
if ((k_tiles - 1) % 2 == 0) ::nvcuda::wmma::mma_sync(c, a0, b0, c);
else                         ::nvcuda::wmma::mma_sync(c, a1, b1, c);
```

Simpler unroll (since k_tiles=8 for head_dim=128 is even):
```cpp
load a0, b0 (tile 0)
load a1, b1 (tile 1); mma c += a0*b0
load a0, b0 (tile 2); mma c += a1*b1
load a1, b1 (tile 3); mma c += a0*b0
... (4 iterations of 2 tiles each)
mma c += a1*b1 (tile 7)  // epilogue
```

### 2. Gate decision on probe

If probe `5h_pipelined` cycles < `5g_interleave` by ≥15% (target ~500 vs
1044), proceed to implementation. Otherwise reject and document.

### 3. Production change (if probe passes)

Modify the Phase 3 WMMA QK block in `csrc/include/flashinfer_decode_turboquant_v5_tc.cuh`
with the unrolled pipelined pattern. Same pattern as probe.

### 4. Benchmark

Full sweep on notebook (pod 954bf563 or fresh), seq ∈ {256, 512, 1024, 2048,
4096, 8192, 16384, 32768}. Compare `tq_v5_paged_split_graph` latency pre/post.

### 5. Correctness

`tests/test_v5_graph.py` cosine gate + per-call cosine in bench. Expect
cos ≥ 0.9999 (fp16 accumulation order unchanged, just moved loads earlier).

## Status: pending (probe-gated)

## References

- HYP-031 (v5 tensor-core kernel) — target of modification
- HYP-036 REJECTED (warp softmax) — first lesson: profile after changes
- HYP-037 REJECTED (parallel c_frag) — second lesson: probe the real
  interleaved pattern, not simplified isolates. This hypothesis puts that
  lesson into practice via the probe-first gate.
- Ampere WMMA programming guide: software pipelining is the textbook fix
  for load→mma stalls in WMMA kernels.
- FlashAttention kernel: uses similar pipelining in its outer loop; inner
  mma loop lets the compiler handle WMMA-level scheduling.
- Phase probe data: `results/profile_v5_paged/seq-16384.md` — per-phase
  baseline; Phase 5 = 41% of kernel time at seq=16384.
