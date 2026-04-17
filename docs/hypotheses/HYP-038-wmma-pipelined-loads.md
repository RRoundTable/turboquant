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

## Status: rejected at probe gate (nvcuda::wmma API limitation)

## Results (Forge A100 notebook, 2026-04-17)

Extended the phase probe with a `5h_interleave_PIPELINED` variant exactly as
proposed: double-buffered (a0,b0) and (a1,b1) fragments with loads for tile
N+1 issued before the mma_sync for tile N.

```
5g interleave N=1 (baseline)  1043 cycles  (740 ns)
5h interleave PIPELINED       1040 cycles  (738 ns)  ← same, within noise
```

**Zero speedup from software-pipelined loads.** Probe gate rejected before
touching production.

## Analysis

`nvcuda::wmma::load_matrix_sync` is a **synchronous** call — it blocks the
warp until the smem load completes. The PTX it lowers to (`ldmatrix.sync`)
has no async variant in the WMMA path. Issuing the load earlier in source
code doesn't let it overlap with the preceding mma_sync, because:

1. The PTX `ldmatrix.sync` has a fence before it returns.
2. The compiler serializes the load behind any prior dependent instruction.
3. Even if the load is independent, the warp scheduler won't issue more
   than one memory op at a time without async semantics.

To get real overlap on Ampere WMMA, we'd need:
- `ldmatrix.async` (available in PTX but NOT exposed by `nvcuda::wmma`)
- `cp.async.bulk.tensor` (SM90+, Ampere has only cp.async.ca)
- Drop to raw PTX `mma.sync` + manual fragment register allocation + explicit
  scoreboard management

All of which is a large kernel rewrite: hundreds of lines of inline PTX,
handling register allocation by hand, losing the `nvcuda::wmma` portability.

## Prediction verdicts

| Prediction | Target | Result | Verdict |
|-----------|--------|--------|---------|
| Pipelined 5h ≤ 0.85 × baseline 5g (probe) | ≤ 890 cycles | 1040 cycles (0% win) | ✗ rejected |
| End-to-end seq=4096 | 77 → ~55 μs | not tested | — |

## The common thread across HYP-036, HYP-037, HYP-038 rejections

All three aimed at WMMA_QK (41% of kernel), all three failed:

| hyp | target | why it failed |
|-----|--------|---------------|
| HYP-036 | warp-butterfly softmax | Shuffle reductions slower than unrolled scalar when all lanes broadcast-read same data. |
| HYP-037 | parallel c_frag | Real bottleneck is load→mma dep, not c_frag chain. |
| HYP-038 | pipelined loads | `nvcuda::wmma::load_matrix_sync` is synchronous, can't pipeline. |

**Root cause: the nvcuda::wmma API's abstraction level is too high for the
optimizations that would work on this kernel.** load_matrix_sync + mma_sync
are black boxes; we can't expose their internal pipelining to the scheduler.

## Decision

Stop pushing on WMMA_QK via the high-level API. Possible next moves, in
order of effort:

1. **Accept current state** (within 1.47× of FlashInfer at seq=4096, beating
   FI at seq=512). Memory savings remain 3.8×. Ship.
2. **Raw PTX rewrite**: ~weeks of work, uncertain payoff, loses portability.
3. **Investigate INT4 tensor cores** (HYP-019 was rejected long ago but
   perhaps worth revisiting with the current profile data).
4. **Just stop here**: this is a natural stopping point for the v5_paged
   optimization effort.

Recommending option 4 for this session. The WMMA_QK bottleneck is
addressable only through a full PTX rewrite, which is a different class of
effort than the single-file HYPs we've been running.

## References

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
