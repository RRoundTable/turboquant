# HYP-035: Delete the gather — make v5 read paged KV directly

## Hypothesis

`decode_v5_from_cache_splitkv_ws` (HYP-034) currently does the work in two
passes: a `gather_paged_to_contiguous` kernel copies paged vLLM KV into a
contiguous `[batch, heads, max_len, ebs]` workspace, then the WMMA compute
kernel (`TurboQuantContiguousDecodeKernelV5TC`) reads that workspace with
simple `base + token_idx * stride` pointer math. The gather exists because
v5 was first implemented against an eager contiguous test harness (HYP-031)
and retrofitted for paged vLLM storage by prepending a copy, not because
the WMMA kernel inherently needs contiguous HBM.

`decode_v4_from_cache` proves paged-native is feasible in production today —
v4's inner loop walks the page table with `page_iter = indptr[b] + t/block_size;
entry = t % block_size; page_idx = indices[page_iter]` and reads quant/norms
directly from `kv_cache[page_idx, entry, head, :]`. It pays ~10–20 cycles of
integer indexing per tile, amortized across 16 bytes of quant data, or
<1 cycle per byte. Tiny vs the dequant + WMMA cost it fronts.

This hypothesis moves v5 to the same pattern: extend its load prolog (phases 1
and 5 of `TurboQuantContiguousDecodeKernelV5TC`) to walk the page table per
WMMA tile, eliminating the gather + contiguous workspace entirely.

## Prediction

A100, same rig as HYP-034 (batch=1, kv_heads=8, num_qo_heads=32, head_dim=128):

- v5-paged-split at seq=4096 drops from **196.8 μs to ~100–130 μs** (35–50%
  reduction), saving:
  - ~15–25 μs from deleting two `gather_paged_to_contiguous` launches
  - ~5–10 μs from deleting five `cudaMemsetAsync` workspace initializations
  - ~3–5 μs from deleting `mark_empty_splits_v5` (no stale partitions to fix —
    kernel only touches real tokens)
  - ~10–20 μs from halving HBM traffic (data crosses HBM once instead of
    gather-write + compute-read)
- v5-paged-split / FlashInfer at seq=4096 ≤ **3.0×** (vs 4.52× today).
- v5-paged-split matches HYP-034's current latency at seq ≤ 512 within noise
  (gather overhead is a smaller absolute share at short seq).
- **Memory:** workspace HBM drops by batch × kv_heads × max_len × ebs for
  quant + norms = ~100 MB at bs=32, seq=4096 (eliminated entirely).
- **Correctness:** cosine(v5-paged, v5-contiguous) = 1.0000 on identical
  paged KV input. The math is unchanged; only the source layout differs.

## Method

### 1. Kernel params: carry the page table

Extend `ContiguousTurboQuantDecodeParams` (or add a parallel
`PagedTurboQuantDecodeParams`) with the fields v4 already uses:

```cpp
// New: paged-source pointers and layout
const uint8_t*  kv_slab;        // kv_cache.select(0, k_or_v) base
const int32_t*  indices;        // [total_pages]  page table
const int32_t*  indptr;         // [batch + 1]
int32_t         block_size;
int32_t         ebs;            // entry byte stride (per-head packed)
int32_t         qbytes;         // quant bytes per token per head
int32_t         dim_chunks;     // norms per token per head
```

Keep the old contiguous fields too so the same kernel template supports both
layouts — select at compile time via a `bool paged` template parameter.

### 2. Load prolog: swap base+stride math for page-walk

Today (contiguous, `decode_v5_tc.cuh:304-311`):
```cpp
const uint8_t* src = params.k_quant;
if (valid) {
    uint32_t token_idx = tile_start_abs + row;
    src = params.k_quant + quant_base
          + (size_t)token_idx * params.quant_stride_token
          + seg_in_row * 16;
}
```

Paged-native replacement (mirrors v4's `paged_kv_turbo_t::get_k_ptr`):
```cpp
const uint8_t* src = params.kv_slab;  // default (invalid)
if (valid) {
    uint32_t token_idx = tile_start_abs + row;
    uint32_t page_iter = params.indptr[b] + token_idx / params.block_size;
    uint32_t entry     = token_idx % params.block_size;
    uint32_t page_idx  = params.indices[page_iter];
    src = params.kv_slab
          + (size_t)page_idx * params.block_size * params.num_kv_heads * params.ebs
          + (size_t)entry * params.num_kv_heads * params.ebs
          + (size_t)head_idx * params.ebs
          + seg_in_row * 16;
}
```

For `page_size = 16 = tile_n` (vLLM default), one page lookup per WMMA tile
covers all 16 tokens — no scatter within a tile. For page_size != tile_n,
each token reads its own page lookup; the extra indexing is still cheap.

Identical change in phase 5 for V, plus in the norm-precompute loops that use
`params.k_norms` / `params.v_norms` (swap for page-walked norm pointers
starting at `kv_slab + ... + qbytes` within the entry).

### 3. Boundary handling (replaces `mark_empty_splits_v5`)

In the paged layout we only walk real tokens, so:
- Each block reads `seq_lens[b]` directly and clamps its inner loop to
  `[chunk_start, min(chunk_end, seq_lens[b]))`.
- Splits entirely past `seq_lens[b]` take the fast path: zero outputs,
  write LSE = -1e30 directly, skip the gather and the WMMA work.
- The separate `mark_empty_splits_v5` kernel is deleted.

### 4. Op signature simplification

New binding `decode_v5_from_cache_paged_splitkv_ws`:

```cpp
torch::Tensor decode_v5_from_cache_paged_splitkv_ws(
    torch::Tensor q, torch::Tensor kv_cache,
    torch::Tensor indices, torch::Tensor indptr, torch::Tensor last_page_len,
    torch::Tensor seq_lens,
    int num_kv_heads, int page_size, int head_dim, int padded_dim,
    float sm_scale, torch::Tensor hadamard_signs, int qbytes, int nbytes,
    torch::Tensor o_ws,                  // [batch, num_qo_heads, padded_dim] fp16
    int max_len,
    torch::Tensor partition_o_ws,         // [batch * num_splits, num_qo_heads, padded_dim]
    torch::Tensor partition_lse_ws,       // [batch * num_splits, num_qo_heads]
    torch::Tensor request_indices, torch::Tensor kv_tile_indices,
    torch::Tensor split_indptr, torch::Tensor kv_chunk_size_t,
    int num_splits
);
```

Removed from the HYP-034 signature:
- `k_quant_ws`, `v_quant_ws`, `k_norms_ws`, `v_norms_ws` (gather workspaces)

### 5. vLLM backend wiring

`_get_v5_ws` stops allocating gather workspaces (saves ~100 MB at batch=32,
seq=4096). Signature of the cached dict shrinks to `{o, partition_o,
partition_lse, request_indices, kv_tile_indices, split_indptr, kv_chunk_size}`.
The forward call switches from `decode_v5_from_cache_splitkv_ws` to
`decode_v5_from_cache_paged_splitkv_ws`; non-split path gets a corresponding
`decode_v5_from_cache_paged_ws` for symmetry.

The old `_ws` + `_splitkv_ws` ops stay exported for now (backward compat +
A/B benchmarking) but the backend exclusively uses the paged-native path.

### 6. Benchmark

Extend `tests/bench_v5_graph.py` with a sixth variant `tq_v5_paged_split_graph`
so the sweep produces side-by-side: FP16 SDPA, FlashInfer, v4-graph,
v5-nosplit (HYP-033), v5-contig-split (HYP-034), v5-paged-split (this hyp).
Same Forge fan-out: 5 jobs across seq ∈ {256, 512, 1024, 2048, 4096}.

Correctness: cosine(paged, contiguous) = 1.0000 at every seq on identical
input. Any deviation means the page-walk math disagrees with the gather's
linearization — bug, not a numerical tolerance issue.

## Status: pending

## References

- HYP-018 (contiguous + split-KV) — confirmed. Established ~48 μs @ seq=1024
  target for paged + WMMA + split-KV (v4 scalar-FMA number; v5 WMMA should
  match or beat).
- HYP-029 (`decode_v4_from_cache`) — confirmed. Established the paged-native
  read pattern this hypothesis ports into v5. `paged_kv_turbo_t` is the
  reference implementation.
- HYP-031 (tensor-core dequant v5 kernel) — pending. The WMMA kernel whose
  load prolog we're modifying.
- HYP-033 (v5 graph-safety) — confirmed. Workspace + torch.library pattern
  we inherit; this hypothesis simplifies the workspace set.
- HYP-034 (v5 split-KV from cache) — confirmed. The current state; this
  hypothesis extends it by removing the gather.
- HYP-032 (Marlin dequant → fp16 → tensor core) — pending. Independent win
  that attacks a different phase of the kernel (compute, not load). HYP-035
  lands first; HYP-032 builds on top because its per-tile load prolog is
  much simpler when starting from paged HBM directly.
- Phase 13 in `docs/ROADMAP.md` — this hypothesis is the direct continuation
  of 13e (HYP-034), closing the rest of the gap that isn't dequant.
