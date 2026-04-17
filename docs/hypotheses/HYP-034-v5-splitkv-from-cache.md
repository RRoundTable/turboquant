# HYP-034: Port split-KV into v5 `decode_v5_from_cache_ws`

## Hypothesis

HYP-033 made the v5 tensor-core decode kernel graph-capturable but left the
launch grid at `(batch, num_kv_heads)` — only 8 blocks on a 108-SM A100 when
bs=1, kv_heads=8. Each block walks the full seq_len serially, so wall time
grows linearly (134 → 1323 μs as seq_len goes 256 → 4096). FlashInfer stays
flat at ~40 μs across the same sweep because its grid is
`(batch × num_kv_heads × num_splits)` — as seq_len grows, `num_splits` grows,
blocks saturate the SMs, and per-SM work stays ~constant.

Porting split-KV into `decode_v5_from_cache_ws` will flatten the latency curve
the same way. The kernel, combine, and split-index math already exist in the
contiguous path (`decode_v5_tc_contiguous_splitkv`, `SplitKVCombineKernel`);
what's missing is plumbing them through the paged→contiguous gather + the
pre-allocated workspace interface introduced by HYP-033. No new kernel math.

HYP-018 confirmed v4 contiguous+split-KV is flat at ~48 μs across
seq ∈ {128, 1024} and 59 μs at seq=2048. v5 replaces v4's scalar-FMA QK with
WMMA, so v5 split-KV should land at ~ HYP-018 numbers or slightly better.

## Prediction

Same rig as HYP-033 (A100, batch=1, num_kv_heads=8, num_qo_heads=32, bdy=4):

- v5-split latency within **15%** of HYP-018's v4 contiguous+split numbers at
  every seq (WMMA should not regress; may improve slightly at seq ≥ 1024).
- v5-split / v5-nosplit (HYP-033 numbers) **≤ 0.15** at seq=4096 (≥ 6.7×
  speedup from the fan-out alone).
- v5-split / FlashInfer **≤ 2.5×** at seq=4096 (vs current 30.77×).
- Correctness: cosine ≥ 0.9999 vs `decode_v5_from_cache_ws` non-split output
  on the same inputs (combine kernel is numerically equivalent to the
  sequential accumulation — HYP-022's stability analysis carries over).
- Graph-safety: kernel + combine both capturable; no new `.item()` or
  allocation inside the op.

Target table:

| seq  | HYP-033 v5-nosplit | HYP-034 v5-split (target) | FlashInfer |
|------|-------------------:|---------------------------:|-----------:|
|  256 |         134.2 μs   |                   ~130 μs  |     40.8 μs|
| 1024 |         354.5 μs   |                   ~60 μs   |     39.9 μs|
| 2048 |         673.2 μs   |                   ~70 μs   |     39.9 μs|
| 4096 |        1323.2 μs   |                  ~100 μs   |     43.0 μs|

## Method

### 1. C++ op: `decode_v5_from_cache_splitkv_ws`

New binding in `csrc/src/decode_v5_tc_binding.cu`, sitting alongside HYP-033's
`_ws` variant. Reuses the existing gather kernel and
`TurboQuantContiguousDecodeKernelV5TC` — only the grid dim, params, and
post-kernel combine change.

Signature (additions over HYP-033):
```cpp
torch::Tensor decode_v5_from_cache_splitkv_ws(
    /* all HYP-033 args */,
    torch::Tensor partition_o_ws,    // [padded_batch, num_qo_heads, padded_dim] fp32
    torch::Tensor partition_lse_ws,  // [padded_batch, num_qo_heads]             fp32
    torch::Tensor request_indices,   // [padded_batch] int32 (b for each split)
    torch::Tensor kv_tile_indices,   // [padded_batch] int32 (split idx)
    torch::Tensor split_indptr,      // [batch + 1]    int32
    torch::Tensor kv_chunk_size_t,   // [1]            int32
    int num_splits                   // static, baked into captured graph
);
```

The five new tensors are the existing split-KV scratch buffers from
`decode_v5_tc_contiguous_splitkv` (lines 134–149 of the binding) — we just
move them from inside-the-op allocation to caller-provided workspace.

Internal flow:
1. `cudaMemsetAsync` K/V quant/norms workspaces (HYP-033 behavior).
2. Run both gather kernels into K/V workspaces (HYP-033 behavior).
3. Init `partition_lse_ws` to −∞ via `cudaMemsetAsync` (sentinel for combine).
4. Launch `TurboQuantContiguousDecodeKernelV5TC` with grid
   `(padded_batch, num_kv_heads)`, partition_kv=true, reusing the same
   `CP p` except with `partition_o` / `partition_lse` → workspace pointers.
5. Launch `SplitKVCombineKernel<__half>` grid `(batch, num_qo_heads)` to
   reduce partitions into `o_ws`.

### 2. vLLM backend: adaptive num_splits per capture bucket

`turboquant/vllm_backend_fused.py`: `_get_v5_ws` picks `num_splits` at
workspace-allocation time based on `(batch_size, max_pages, num_kv_heads)`:

```python
available_sms = 108  # A100; query via torch.cuda.get_device_properties
target_blocks = available_sms * 2  # fill SMs + some over-subscription
num_splits = min(
    max(1, target_blocks // (batch_size * self.num_kv_heads)),
    # Don't over-split: each split needs >= 1 KV tile (16 tokens).
    max_len // 16,
)
```

`num_splits` is computed from host-visible values (`batch_size`,
`max_pages`, `num_kv_heads`) — no GPU sync. It's cached with the workspace,
so the graph captures a fixed split count per shape bucket. Small-seq
buckets still pick `num_splits=1` (avoids the ~25 μs combine overhead
HYP-018 measured at seq ≤ 256).

### 3. Workspace additions

Added to `_v5_ws_cache[(batch, max_pages)]`:
```
partition_o      : [batch * num_splits, num_qo_heads, padded_dim]  fp32
partition_lse    : [batch * num_splits, num_qo_heads]              fp32
request_indices  : [batch * num_splits]                             int32
kv_tile_indices  : [batch * num_splits]                             int32
split_indptr     : [batch + 1]                                      int32
kv_chunk_size    : [1]                                              int32
```

At bs=1, splits=16, num_qo_heads=32, padded_dim=128:
`partition_o ~= 1 × 16 × 32 × 128 × 4 = 256 KB`, negligible.

Pre-fill `request_indices`, `kv_tile_indices`, `split_indptr`,
`kv_chunk_size` **once** at workspace creation (these are pure functions of
the shape bucket, identical across replays). The fill runs on the Python
side during warmup, outside capture. No per-replay cost.

### 4. Benchmark + verification

Extend `tests/bench_v5_graph.py` with a fifth variant `tq_v5_split_graph`
via `torch.ops.turboquant_v5.decode_v5_from_cache_splitkv_ws`. Keep the
other four variants as-is so the HYP-033 numbers are re-collected on the
same run (sanity against drift).

Correctness gate in `tests/test_v5_graph.py`: add a
`test_v5_split_matches_nosplit_under_graph` case that captures the split
variant, replays 10×, and asserts `cosine(nosplit, split) ≥ 0.9999` at
seq ∈ {256, 1024, 4096}. Bit-exactness is not expected (split-KV reduces
via log-sum-exp; non-split uses a different accumulation order) — hence
cosine, not max_abs.

### 5. Forge sweep

Reuse the HYP-033 5-job pattern: seq ∈ {256, 512, 1024, 2048, 4096}, one
job per seq_len, 1 GPU each, 5 GPUs parallel (48 quota). Same entrypoint
script, same aggregator. Each JSON gains a `tq_v5_split_graph` entry;
`scripts/aggregate_bench.py` already tolerates missing keys.

## Status: pending

## References

- HYP-018 (contiguous + split-KV) — **confirmed**. Flat ~48 μs at seq=128–1024
  for v4. This hypothesis ports that wiring into the paged `_from_cache_ws`
  path.
- HYP-022 (fused combine) — **rejected** for the full-fuse version, but the
  separate `SplitKVCombineKernel` it introduced is what we re-use here.
- HYP-029 (decode-read-from-cache) — **confirmed**. Established the paged
  `_from_cache` call pattern that HYP-033 + this hypothesis extend.
- HYP-031 (tensor-core dequant) — **pending**. v5's WMMA kernel whose grid we
  are about to widen.
- HYP-033 (v5 graph-safety) — **confirmed (engineering)**. Workspace pattern
  and library registration this hypothesis builds on.
- Phase 13b in `docs/ROADMAP.md` calls out the same gap for v4:
  > "Port split-KV into `decode_v4_from_cache`. Target: 2–3× speedup at seq ≥ 1024."
  v5 inherits the identical limitation; this fixes it for v5.
