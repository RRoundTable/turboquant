# HYP-013: Split-KV parallelism (FlashDecoding) to fill all SMs

## Hypothesis

HYP-012 profiling shows 92% of SMs idle: only 8 grid blocks (batch=1 × 8 KV heads)
for 108 SMs on A100. Both compute (1%) and bandwidth (0.6%) utilization are under 2%.
The kernel is latency-bound due to SM underutilization.

Split-KV (FlashDecoding pattern) partitions the sequence across multiple blocks per
KV head. Instead of one block processing all 1024 tokens, 16 blocks each process 64
tokens. This gives 8 × 16 = 128 grid blocks → all SMs busy.

## Prediction

- Single-request latency at seq=1024: **89μs → 15-25μs** (4-6× speedup)
- With batch=1: grid blocks = kv_heads × num_splits. At 16 splits: 128 blocks for 108 SMs.
- The combine step adds ~5μs (reduce partial results across splits).
- Total: ~20-30μs, competitive with SDPA's 20μs.

## How FlashDecoding works

Standard decode: 1 block per (batch, kv_head), processes ALL seq tokens.
```
Block (b=0, h=0): tokens 0..1023 → output[0, h=0]
Block (b=0, h=1): tokens 0..1023 → output[0, h=1]
...
Grid: batch × kv_heads = 8 blocks
```

Split-KV: multiple blocks per (batch, kv_head), each processes a CHUNK of seq.
```
Block (b=0, h=0, split=0):  tokens 0..63    → partial[0, h=0, s=0] + lse[0, h=0, s=0]
Block (b=0, h=0, split=1):  tokens 64..127  → partial[0, h=0, s=1] + lse[0, h=0, s=1]
...
Block (b=0, h=0, split=15): tokens 960..1023 → partial[0, h=0, s=15] + lse[0, h=0, s=15]
Grid: batch × kv_heads × num_splits = 128 blocks

Combine kernel: merge partial[0, h=0, 0..15] → output[0, h=0] using online softmax merge
```

### Merge formula (online softmax across splits)
```
For each split s:
  m_s = max attention score in split s
  d_s = sum of exp(score - m_s) in split s
  o_s = weighted sum of V in split s (under local softmax)

Global merge:
  m_global = max(m_0, m_1, ..., m_S)
  For each split s:
    scale_s = exp(m_s - m_global) * d_s
  d_global = sum(scale_s)
  o_global = sum(scale_s * o_s) / d_global
```

This is exactly what our cross-tz merge (sync_state) already does — but across
blocks instead of within a block. We already have working merge logic.

## Method

### Phase 1: Modify v4 kernel to support partition_kv mode
- Already have `partition_kv`, `kv_chunk_size_ptr`, `kv_tile_indices` in Params
- The v4 kernel already reads chunk_start/chunk_end from these
- Need: each block writes to `tmp_o[bx, qo_head, head_dim]` and `lse[bx, qo_head]`
  instead of final output

### Phase 2: Write combine kernel
- Input: tmp_o[num_blocks, num_qo_heads, head_dim], lse[num_blocks, num_qo_heads]
- Output: o[batch, num_qo_heads, head_dim]
- Simple reduction: online softmax merge across splits per (batch, qo_head)

### Phase 3: Host-side dispatch
- Compute num_splits = min(ceil(seq_len / chunk_size), max_splits)
- Set up request_indices, kv_tile_indices, block_valid_mask
- Launch v4 kernel with partition_kv=true
- Launch combine kernel

## Prior art

- FlashDecoding (Dao et al.): split-KV for decode attention, 8× speedup
- FlashDecoding++: improved load balancing across splits
- FlashInfer's `BatchDecodeWithPagedKVCacheDispatched` already does this —
  the Params struct has `partition_kv`, `kv_chunk_size_ptr`, etc.
- Our v4 kernel already has the partition_kv plumbing in the Params struct

## Risks

1. Combine kernel overhead: ~5μs per combine call. At small seq_len,
   the overhead may exceed the parallelism benefit.
2. Temporary memory: need tmp_o and tmp_lse buffers (num_blocks × qo_heads × head_dim).
   At 128 blocks: 128 × 16 × 128 × 4 = 1MB. Negligible.
3. Load imbalance: last split may have fewer tokens. Mitigated by block_valid_mask.

## Results

Correctness: cos>0.999 at all split counts (2, 4, 8, 16).

| splits | blocks | seq=1024 | vs nosplit |
|--------|--------|----------|-----------|
| none | 8 | **103 μs** | 1.0× |
| 2 | 16 | 327 μs | 3.2× slower |
| 4 | 32 | 282 μs | 2.7× slower |
| 8 | 64 | 257 μs | 2.5× slower |
| 16 | 128 | 255 μs | 2.5× slower |
| 32 | 256 | 257 μs | 2.5× slower |

**Split-KV is 2.5× SLOWER.** The overhead outweighs SM utilization gain.

### Overhead breakdown (~155 μs total overhead)

1. **half→float conversion** (~50 μs): `tmp_o_half.to(kFloat32)` copies all partial
   results from fp16 to fp32 for the combine kernel. This is a full GPU memcpy.
2. **Combine kernel launch** (~10 μs): SplitKVCombineKernel reduces partial results.
3. **Python tensor setup** (~50 μs): creating request_indices, kv_tile_indices,
   block_valid_mask on CPU and copying to GPU.
4. **Smaller bdz** (bdz=4 vs 16): each split-block uses fewer threads, less
   internal parallelism. This partially offsets the SM-level parallelism gain.

### Why it doesn't help at seq=1024

At seq=1024 with 8 KV heads, the non-split kernel takes 103μs total:
- 8 blocks finish in ~103μs (each block does 1024/64 = 16 tile iterations)
- Even with 128 blocks (all SMs busy), each block does 1 iteration → ~10μs
- But 155μs overhead > 93μs savings

### When split-KV WOULD help

At longer sequences where per-block work >> overhead:
- seq=32K: non-split would take ~3200μs. Split-16: ~200μs + 155μs overhead = 355μs.
- The crossover is around seq ≈ 4K-8K.

### Fixes needed for production split-KV

1. **Float output in kernel**: avoid half→float conversion by writing float directly
2. **Fuse tensor setup in C++**: avoid Python tensor creation overhead
3. **Adaptive split count**: only split when seq_len > threshold (~4K)

## Status: rejected (at seq≤2K)
Overhead (155μs) exceeds SM utilization benefit. Needs optimization for long contexts.
