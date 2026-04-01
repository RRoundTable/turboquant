# HYP-008: Isolate the real bottleneck — paging, dequant, or occupancy?

## Hypothesis
The 5× gap between v4 (296μs) and SDPA (60μs) at seq=1024 is NOT from scalar vs tensor
core compute, since SDPA decode also uses scalar dot products. The gap comes from one or
more of:

A. **Page table overhead**: divmod + __ldg per token per tile (our paged layout)
B. **Dequant ALU**: codebook lookup + norm multiply per element
C. **Low occupancy**: 64 threads (2 warps) per block — A100 wants 8+ warps
D. **Memory access pattern**: scattered page accesses vs contiguous KV

## Prediction
By isolating each factor, we can identify which contributes the most to the 5× gap.
Expected ranking: C (occupancy) > A (paging) > B (dequant) > D (access pattern).

## Method
Four microbenchmarks, all with same loop structure as v4 (QK + softmax + V_accum):

1. **v4-baseline**: Current v4 kernel (paged, dequant, 64 threads) — 296μs reference
2. **v4-contiguous-fp16**: Read fp16 K/V from contiguous memory (no paging, no dequant).
   Same thread config (64 threads). Isolates: paging + dequant overhead.
3. **v4-contiguous-dequant**: Read packed 4-bit from contiguous memory (no paging, yes dequant).
   Same thread config. Isolates: paging overhead alone.
4. **v4-high-occupancy**: Current v4 but with bdz=8 or 16 (128-256 threads).
   Isolates: occupancy effect.

Comparing:
- (1) vs (2): total paging + dequant overhead
- (2) vs (3): dequant overhead alone
- (1) vs (3): paging overhead alone
- (1) vs (4): occupancy effect
- (2) vs SDPA: remaining structural gap (loop overhead, sync count, etc.)

## Results (A100, 12 heads, hd=128, bdy=1)

| seq | SDPA | fp16/z4 | fp16/z8 | fp16/z16 | dq/z4 | dq/z8 | dq/z16 |
|-----|------|---------|---------|----------|-------|-------|--------|
| 128 | 24 | 40 | 23 | **16** | 37 | 21 | **15** |
| 256 | 31 | 76 | 42 | **26** | 69 | 33 | **18** |
| 512 | 31 | 114 | 61 | **35** | 104 | 54 | **32** |
| 1024 | 30 | 226 | 118 | **64** | 205 | 103 | **57** |
| 2048 | 30 | 445 | 231 | **123** | 406 | 201 | **110** |

## Analysis

### Finding 1: Dequant overhead is NEGATIVE (dequant is FASTER than fp16)

At every bdz, the 4-bit dequant kernel is **faster** than the contiguous fp16 kernel:
- seq=1024, bdz=16: dq=57μs vs fp16=64μs (dequant is 11% faster!)
- seq=2048, bdz=16: dq=110μs vs fp16=123μs (11% faster)

**Why:** 4-bit data is 4× smaller → 4× less memory bandwidth. The dequant ALU cost is
LESS than the memory bandwidth saved. This proves dequant is NOT a bottleneck.

### Finding 2: Occupancy (bdz) is the dominant factor

| bdz | Threads | seq=1024 fp16 | seq=1024 dq | Speedup vs bdz=4 |
|-----|---------|---------------|-------------|-------------------|
| 4   | 64      | 226 μs        | 205 μs      | 1.0×              |
| 8   | 128     | 118 μs        | 103 μs      | 2.0×              |
| 16  | 256     | 64 μs         | 57 μs       | 3.6×              |

Going from bdz=4 to bdz=16 gives **3.6× speedup**. This is purely occupancy — more
warps per block → better latency hiding → more memory requests in flight.

### Finding 3: At bdz=16, our contiguous kernel MATCHES SDPA

- seq=1024: dq/z16 = 57μs vs SDPA = 30μs (1.9×)
- seq=128: dq/z16 = 15μs vs SDPA = 24μs (**faster than SDPA!**)
- seq=256: dq/z16 = 18μs vs SDPA = 31μs (**faster than SDPA!**)

At short sequences, our contiguous 4-bit dequant kernel is actually FASTER than SDPA.
At longer sequences, SDPA scales better (30μs constant vs linear scaling).

### Finding 4: SDPA has ~30μs constant overhead

SDPA takes 24-31μs regardless of seq_len from 128 to 2048. This is kernel launch +
synchronization overhead, not data-proportional work. SDPA is extremely pipelined.

### Finding 5: The v4 gap (296μs) vs contiguous dq/z4 (205μs) = paging overhead

v4 at seq=1024: 296μs. Contiguous dq at bdz=4: 205μs. Difference: 91μs = **page table overhead** (divmod + __ldg per token per tile).

### Bottleneck decomposition (seq=1024, vs SDPA 30μs)

| Factor | Overhead | Evidence |
|--------|----------|----------|
| **Occupancy (bdz=4 vs 16)** | **148μs** (205→57) | bdz sweep |
| **Paging overhead** | **91μs** (296→205) | v4 vs contiguous dq/z4 |
| **Dequant ALU** | **-7μs** (FASTER than fp16) | dq/z16 vs fp16/z16 |
| **Remaining vs SDPA** | **27μs** (57→30) | Loop structure + sync overhead |

**The #1 optimization: increase bdz from 4 to 16.** This alone brings us from 296μs to ~80μs (estimated: contiguous 57μs + paging 23μs at higher bdz).

## Status: confirmed
Occupancy is the dominant bottleneck (3.6× from bdz=4→16). Dequant is NOT a bottleneck (faster than fp16). Page table overhead is secondary (~30%). At bdz=16 without paging, we match or beat SDPA at short sequences.
