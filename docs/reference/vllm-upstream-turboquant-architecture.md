# Upstream vLLM native TurboQuant — architecture reference

Source: vLLM v0.20.0 tag (commit `579602aa4be6`).
Module root: `vllm/model_executor/layers/quantization/turboquant/` + `vllm/v1/attention/{backends,ops}/turboquant_*`.

PRs:
- **#38479** (Vibhav Agarwal, 2026-04-15) — initial implementation, 27 files / +2940 LOC.
- **#40194** (Dan Alistarh, 2026-04-18) — removed random signs + prior-art attribution (HIGGS / Cache-Me-If-You-Must).

This doc is a **baseline to improve upon**. Covers the quant/dequant/fusion
paths end-to-end, then — separately — how the paper's QJL step would
slot in (since upstream explicitly omits it).

---

## 1. File tree + responsibility

| file | LOC | purpose |
|---|---:|---|
| `quantization/turboquant/config.py` | 195 | `TurboQuantConfig` dataclass + 4 named presets + slot-size math |
| `quantization/turboquant/centroids.py` | 86 | Lloyd-Max centroid solver (Gaussian N(0, 1/d)) + `@lru_cache` |
| `quantization/turboquant/quantizer.py` | 8 | empty stub — "Triton kernels handle all quantization" |
| `v1/attention/backends/turboquant_attn.py` | 800 | attention backend: prefill/decode/continuation dispatch, metadata builder |
| `v1/attention/ops/triton_turboquant_store.py` | 447 | store kernel — fused quant + pack + write (FP8 path and MSE path) |
| `v1/attention/ops/triton_turboquant_decode.py` | 623 | decode kernel — split-KV tiled attention with on-the-fly dequant |
| `v1/kv_cache_interface.py` | +26 | custom `page_size_bytes` hook (same concept as our PR #39868) |
| `model_executor/layers/attention/attention.py` | +82 | TQ backend dispatch, skip-layer logic |

**Pure Python orchestration. All GPU math is Triton.** No custom CUDA (.cu) files — everything is JIT-compiled Triton at first use.

---

## 2. Cache slot layout — byte level

Cache tensor shape (no leading `2` for K/V separation, unlike standard backends):

```
kv_cache : uint8 [num_blocks, block_size, num_kv_heads, slot_size_aligned]
```

Per-slot layout (one head, one token position):

```
┌─────────────────────────────────┬───────────────────────────────────┐
│   KEY_packed  (key_packed_size) │  VALUE_packed (value_packed_size) │
└─────────────────────────────────┴───────────────────────────────────┘
                                                              │
slot_size = key_packed_size + value_packed_size               │
slot_size_aligned = round_up_even(slot_size)                  │
```

### KEY layout — two modes

**FP8 key (`turboquant_k8v4`):**
```
┌──────────── D bytes ────────────┐
│  FP8(E4M3 or E4B15) key element │   (1 byte per dim, no norm, no rotation)
└──────────────────────────────────┘
key_packed_size = D
```

**MSE key (`turboquant_{4bit_nc, k3v4_nc, 3bit_nc}`):**
```
┌─ceil(D·mse_bits/8) bytes─┬─ 2 bytes ─┐
│  packed MSE indices       │ ‖k‖₂ fp16 │
└───────────────────────────┴───────────┘
key_packed_size = ceil(D · mse_bits / 8) + 2
                   (16 bytes for D=128, 4-bit)
                   (48 bytes for D=128, 3-bit)
```

### VALUE layout — uniform quant (same for all presets)

```
┌─ceil(D·vqb/8) bytes─┬─ 2 bytes ─┬─ 2 bytes ─┐
│  packed value data   │ scale fp16 │ zero fp16 │
└──────────────────────┴────────────┴───────────┘
value_packed_size = ceil(D · value_quant_bits / 8) + 4
                   (68 bytes for D=128, 4-bit)
                   (52 bytes for D=128, 3-bit)
```

### Worked sizes for Qwen3-4B head_dim=128

| preset | kps | vps | slot | aligned | compression vs fp16 (512B/slot) |
|---|---:|---:|---:|---:|---:|
| `turboquant_k8v4` | 128 | 68 | 196 | **196** | **2.61×** |
| `turboquant_4bit_nc` | 66 | 68 | 134 | **134** | **3.82×** |
| `turboquant_k3v4_nc` | 50 | 68 | 118 | **118** | **4.34×** |
| `turboquant_3bit_nc` | 50 | 52 | 102 | **102** | **5.02×** |

(kps = key_packed_size, vps = value_packed_size)

---

## 3. Prefill / store flow

### Outer dispatch (`turboquant_attn.py:do_kv_cache_update`)

1. Model runner calls `do_kv_cache_update(key, value, kv_cache, slot_mapping)` **before** attention forward (separate custom op, matches FlashAttention split pattern).
2. Reshape raw post-RoPE K/V `[N, Hk, D]` → `[NH, D]`.
3. Dispatch to FP8 or MSE kernel based on `key_fp8` flag.

### FP8 path (`_tq_fused_store_fp8`)

One CUDA thread block per `(token, head)`. Per-slot work:

```
K:  raw_fp16 ──→ .to(tl.float8e4nv | tl.float8e4b15)  bitcast→uint8
    └── stored directly at slot_base + [0..D)

V:  raw_fp16 ──→ per-vector min/max ──→ scale = (max-min)/15
                 quantize: q = clamp((v - min)/scale + 0.5, 0, 15)
                 pack: two 4-bit values per byte
                 store: data + scale(fp16) + zero(fp16)=min
```

- **FP8 format selection:** `FP8_E4B15` constant chosen at launch from
  `_use_fp8_e4b15(dev)`, which returns `1` iff device `SM < 8.9` (Ampere/Ada).
  Hopper+ uses `float8e4nv` directly.
- **This is the path that produces garbage on A100** (our HYP-057 result,
  0.012 vs fp16 0.591). The `float8e4b15` fallback in Triton on Ampere
  silently corrupts.

### MSE path (`_tq_fused_store_mse`, two-stage)

Stage 1 runs outside the kernel (cuBLAS is faster for the GEMM):
```python
# turboquant_attn._store_kv
k_flat = key.float().reshape(NH, D)
norms  = k_flat.norm(dim=1, keepdim=True)         # [NH, 1]
x_hat  = k_flat / (norms + 1e-8)                  # unit-normalize
y      = x_hat @ PiT                              # rotate by Hadamard
```

`PiT = Pi = H_d / √d` — Sylvester Hadamard, built once per `(d, device)`
via `@functools.cache`. The PR authors deliberately removed random sign
flips (PR #40194) — "random sign flips do not improve Lloyd-Max because
quantizer is symmetric around zero".

Stage 2 runs in one fused Triton kernel, per `(token, head)`:
```
a) Binary-search bucketize y → idx ∈ [0, 2^mse_bits)
   (MSE_BITS iterations vs N_CENTROIDS-1 for linear scan)
b) Pack idx into mse_bits-per-element stream (3-bit → 8-per-24, 4-bit → 2-per-byte)
c) Store packed MSE bytes at slot_base
d) Store ‖k‖₂ fp16 at slot_base + MSE_BYTES
e) Value quant same as FP8 path (uniform, 3 or 4 bit) at slot_base + KPS
```

### Lloyd-Max centroids

`centroids.py:solve_lloyd_max(d, bits)`:
- Assume post-rotation each coord ~ `N(0, 1/d)` (true for `d ≥ 64`).
- Iteratively solve for up to 200 rounds or `tol=1e-10` convergence.
- Uses a trapezoidal integrator (no scipy dependency).
- Cached via `@lru_cache(maxsize=32)` — computed once per `(d, bits)` per process.

```python
for iter in range(200):
    boundaries[i] = (centroids[i] + centroids[i+1]) / 2
    centroids_new[i] = ∫ x·pdf(x) dx / ∫ pdf(x) dx   over [boundary_{i-1}, boundary_i]
```

`centroid` tensor is loaded to GPU once per layer and passed into the store/decode kernels.

---

## 4. Decode flow

### Outer dispatch (`_decode_attention`)

For decode step: `query.shape = [B, Hq, D]` (one token per request).
Calls Triton `triton_turboquant_decode_attention` with split-KV partition.

**Q pre-rotation**: before entering the kernel, Q is rotated by Pi and (optionally) has norm correction applied so the Triton kernel can compute `q_rot · centroid` scores directly without per-tile rotation. Implementation uses a small cuBLAS GEMM `q @ Pi`.

### Core kernel (`_tq_decode_stage1`)

Launch grid: `(batch, Hq_per_kv_head, NUM_KV_SPLITS)`.
Per block processes one KV split for one `(request, q_head)`.

```
For each BLOCK_KV (=16) tokens in this KV split:
  ┌─ KEY PHASE ──────────────────────────────────────────┐
  │  if KEY_FP8:                                          │
  │     k_raw = load(KV_cache[slot_base + d_offs])        │
  │     k_float = k_raw.bitcast(float8e4[nv|b15]).to(fp32)│
  │     score = dot(q_rot, k_float) * scale               │
  │                                                       │
  │  else (MSE path):                                     │
  │     # Unpack mse_bits from packed byte stream         │
  │     mse_byte_idx = d_offs * MSE_BITS // 8             │
  │     mse_bit_shift = d_offs * MSE_BITS % 8             │
  │     raw16 = load(byte[idx]) | load(byte[idx+1]) << 8  │
  │     mse_idx = (raw16 >> bit_shift) & mask             │
  │     c_vals = load(Centroids[mse_idx])                 │
  │     if NORM_CORRECTION:                               │
  │        c_vals = c_vals / ‖c_vals‖                     │
  │     vec_norm = load(KV_cache[slot_base + MSE_BYTES])  │
  │     score = vec_norm * dot(q_rot, c_vals) * scale     │
  └───────────────────────────────────────────────────────┘
  
  ┌─ ONLINE SOFTMAX (FlashAttention-style) ──────────────┐
  │  m_new = max(m_prev, max(scores))                     │
  │  re_scale = exp(m_prev - m_new)                       │
  │  p = exp(scores - m_new)                              │
  │  l_new = l_prev * re_scale + sum(p)                   │
  │  m_prev, l_prev = m_new, l_new                        │
  └───────────────────────────────────────────────────────┘
  
  ┌─ VALUE PHASE ────────────────────────────────────────┐
  │  Unpack value indices (3 or 4 bit)                    │
  │  v_scale, v_zero = load(slot_base + KPS + VAL_DATA)  │
  │  values = idx * v_scale + v_zero                      │
  │  acc = acc * re_scale + sum(p * values)               │
  └───────────────────────────────────────────────────────┘

Store partial result Mid_o = acc / l; lse = m + log(l)
```

### Split-KV stage-2 (`_fwd_kernel_stage2`, reused from vLLM)

**Not a custom kernel** — calls `vllm.v1.attention.ops.triton_decode_attention._fwd_kernel_stage2` directly. This is the generic log-sum-exp reduction that merges `NUM_KV_SPLITS` partial attention results per `(batch, q_head)` into final output. Saves ~200 LOC.

### Kernel-level optimizations

| technique | where | effect |
|---|---|---|
| **Binary-search bucketize** at store | `_tq_fused_store_mse` | O(mse_bits) vs O(2^mse_bits) linear scan |
| **In-register value dequant** | `_tq_decode_stage1` | no intermediate fp16 smem buffer |
| **Fused QK + softmax + VA** | `_tq_decode_stage1` | one kernel per split, no mat round-trip |
| **Split-KV** | `NUM_KV_SPLITS` grid dim | SM saturation for long contexts |
| **Hadamard matmul** | outside kernel (cuBLAS GEMM) | beats in-kernel butterfly on D=128 (~1 kernel launch vs log₂D) |
| **Centroid `@lru_cache`** | `centroids.py` | compute once per process |
| **`functools.cache` Hadamard** | `_build_hadamard_cached` | one allocation per `(d, device)` |
| **Online softmax** | stage1 | FlashAttention-style numeric stability |

---

## 5. Continuation prefill

Not all prefill is "all-new tokens". If a request has `cached_len > 0`
(prior KV in TQ cache) and a new chunk `q_len` arrives:

```
if q_len <= 128 (_CONTINUATION_DECODE_THRESHOLD):
    # Fast path: pretend each query token is a decode request
    # with seq_len ranging over cached_len+1 .. cached_len+q_len
    # (creates causal mask implicitly)
    triton_turboquant_decode_attention(query=q_seq, ...)

else:
    # Bulk dequant cached KV to fp16 via _tq_full_dequant_kv,
    # concat with new chunk's raw K/V, hand to flash_attn_varlen_func
    _continuation_prefill(...)
```

### `_tq_full_dequant_kv` (the bulk dequant kernel)

One program per `(token_position, batch * head)`. Walks the block table,
reconstructs K and V to fp16 in contiguous `[B, Hk, max_seq, D]` buffers,
then caller runs flash_attn on the full dequantized tensor.

This is where we'd see highest memory cost if a model has many long-context
continuations — the 16 MB dequant buffer is reused across layers via
`layer._tq_k_dequant_buf` / `_tq_v_dequant_buf`.

---

## 6. First-chunk prefill — not quantized at all

```
if q_len == seq_len and flash_attn available:
    flash_attn_varlen_func(q, k, v, ..., causal=True)
```

**First-chunk prefill attention runs on raw fp16 K/V.** Quantization only
happens afterwards, when `do_kv_cache_update` stores those K/V to cache.
This preserves prefill attention quality exactly — only decode (and
continuation-prefill) ever see the quantized cache.

---

## 7. Fusion map

```
PREFILL (first chunk)
──────────────────────
        raw fp16 K,V
              │
              ├── flash_attn_varlen_func ──────→ attention output (fp16)
              │
              └── do_kv_cache_update
                    │
                    ├──→ [FP8 path]   _tq_fused_store_fp8
                    │                 (cast + V quant + pack + store,
                    │                  1 kernel)
                    │
                    └──→ [MSE path]   1) cuBLAS GEMM y = x_hat @ PiT
                                      2) _tq_fused_store_mse
                                         (bucketize + pack + norm +
                                          V quant + pack + store,
                                          1 kernel)

DECODE / CONTINUATION
──────────────────────
    query q  ──(cuBLAS GEMM q_rot = q @ Pi)──┐
                                             │
                                             ▼
                        _tq_decode_stage1 (per KV split):
                          • unpack K indices → centroids
                          • dot(q_rot, c_vals) * vec_norm * scale
                          • online softmax
                          • unpack V indices → scale/zero dequant
                          • weighted value accum
                                             │
                                             ▼
                        _fwd_kernel_stage2 (generic vLLM split-merge)
                                             │
                                             ▼
                                  attention output (fp16)
```

**Number of kernel launches per decode step (one layer):**
1. `q @ Pi` (cuBLAS) — ~1 µs
2. `_tq_decode_stage1` — main cost
3. `_fwd_kernel_stage2` — LSE reduction
4. (no separate dequant kernel — fused)

---

## 8. Boundary-skip layers

`TurboQuantConfig.get_boundary_skip_layers(num_layers, n=2)` returns
`["0", "1", "N-2", "N-1"]` — convention from prior KV-quant work (KVQuant,
Cache-Me-If-You-Must) that the first and last few layers are most
sensitive. Consumed via `kv_cache_dtype_skip_layers` vLLM arg.

---

## 9. What upstream **does not** include — paper's QJL pipeline

The paper's Algorithm 2 (inner-product-unbiased variant) layers QJL on top
of the MSE path. Upstream explicitly omits this. If we ever want to add
it back as an improvement axis, the integration surface is:

### Paper Algorithm 2 recap (from
`docs/reference/turboquant-paper-methodology.md` §1):

```
Global setup:
  MSE centroids for bit-width (b − 1)           ← Alg 1's centroids at one-bit-less
  S ∈ R^{d×d} with S_ij ~ N(0, 1)               ← DENSE random projection matrix

Quant_prod(x):                                   # encode
  idx = Quant_mse(x)                             # same as current upstream
  r = x − DeQuant_mse(idx)                       # residual in input space
  qjl = sign(S · r)                              # d-dim ±1, 1 bit per dim
  return (idx, qjl, ‖r‖₂)

DeQuant_prod(idx, qjl, γ):                       # decode
  x̃_mse = DeQuant_mse(idx)
  x̃_qjl = (√(π/2) / d) · γ · S^T · qjl           # dense d×d matmul
  return x̃_mse + x̃_qjl
```

### What adding QJL to upstream would require

1. **Cache slot layout +18 bytes per key** (d=128):
   ```
   ┌─ MSE bytes ─┬─ ‖k‖₂ (2B) ─┬─ QJL signs (d/8 = 16 B) ─┬─ ‖r‖₂ (2 B) ─┐
   ```
   Adds 18 B/key → compression drops from 3.82× to ~3.30× at `4bit_nc`.
   Value path unchanged.

2. **Store kernel change** (`_tq_fused_store_mse`):
   After computing `idx` via bucketize, reconstruct in-register:
   ```
   mse_hat = centroids[idx]              # d-element vector in register
   r = y − mse_hat                       # residual (y = x_hat @ PiT)
   r_norm = ‖r‖₂
   # ... then compute sign(S · r) — this is the new heavy op
   ```
   `sign(S · r)` is a `d × d` GEMV per token — at d=128, that's **16 384 FMAs** per token per head. Currently, the store kernel does zero `d²` work (only bucketize + pack). This is the **main performance cost** of reintroducing QJL.

3. **Decode kernel change** (`_tq_decode_stage1`):
   Score for inner-product mode becomes:
   ```
   score_mse = vec_norm · dot(q_rot, c_vals)     # current
   score_qjl = (√(π/2) / d) · ‖r‖₂ · dot(q_rot_S, qjl_signs)
   #            ^— where q_rot_S = S · q_rot (1 d²-GEMV per decode step, per q_head)
   score = (score_mse + score_qjl) · scale
   ```
   The `q_rot_S` GEMV is a per-step per-head cost. If `d = 128, Hq = 32, batch = 1`, that's `128² · 32 = 524 288` FMAs per decode step per layer — on A100 fp16 (312 TFLOPS) about **1.7 µs per layer**, ~60 µs over 36 layers. For context our v5 kernel spends ~60 µs **total** at short-seq decode. So **dense QJL nearly doubles decode TPOT.**

4. **`S` matrix storage + regeneration.** Dense Gaussian `d × d` per layer per head (or shared? paper is ambiguous). At d=128 fp16: 32 KB/head. Qwen3-8B 8 KV heads × 36 layers = ~9 MB constant. Small.

5. **Hadamard-structured JL as an alternative.** Instead of dense S, use `S = D · H` where D is ±1 diagonal and H is Hadamard. FWHT is `O(d log d) = 896` FMAs vs `d² = 16 384`. **18× cheaper.** Variance constant slightly worse but the O(1/d) bound survives. This is well-studied (Ailon-Chazelle FJLT, SRHT). Upstream chose to drop QJL entirely instead of investigating structured variants.

### Why upstream omits QJL

Per PR #40194 docstring verbatim:
> "QJL is intentionally omitted — community consensus (5+ independent
> groups) found it hurts attention quality by amplifying variance through
> softmax."

Our HYP-049/050/052/054/055c form four of those five rejections — all
found QJL net-negative on LongBench / synthetic inner-product benchmarks
at the bit-widths we tested.

### The interesting open direction — per HYP-055b's close-out

The only regime where QJL showed a consistent net-positive on our stack
was **3.5-bit outlier-aware with QJL on regulars only** (not paper's
"both tiers"). Upstream's current presets are all uniform (no outlier
tier) — if we build a **mixed-precision TQ variant in upstream style** we
could revisit QJL-on-regulars as an improvement path. See HYP-057 §Implications.

---

## 10. Comparison vs our `vllm_backend_fused.py` plugin stack

| layer | upstream | our plugin (pre-v0.20.0) |
|---|---|---|
| attention backend | `turboquant_attn.py` (Python, 800 LOC) | `vllm_backend_fused.py` (Python) |
| store kernel | Triton `_tq_fused_store_{fp8,mse}` | CUDA `write_kernel.py` (JIT) |
| decode kernel | Triton `_tq_decode_stage1` + generic stage2 | CUDA v5 `decode_kernel.py` (JIT) |
| page-size hook | upstreamed (`v1/kv_cache_interface.py`) | our `docker/vllm_patches/` (now redundant) |
| key algo | Hadamard + Lloyd-Max MSE or FP8 | Hadamard + Lloyd-Max MSE (4-bit only) |
| value algo | Uniform quant + scale/zero (per-token) | Lloyd-Max MSE (same as keys) |
| QJL | omitted | omitted |
| outlier-aware | no | no (shipped path) — HYP-055c's B_35_prime has the split |
| CUDA graph | `UNIFORM_BATCH` support | our `decode_v5_from_cache_ws` supports graphs |
| A100 correctness | **3/4 presets OK, k8v4 broken** (HYP-057) | 4-bit MSE passes HYP-029, 3.76× confirmed |
| supported bit-widths | 3, 4, FP8 (via `2^b` Lloyd-Max) | 4 only (our codebook.py caps at 4-bit) |

---

## 11. Improvement axes (ordered by ROI)

1. **Port `k8v4` fix for A100.** File upstream bug with HYP-057
   test case (`tests/bench_longbench_vllm.py`). Most likely fix: switch
   `float8e4b15` Ampere path to `float8e5m2` or emulated fp8 — Triton's
   `e4b15` format is known-problematic on SM80.

2. **Add outlier-aware mixed precision.** Upstream's all-uniform presets
   leave 1–2 pp on the table at 3.5-bit vs our HYP-055c outlier-aware
   `B_35_prime`. Add a new preset `turboquant_k4v3_outlier_nc` =
   `{32ch k4v4 + 96ch k3v3_nc}` = 3.5 avg bits. Implementation: second
   set of store/decode kernels or per-channel mask in existing kernels.

3. **Structured QJL (Hadamard JL) as an optional residual.** Only at 3.5+
   avg bits (HYP-055b's regime where QJL net-wins). Adds 18 B per key.
   Decode adds `O(d log d)` FMAs vs dense QJL's `O(d²)` — 18× cheaper.
   Per-layer overhead ~4 µs per decode step (tolerable).

4. **Skip-layer heuristic auto-tuning.** Upstream uses hard-coded
   `n=2` boundary layers. A per-model calibration (measure which layers
   have highest MSE after TQ) could reduce skip set.

5. **Value quant — try Lloyd-Max on values instead of uniform.** Our
   own experiments show Lloyd-Max beats uniform at 3-bit regardless of
   data distribution. Currently upstream value path only does uniform
   min/max + scale/zero. Would need a small 3-bit Lloyd-Max codebook
   for values and a store-time bucketize (similar to key path).

6. **Tile size auto-tune.** `BLOCK_KV=16` is hard-coded. On H100/B200
   with more smem, `BLOCK_KV=32` or `64` may reduce loop overhead.

7. **`q_rot = q @ Pi` in-kernel.** Currently a cuBLAS GEMM. For `Hq=32`,
   `d=128`, that's a `32×128×128 = 524 288` FMA GEMM — ~0.5 µs on A100.
   Tiny but nonzero. Fusing into `_tq_decode_stage1` saves a kernel
   launch per decode step.

---

## 12. Cross-references

- **Paper methodology:** `docs/reference/turboquant-paper-methodology.md`
  (§1 algorithms, §2 outlier split, §3 LongBench setup, §6 paper-vs-upstream delta).
- **Our HYP-057 verification:** `docs/hypotheses/HYP-057-upstream-vllm-turboquant-longbench.md`
  (3 of 4 upstream presets reproduce paper's fp16-parity claim; k8v4 broken on A100).
- **Our prior QJL rejections:** HYP-049/050/052/054/055c — form 4 of the
  "5+ groups" upstream cites as justification for QJL omission.
- **Our own store/decode kernels:** `turboquant/write_kernel.py`,
  `turboquant/decode_kernel.py`, `turboquant/decode_kernel_v4.py`,
  `csrc/include/flashinfer_decode_turboquant_v*.cuh`.
