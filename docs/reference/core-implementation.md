# TurboQuant Core Implementation

GPU-native implementation of the TurboQuant algorithm (arXiv:2504.19874).
All tensors live on CUDA. No CPU fallback.

---

## Algorithm Overview

TurboQuant compresses vectors by exploiting the geometry of random rotations on the unit sphere.

```
Input vector x ∈ R^d
  → extract norm:  ‖x‖
  → normalize:     x̂ = x / ‖x‖
  → random rotate: y = FWHT(signs ⊙ pad(x̂))     ← O(d log d)
  → scalar quantize each coordinate of y           ← codebook lookup
  → bit-pack indices                                ← compression
  → store: (packed_indices, norm)

Reconstruct:
  → unpack indices
  → centroid lookup: ŷ from codebook
  → inverse rotate: x̂ = unpad(signs ⊙ FWHT(ŷ))
  → rescale: x̃ = x̂ · ‖x‖
```

**Why it works:** The random Hadamard rotation maps any unit vector to a near-uniform distribution on the sphere. After rotation, each coordinate follows approximately N(0, 1/d), which can be optimally quantized with precomputed Lloyd-Max centroids.

---

## Modules

### Codebook (`codebook.py`)

Precomputed Lloyd-Max centroids for N(0, 1), scaled by 1/√d at init.

| Bit-width | Levels | Centroids (N(0,1)) |
|-----------|--------|-------------------|
| 1 | 2 | ±0.798 |
| 2 | 4 | ±0.453, ±1.510 |
| 3 | 8 | ±0.245, ±0.756, ±1.344, ±2.152 |
| 4 | 16 | ±0.128, ±0.388, ..., ±2.733 |

- **Quantize:** `torch.bucketize(x, boundaries)` — maps values to nearest centroid index
- **Dequantize:** Table lookup `centroids[indices]`

### Hadamard (`hadamard.py`)

**Fast Walsh-Hadamard Transform (FWHT):** Normalized in-place butterfly algorithm. O(d log d) for d-dimensional vector. Requires d = power of 2.

**RandomHadamardRotation:** Combines FWHT with random ±1 signs to create a pseudo-random orthogonal rotation:
- Forward: `y = FWHT(signs ⊙ pad(x))`
- Inverse: `x = unpad(signs ⊙ FWHT(y))`

Non-power-of-2 dimensions are zero-padded, then the extra coordinates are stripped on inverse.

Signs are generated from a seeded CPU generator then moved to CUDA for deterministic reproducibility across runs.

### QJL (`qjl.py`)

1-bit Quantized Johnson-Lindenstrauss transform for unbiased inner-product estimation.

- **Quantize:** `Q(x) = sign(S · x)` where S is a d×d Gaussian random matrix
- **Dequantize:** `Q⁻¹(z) = √(π/2) / d · γ · Sᵀ · z` where γ is the residual norm

Guarantees: E[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩ (unbiased) with variance ≤ π/(2d) · ‖y‖²

Only used by TurboQuantProd (Algorithm 2) for the residual stage.

### Quantizer (`quantizer.py`)

**TurboQuantMSE (Algorithm 1):** Minimizes reconstruction MSE.

```
quantize(x) → (packed_indices, norms)
dequantize(packed_indices, norms) → x̃
```

Distortion bound: D_mse ≤ √(3π)/2 · 4^(-b) for unit vectors.

**TurboQuantProd (Algorithm 2):** Provides unbiased inner-product estimation.

Two-stage:
1. MSE quantizer with (b-1) bits on the input
2. QJL (1-bit) on the residual

```
quantize(x) → (mse_packed, mse_norms, qjl_signs, residual_norms)
dequantize(mse_packed, mse_norms, qjl_signs, residual_norms) → x̃
```

Guarantee: E[⟨y, x̃⟩] = ⟨y, x⟩ (unbiased)

---

## Bit Packing

Indices are packed into bytes for compression. The packing scheme:

| Bit-width | Values per byte | Bits used | Bits wasted |
|-----------|----------------|-----------|-------------|
| 1 | 8 | 8 | 0 |
| 2 | 4 | 8 | 0 |
| 3 | 2 | 6 | 2 |
| 4 | 2 | 8 | 0 |

**3-bit compression math** (head_dim=128):
- Indices: 128 coordinates → 64 packed bytes
- Norm: 1 × float32 = 4 bytes
- Total: 68 bytes per vector
- FP16 baseline: 128 × 2 = 256 bytes
- **Compression: 256 / 68 ≈ 3.8×**

Note: 3-bit is suboptimal for byte packing (2 wasted bits per byte). True 5× compression requires either kernel-level bit manipulation or 2-bit / 4-bit widths.

---

## Storage Layout

For a KV cache entry with shape `[batch, num_heads, seq_len, head_dim]`:

```
Quantized storage:
  packed_indices: [batch, num_heads, seq_len, packed_dim]  dtype=uint8
  norms:          [batch, num_heads, seq_len]               dtype=float32

Where packed_dim = ceil(padded_dim / vals_per_byte)
      padded_dim = next_power_of_2(head_dim)
      vals_per_byte = 8 // bit_width
```

---

## Measured Quality (DGX Spark, SM121)

### MSE distortion (unit vectors, d=128, 500 samples)

| Bit-width | Measured MSE | Paper bound |
|-----------|-------------|-------------|
| 1-bit | ~0.36 | 0.36 |
| 2-bit | ~0.12 | 0.117 |
| 3-bit | ~0.03 | 0.03 |
| 4-bit | ~0.009 | 0.009 |

### Attention quality (3-bit, d=128, 8 heads, 256 KV tokens)

- Relative error vs FP32 attention: < 0.30
- Cosine similarity of attention output: > 0.95

---

## Dependency Graph

```
TurboQuantMSE ──→ Codebook
              ──→ RandomHadamardRotation ──→ fwht

TurboQuantProd ──→ TurboQuantMSE
               ──→ QJL
```

All leaf modules (Codebook, RandomHadamardRotation, QJL) are independent with no cross-imports.
