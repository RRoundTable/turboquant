# TurboQuantTile Memory Layout

## Overview

TurboQuantTile stores a compressed KV cache tile of **16 tokens × 64 dimensions** using **3.5-bit quantization** (sign included). It is designed for FlashInfer integration and optimized for 128-bit vectorized GPU memory loads.

```
Total size: 480 bytes (30 × 16 bytes)
Alignment:  16-byte (128-bit)
Effective:  3.5 bits per value
```

## Struct Layout

```
Byte Offset   Field        Size      Description
───────────────────────────────────────────────────────────
0             norms        32 B      FP16 scale per row (16 × 2 bytes)
32            quant_hi    256 B      4-bit signed outlier channels (16 rows × 16 bytes)
288           quant_lo    192 B      3-bit signed normal channels (16 rows × 12 bytes)
───────────────────────────────────────────────────────────
              TOTAL       480 B
```

All fields start at 16-byte aligned offsets (0, 32, 288).

```
┌──────────────────────────────────────────────────────────────┐
│  norms (32 B)                                                │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐         │
│  │n0│n1│n2│n3│n4│n5│n6│n7│n8│n9│..│..│..│..│..│nF│  FP16   │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘         │
├──────────────────────────────────────────────────────────────┤
│  quant_hi (256 B) — 4-bit signed, 32 outlier dims            │
│  ┌────────────────────────────────────────────────┐          │
│  │ Row 0:  16 bytes  (32 dims × 4 bits)           │ ← 1 uint4│
│  │ Row 1:  16 bytes                                │   load   │
│  │ ...                                             │          │
│  │ Row 15: 16 bytes                                │          │
│  └────────────────────────────────────────────────┘          │
├──────────────────────────────────────────────────────────────┤
│  quant_lo (192 B) — 3-bit signed, 32 normal dims             │
│  ┌────────────────────────────────────────────────┐          │
│  │ Row 0:  12 bytes  (32 dims × 3 bits)           │          │
│  │ Row 1:  12 bytes                                │          │
│  │ ...                                             │          │
│  │ Row 15: 12 bytes                                │          │
│  └────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

## 3.5-Bit Quantization Scheme

Each value is encoded as a **signed index** where the MSB carries the sign and the remaining bits carry the magnitude:

```
4-bit signed (outlier channels, dims 0..31):
┌───────┬───────────────┐
│ bit 3 │ bits 2..0     │
│ sign  │ magnitude idx │
└───────┴───────────────┘
  sign: 0 = positive, 1 = negative
  magnitude: 3 bits → 8 levels (0..7)
  dequantized value = ±(mag_idx + 0.5) / 8 × norm

3-bit signed (normal channels, dims 32..63):
┌───────┬───────────┐
│ bit 2 │ bits 1..0 │
│ sign  │ mag idx   │
└───────┴───────────┘
  sign: 0 = positive, 1 = negative
  magnitude: 2 bits → 4 levels (0..3)
  dequantized value = ±(mag_idx + 0.5) / 4 × norm
```

Average bits per value: (4 × 32 + 3 × 32) / 64 = **3.5 bits**

## Bit Budget Per Value

```
                    Outlier (4-bit)    Normal (3-bit)
                    ───────────────    ──────────────
Sign                1 bit              1 bit
Magnitude index     3 bits (8 lvls)    2 bits (4 lvls)
                    ───────────────    ──────────────
Total per value     4 bits             3 bits
Dims                32                 32
Bits per row        128                96
Bytes per row       16                 12
```

## Packing Formats

### 4-bit: Nibble Packing

Two values per byte. High nibble = even index, low nibble = odd index.

```
byte = (value[2i] << 4) | value[2i+1]

Example: dims d0=0x0A, d1=0x05 → byte = 0xA5

Byte layout for 32 dims (16 bytes):
  byte[0] = [d0:d1]  byte[1] = [d2:d3]  ...  byte[15] = [d30:d31]
```

### 3-bit: Group-of-8 Packing

8 values packed into 3 bytes (24 bits). GGML-compatible, little-endian.

```
byte[0] = v0 | (v1 << 3) | (v2 << 6)           v2: low 2 bits
byte[1] = (v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)   v5: low 1 bit
byte[2] = (v5 >> 1) | (v6 << 2) | (v7 << 5)

Bit diagram (LSB on left):
byte[0]:  [v0₀ v0₁ v0₂ | v1₀ v1₁ v1₂ | v2₀ v2₁]
byte[1]:  [v2₂ | v3₀ v3₁ v3₂ | v4₀ v4₁ v4₂ | v5₀]
byte[2]:  [v5₁ v5₂ | v6₀ v6₁ v6₂ | v7₀ v7₁ v7₂]

32 dims = 4 groups × 3 bytes = 12 bytes per row
```

## Norms (FP16 Scale)

One FP16 norm per row (per token). Stored as raw `uint16_t` in IEEE 754 half-precision format.

```
Quantize:   norm = max(|values[0..63]|)  →  store as FP16
Dequantize: value = decode_signed(index) × fp16_to_float(norm)
```

The norm defines the dynamic range of the row. All magnitudes are normalized to [0, 1] before quantization.

## Per-Row Access Pattern (for CUDA Kernels)

To dequantize row `r` (one token, 64 dims):

```
norm     = fp16_to_float(tile.norms[r])               // 2 bytes at offset r*2
hi_data  = tile.quant_hi + r * 16                     // 16 bytes at offset 32 + r*16
lo_data  = tile.quant_lo + r * 12                     // 12 bytes at offset 288 + r*12
```

The 4-bit row (16 bytes) is exactly one 128-bit `uint4` load — optimal for GPU coalescing.

## Quantization Error Characteristics

Using Phase 1 uniform quantization:

```
                   Outlier (4-bit)         Normal (3-bit)
                   ───────────────         ──────────────
Magnitude levels   8                       4
Step size          1/8 = 0.125             1/4 = 0.25
Max error/norm     0.0625                  0.125
RMSE/norm          ~0.036                  ~0.072
Expected variance  step²/12 = 0.00130     step²/12 = 0.00521
```

Overall relative RMSE: ~5-8% for uniform input, ~15-22% for Gaussian (heavy tails).
Lloyd-Max codebooks (Phase 2) will significantly reduce error for Beta-distributed
data after PolarQuant rotation.

## Compression Ratio

```
FP16 baseline:  16 tokens × 64 dims × 16 bits = 16,384 bits = 2,048 bytes
TurboQuantTile: 480 bytes (including norms)

Compression ratio: 2,048 / 480 = 4.27×
Effective bits:    480 × 8 / 1024 = 3.75 bits/value (including amortized norms)
```

## Comparison to Paper

```
                    Our Impl        Paper (TQ3)     Paper (TQ3.5)
                    ────────        ───────────     ─────────────
Bits/value          3.5             3.0             3.5
Sign handling       MSB of index    MSB of index    MSB of index
Magnitude levels    8/4 (hi/lo)     4 (uniform)     8/4 (mixed)
Scale               FP16 per-row    FP32 per-128d   FP32 per-128d
Codebook            Uniform (Ph1)   Lloyd-Max       Lloyd-Max
Compression         4.27×           4.9×            ~4.3×
```
