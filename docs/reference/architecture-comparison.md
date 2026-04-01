# FlashInfer vs TurboQuant Architecture Comparison

This document compares the data flow and memory hierarchy of FlashInfer (FP16 baseline)
and FlashInfer + TurboQuant (4-bit quantized KV cache) across prefill, decode, and the
GPU memory stack.

## GPU Memory Hierarchy

TurboQuant changes only the VRAM storage format and the VRAM-to-SMEM loading path.
Everything from shared memory onward is identical to FlashInfer FP16.

```mermaid
graph TB
    subgraph FP16["FlashInfer FP16"]
        direction TB
        R1["<b>Registers</b> (per thread)<br/>q_vec[8]: float32<br/>s[j]: float32<br/>st.o[8]: float32<br/>~50 regs/thread"]
        S1["<b>Shared Memory (L1)</b><br/>48-100 KB/SM<br/><br/>k_smem: fp16<br/>v_smem: fp16<br/>[tile_tokens x head_dim]"]
        L1["<b>L2 Cache</b><br/>6-96 MB<br/><br/>KV tiles cached: fp16<br/>2 bytes/element"]
        V1["<b>VRAM (HBM)</b><br/>16-80 GB<br/><br/>KV cache: fp16<br/>512 bytes/token/head"]

        V1 -->|"cp_async (HW DMA)<br/>2 bytes/elem"| L1
        L1 --> S1
        S1 -->|"cast_load<br/>fp16 → float32"| R1
    end

    subgraph TQ["FlashInfer + TurboQuant"]
        direction TB
        R2["<b>Registers</b> (per thread)<br/>q_vec[8]: float32<br/>s[j]: float32<br/>st.o[8]: float32<br/>codebook[16]: const mem<br/>~50 regs/thread"]
        S2["<b>Shared Memory (L1)</b><br/>48-100 KB/SM<br/><br/>k_smem: fp16 (SAME)<br/>v_smem: fp16 (SAME)<br/>[tile_tokens x head_dim]"]
        L2["<b>L2 Cache</b><br/>6-96 MB<br/><br/>KV tiles: 4-bit packed<br/>0.5 bytes/elem<br/><i>3.76x more fits</i>"]
        V2["<b>VRAM (HBM)</b><br/>16-80 GB<br/><br/>KV cache: 4-bit packed<br/>136 bytes/token/head<br/><b>3.76x smaller</b>"]

        V2 -->|"global load<br/>0.5 bytes/elem<br/><b>3.76x less BW</b>"| L2
        L2 -->|"codebook lookup<br/>nibble → fp16"| S2
        S2 -->|"cast_load<br/>fp16 → float32<br/>(IDENTICAL)"| R2
    end

    style V1 fill:#f9d0d0
    style V2 fill:#d0f9d0
    style L1 fill:#f9d0d0
    style L2 fill:#d0f9d0
    style S1 fill:#e8e8e8
    style S2 fill:#e8e8e8
    style R1 fill:#e8e8e8
    style R2 fill:#e8e8e8
```

**Key difference:** VRAM stores 4-bit packed data (3.76x smaller). The dequant-load replaces
`cp_async` with a codebook lookup that writes fp16 into shared memory. From SMEM onward,
the compute path is byte-for-byte identical.

## Prefill Phase

During prefill, both systems compute attention identically. The difference is how
the KV cache is stored afterward.

```mermaid
flowchart LR
    subgraph Input
        QKV["Q, K, V<br/>[seq_len, heads, dim]<br/>fp16"]
    end

    subgraph FA["FlashAttention (same kernel)"]
        ATTN["Tiled QK^T → softmax → xV<br/>→ Output"]
    end

    QKV --> FA

    subgraph FP16_Store["FlashInfer FP16 Store"]
        direction TB
        KS1["Store K: fp16"] --> VRAM1["VRAM<br/>2 bytes/elem<br/>512 B/token/head"]
        VS1["Store V: fp16"] --> VRAM1
    end

    subgraph TQ_Store["TurboQuant Store"]
        direction TB
        NORM["1. L2 normalize<br/>norm = ||x||, x̂ = x/norm"]
        HAD["2. Hadamard rotate<br/>x̃ = signs * FWHT(x̂)<br/>coords → ~N(0,1/d)"]
        QUANT["3. Lloyd-Max quantize<br/>4-bit codebook (16 levels)<br/>→ nearest centroid index"]
        PACK["4. Bit-pack & store<br/>2 nibbles/byte + fp16 norm"]
        VRAM2["VRAM<br/>0.53 bytes/elem<br/>136 B/token/head"]

        NORM --> HAD --> QUANT --> PACK --> VRAM2
    end

    FA --> FP16_Store
    FA --> TQ_Store

    style VRAM1 fill:#f9d0d0
    style VRAM2 fill:#d0f9d0
    style FA fill:#e8e8ff
```

### Prefill VRAM Layout Comparison

Per page (16 tokens, head_dim=128):

```mermaid
flowchart LR
    subgraph FP16["FlashInfer FP16 — 8192 B/page/head"]
        direction TB
        K1["K data<br/>16 x 128 x 2B<br/><b>4096 B</b>"]
        V1["V data<br/>16 x 128 x 2B<br/><b>4096 B</b>"]
    end

    subgraph TQ["TurboQuant 4-bit — 2112 B/page/head"]
        direction TB
        K2["K quant<br/>16 x 64B<br/><b>1024 B</b>"]
        KN["K norms<br/>16 x 2B<br/><b>32 B</b>"]
        V2["V quant<br/>16 x 64B<br/><b>1024 B</b>"]
        VN["V norms<br/>16 x 2B<br/><b>32 B</b>"]
    end

    style K1 fill:#f9d0d0
    style V1 fill:#f9d0d0
    style K2 fill:#d0f9d0
    style KN fill:#d0f9d0
    style V2 fill:#d0f9d0
    style VN fill:#d0f9d0
```

## Decode Phase

Decode reads ALL past KV tokens but produces only 1 output token per request.
This makes it **memory-bandwidth bound** — the bottleneck is reading KV from VRAM.
TurboQuant reads 3.76x less data.

```mermaid
flowchart TB
    subgraph FP16["FlashInfer FP16 Decode"]
        direction TB
        Q1["Load Q (1 token, tiny)"]

        subgraph TILE1["For each KV tile"]
            direction TB
            LK1["<b>1. Load K tile</b><br/>cp_async: VRAM → SMEM<br/>HW DMA, fp16<br/>2 bytes/elem"]
            QK1["<b>2. QK dot product</b><br/>SMEM → cast_load → regs<br/>s[j] = Σ q[i]*k[i]<br/>+ warp shuffle reduce"]
            SM1["<b>3. Online softmax</b><br/>m = max(m, s[j])<br/>d += exp2(s-m)<br/>rescale o_acc"]
            LV1["<b>4. Load V tile</b><br/>cp_async: VRAM → SMEM<br/>HW DMA, fp16"]
            VA1["<b>5. V accumulate</b><br/>o[i] += s[j] * v[i]"]

            LK1 --> QK1 --> SM1 --> LV1 --> VA1
        end

        MERGE1["<b>6. sync_state()</b><br/>Cross-warp merge"]
        OUT1["<b>7. Output + LSE</b><br/>Write to VRAM"]

        Q1 --> TILE1 --> MERGE1 --> OUT1
    end

    subgraph TQ["TurboQuant Decode"]
        direction TB
        Q2["Load Q (1 token, tiny)"]

        subgraph TILE2["For each KV tile"]
            direction TB
            LK2["<b>1. DEQUANT-LOAD K tile</b><br/>global load 4-bit packed<br/>→ codebook[nibble] x norm<br/>→ write fp16 to SMEM<br/><b>3.76x less VRAM read</b>"]
            QK2["<b>2. QK dot product</b><br/><i>IDENTICAL to FP16</i>"]
            SM2["<b>3. Online softmax</b><br/><i>IDENTICAL to FP16</i>"]
            LV2["<b>4. DEQUANT-LOAD V tile</b><br/>same dequant as K<br/><b>3.76x less VRAM read</b>"]
            VA2["<b>5. V accumulate</b><br/><i>IDENTICAL to FP16</i>"]

            LK2 --> QK2 --> SM2 --> LV2 --> VA2
        end

        MERGE2["<b>6. sync_state()</b><br/><i>IDENTICAL</i>"]
        OUT2["<b>7. Output + LSE</b><br/><i>IDENTICAL</i>"]

        Q2 --> TILE2 --> MERGE2 --> OUT2
    end

    style LK1 fill:#f9d0d0
    style LV1 fill:#f9d0d0
    style LK2 fill:#d0f9d0
    style LV2 fill:#d0f9d0
    style QK1 fill:#e8e8e8
    style QK2 fill:#e8e8e8
    style SM1 fill:#e8e8e8
    style SM2 fill:#e8e8e8
    style VA1 fill:#e8e8e8
    style VA2 fill:#e8e8e8
```

### Decode: Dequant-Load Detail

This is the only changed code path. Every other function is FlashInfer's original.

```mermaid
flowchart LR
    subgraph FP16_Load["FlashInfer cp_async"]
        direction LR
        HBM1["VRAM<br/>fp16 KV page"] -->|"cp_async<br/>(HW DMA)"| SMEM1["SMEM<br/>fp16 tile"]
    end

    subgraph TQ_Load["TurboQuant dequant_load"]
        direction LR
        HBM2["VRAM<br/>4-bit packed<br/>+ fp16 norm"] -->|"global load<br/>uint8 bytes"| REG["Registers<br/>packed nibbles"]
        REG -->|"nibble extract<br/>(hi>>4, lo&0xF)"| IDX["Codebook<br/>index [0-15]"]
        IDX -->|"codebook[idx]<br/>x norm x scale"| FP16["fp16 value"]
        FP16 -->|"store"| SMEM2["SMEM<br/>fp16 tile"]
    end

    SMEM1 -..->|"From here on:<br/>IDENTICAL path"| COMPUTE["cast_load → QK → softmax → V"]
    SMEM2 -..->|"From here on:<br/>IDENTICAL path"| COMPUTE

    style HBM1 fill:#f9d0d0
    style HBM2 fill:#d0f9d0
    style SMEM1 fill:#e8e8e8
    style SMEM2 fill:#e8e8e8
```

## Bandwidth Analysis

Decode is memory-bound: it reads the entire KV cache but only computes 1 output token.
TurboQuant's 3.76x compression directly reduces the bandwidth bottleneck.

| Sequence Length | FP16 Read | TurboQuant Read | Savings |
|:-:|:-:|:-:|:-:|
| 1K | 6.0 MB | 1.6 MB | 3.76x |
| 4K | 24 MB | 6.4 MB | 3.76x |
| 16K | 96 MB | 25.5 MB | 3.76x |
| 64K | 384 MB | 102 MB | 3.76x |

The dequant compute cost is **fixed** (~15 us for codebook lookup).
The bandwidth savings **scale linearly** with sequence length.
At long sequences, TurboQuant's decode can be **faster** than FP16 because
bandwidth savings outweigh dequant overhead.

```
Theoretical decode latency @ 500 GB/s bandwidth:

  FP16 (seq=4K):     24 MB / 500 GB/s = 48 us
  TurboQuant (4K):   6.4 MB / 500 GB/s = 13 us + 15 us dequant = 28 us  (1.7x faster)

  FP16 (seq=64K):    384 MB / 500 GB/s = 768 us
  TurboQuant (64K):  102 MB / 500 GB/s = 204 us + 15 us dequant = 219 us (3.5x faster)
```

## Summary: What Changes, What Stays

```mermaid
pie title "TurboQuant Code Changes vs FlashInfer"
    "CHANGED: KV load (dequant_load)" : 5
    "UNCHANGED: QK dot product" : 20
    "UNCHANGED: Online softmax" : 15
    "UNCHANGED: V accumulate" : 20
    "UNCHANGED: Cross-warp merge" : 15
    "UNCHANGED: Output write" : 10
    "CHANGED: KV store (quantize)" : 15
```

| Component | FlashInfer FP16 | TurboQuant | Changed? |
|:--|:--|:--|:-:|
| **Prefill attention** | FlashAttention | FlashAttention | No |
| **KV store** | memcpy fp16 | normalize → Hadamard → quantize → pack | **Yes** |
| **KV load (decode)** | cp_async (HW DMA) | codebook dequant → fp16 SMEM | **Yes** |
| **QK dot product** | warp shuffle reduce | warp shuffle reduce | No |
| **Online softmax** | exp2 + running max | exp2 + running max | No |
| **V accumulate** | fused multiply-add | fused multiply-add | No |
| **Cross-warp merge** | sync_state() | sync_state() | No |
| **Output** | cast_store + LSE | cast_store + LSE | No |
| **SMEM format** | fp16 tiles | fp16 tiles | No |
| **Register format** | float32 | float32 | No |
| **VRAM format** | fp16 (2 B/elem) | 4-bit packed (0.53 B/elem) | **Yes** |
