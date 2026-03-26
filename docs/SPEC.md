# Spec

## Invariants

- Quantized KV cache entries always produce valid dequantized tensors of the original shape
- Compression ratio is deterministic for a given bit-width and head dimension
- Same seed produces identical quantization results across runs
- MSE distortion decreases monotonically as bit-width increases

---

## 1. KV Cache Compression

Users can compress KV cache entries at configurable bit-widths (1-4 bits per coordinate).

### Behaviors

- **Given** a KV cache with 3-bit quantization, **when** key/value tensors are stored and retrieved, **then** the retrieved tensors match the originals with cosine similarity > 0.95
- **Given** a 3-bit configuration, **when** compression ratio is queried, **then** it reports approximately 5.3x vs fp16
- **Given** vectors of any norm, **when** quantized and dequantized, **then** the reconstruction scales correctly with the original norm
- **Given** multiple sequential token batches, **when** appended to the cache, **then** all prior entries remain accessible and the sequence length grows correctly

## 2. Unbiased Attention Estimation

Users can choose an inner-product-optimized mode that guarantees unbiased attention scores.

### Behaviors

- **Given** inner-product mode at 3 bits, **when** attention scores are computed with quantized keys, **then** the expected score equals the true score (unbiased)
- **Given** inner-product mode, **when** bit-width increases, **then** attention score variance decreases
- **Given** inner-product mode, **when** bit-width is 1, **then** the system rejects it (minimum 2 bits required)

## 3. Outlier-Aware Precision

Users can allocate higher precision to high-variance channels to improve quality at the same average bit-width.

### Behaviors

- **Given** 32 outlier channels at 3 bits and 96 regular channels at 2 bits, **when** effective bit-width is queried, **then** it reports 2.5 bits
- **Given** a sample of KV vectors, **when** outlier calibration runs, **then** the highest-variance channels are automatically selected
- **Given** outlier channels are configured, **when** tensors are stored and retrieved, **then** the output has the full original dimension (outlier and regular channels recombined)

## 4. vLLM Serving

Users can apply TurboQuant to a vLLM instance so inference uses compressed KV cache transparently.

### Behaviors

- **Given** a vLLM LLM instance, **when** TurboQuant is applied, **then** subsequent generation calls use quantized KV storage without API changes
- **Given** TurboQuant is applied, **when** the model generates text, **then** output quality is comparable to unquantized inference (< 1% perplexity increase)
- **Given** TurboQuant is applied with 3-bit quantization, **when** long sequences are generated, **then** GPU memory usage for KV cache is reduced by approximately 5x

## 5. SGLang Serving

Users can apply TurboQuant to an SGLang engine for compressed KV cache during inference.

### Behaviors

- **Given** an SGLang engine, **when** TurboQuant is applied, **then** subsequent generation uses quantized KV storage transparently
- **Given** TurboQuant is applied, **when** multi-turn conversations are run, **then** the KV cache across turns uses quantized storage

## 6. FlashInfer Integration

Users can run TurboQuant quantization inside FlashInfer's attention kernels for production-grade performance.

### Behaviors

- **Given** FlashInfer is available, **when** TurboQuant is configured in vLLM, **then** quantization/dequantization runs at the kernel level (not Python-level wrapping)
- **Given** FlashInfer integration is active, **when** paged attention is used, **then** quantized KV entries work with FlashInfer's page table management
- **Given** FlashInfer integration, **when** throughput is measured, **then** there is no significant overhead vs unquantized FlashInfer attention (< 5% latency increase)
