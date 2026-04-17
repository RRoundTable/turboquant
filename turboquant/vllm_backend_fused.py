# TurboQuant vLLM backend — single quantized KV cache.
#
# Architecture:
#   Prefill: FA with fresh fp16 K,V (no cache read) + CUDA write to uint8 cache
#   Decode:  v4 kernel reads quantized uint8 cache directly
#
# Usage: LLM(..., attention_backend="CUSTOM", kv_cache_dtype="fp8")
#   fp8 gives uint8 allocation = 2× memory savings vs fp16.

import math
import os
import functools
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32  # 64 dims × 4 bits / 8


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class TurboQuantBackend(FlashAttentionBackend):
    # vLLM must call do_kv_cache_update separately (not inside forward).
    # We quantize K,V to the fp8 cache in do_kv_cache_update.
    # FA's forward only handles attention — not cache writes.
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TurboQuantFusedImpl

    @staticmethod
    def get_builder_cls():
        return TurboQuantMetadataBuilder

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str in ("fp8", "fp8_e4m3"):
            pd = _next_pow2(head_size)
            dc = pd // TILE_DIMS
            # cp_async.ca.shared.global with 128-bit loads requires 16-byte
            # aligned source — pad so entry byte stride is a multiple of 16.
            raw = dc * QUANT_BYTES_PER_CHUNK + dc * 2
            bytes_per_head = (raw + 15) & ~15
            return (2, num_blocks, block_size, num_kv_heads, bytes_per_head)
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)

    @classmethod
    def get_kv_cache_page_size(
        cls, block_size, num_kv_heads, head_size, dtype,
        cache_dtype_str="auto",
    ):
        if cache_dtype_str in ("fp8", "fp8_e4m3"):
            pd = _next_pow2(head_size)
            dc = pd // TILE_DIMS
            raw = dc * QUANT_BYTES_PER_CHUNK + dc * 2
            bytes_per_head = (raw + 15) & ~15
            return 2 * block_size * num_kv_heads * bytes_per_head
        return None


class TurboQuantMetadataBuilder(FlashAttentionMetadataBuilder):
    pass


class TurboQuantFusedImpl(FlashAttentionImpl):
    """Single quantized KV cache with fused CUDA kernels.

    Prefill: FlashAttention with fresh fp16 K,V + CUDA quantize to cache.
    Decode: v4 kernel reads quantized cache with fused Hadamard rotation.
    """

    def __init__(self, num_heads, head_size, scale, num_kv_heads,
                 alibi_slopes, sliding_window, kv_cache_dtype, *args, **kwargs):
        # Bypass FA's fp8 device check (we use our own decode kernel)
        self._real_cache_dtype = kv_cache_dtype
        parent_dtype = "auto" if kv_cache_dtype in ("fp8", "fp8_e4m3") else kv_cache_dtype
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, parent_dtype, *args, **kwargs)
        self.kv_cache_dtype = self._real_cache_dtype

        self._pd = _next_pow2(self.head_size)
        self._dc = self._pd // TILE_DIMS
        self._qbytes = self._dc * QUANT_BYTES_PER_CHUNK
        self._nbytes = self._dc * 2  # fp16 norms

        self._signs = None
        self._decode_module = None
        self._write_module = None
        self._is_fp8 = kv_cache_dtype in ("fp8", "fp8_e4m3")

    def _ensure(self, dev):
        if self._signs is not None:
            return
        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        signs = torch.sign(torch.randn(self._pd, generator=gen))
        signs[signs == 0] = 1.0
        self._signs = signs.to(dev)

    @staticmethod
    def _fwht(x):
        """Fast Walsh-Hadamard Transform on last dim. In-place-safe."""
        n = x.shape[-1]
        h = 1
        while h < n:
            shape = x.shape[:-1] + (n // (2 * h), 2, h)
            x = x.view(shape)
            a = x[..., 0, :].clone()
            b = x[..., 1, :].clone()
            x[..., 0, :] = a + b
            x[..., 1, :] = a - b
            x = x.view(*x.shape[:-3], n)
            h <<= 1
        return x

    def _hadamard_forward(self, q):
        """Forward rotation: signs × FWHT × scale (applied to Q)."""
        x = q.float() * self._signs
        x = self._fwht(x) * (1.0 / math.sqrt(x.shape[-1]))
        return x.to(q.dtype)

    def _hadamard_inverse(self, out):
        """Inverse rotation: FWHT × scale × signs (applied to output)."""
        x = out.float()
        x = self._fwht(x) * (1.0 / math.sqrt(x.shape[-1])) * self._signs
        return x.to(out.dtype)

    def _ensure_kernels_loaded(self):
        """Load kernel extensions once. JIT-compiles v4 (decode + write) and v5
        (tensor-core decode) on first call."""
        if self._decode_module is None:
            from turboquant.decode_kernel_v4 import _get_module, _CSRC_DIR, _find_flashinfer_include
            self._decode_module = _get_module()

            # v5 tensor-core kernel (HYP-031)
            from torch.utils.cpp_extension import load
            fi_include = _find_flashinfer_include()
            self._v5_module = load(
                name="turboquant_v5_tc",
                sources=[str(_CSRC_DIR / "src" / "decode_v5_tc_binding.cu")],
                extra_include_paths=[str(_CSRC_DIR / "include"), str(fi_include)],
                extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr",
                    "-use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"],
                verbose=False)

        if self._write_module is None:
            from turboquant.decode_kernel_v4 import _CSRC_DIR
            from torch.utils.cpp_extension import load
            self._write_module = load(
                name="turboquant_write",
                sources=[str(_CSRC_DIR / "src" / "quantize_write_binding.cu")],
                extra_include_paths=[str(_CSRC_DIR / "include")],
                extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
                verbose=False)

    def _write_to_cache(self, key, value, kv_cache, slot_mapping):
        """Quantize K,V and scatter into kv_cache via dispatcher-routed op so
        the storage pointer is refreshed on CUDA graph replay (not baked in
        as a raw ptr the way Python advanced-indexing scatter would)."""
        num_tokens = slot_mapping.shape[0]
        self._ensure_kernels_loaded()
        torch.ops.turboquant_write.quantize_write_kv_cache(
            kv_cache,
            key[:num_tokens].float(), value[:num_tokens].float(),
            self._signs,
            slot_mapping.to(torch.int32),
            self.head_size, self._pd,
        )

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self._ensure(key.device)

        if self._is_fp8:
            self._write_to_cache(key, value, kv_cache, slot_mapping)
        else:
            # Legacy fp16 mode: store fp16 in vLLM cache for FA
            from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
            num_tokens = slot_mapping.shape[0]
            key_cache, value_cache = kv_cache.unbind(0)
            reshape_and_cache_flash(
                key[:num_tokens], value[:num_tokens],
                key_cache, value_cache, slot_mapping,
                self.kv_cache_dtype, layer._k_scale, layer._v_scale,
            )

    @torch.no_grad()
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, **kwargs):
        if attn_metadata is None:
            if output is None:
                output = torch.zeros_like(query)
            return output

        max_query_len = attn_metadata.max_query_len
        is_decode = (max_query_len == 1)

        if not is_decode:
            if self._is_fp8:
                # Prefill: call flash_attn directly with fresh fp16 K,V.
                # Cannot use super().forward() — FA writes to cache slots
                # that may exceed a dummy cache's bounds.
                try:
                    from vllm.vllm_flash_attn import flash_attn_varlen_func
                except ImportError:
                    from flash_attn import flash_attn_varlen_func

                num_tokens = attn_metadata.num_actual_tokens
                q = query[:num_tokens].to(torch.float16)
                k = key[:num_tokens].to(torch.float16)
                v = value[:num_tokens].to(torch.float16)
                # Debug: verify dtypes in TP worker
                assert q.dtype == torch.float16, f"q dtype={q.dtype}"
                assert k.dtype == torch.float16, f"k dtype={k.dtype}"

                if output is None:
                    output = torch.empty_like(query[:num_tokens])

                seq_start = attn_metadata.query_start_loc
                max_seqlen = attn_metadata.max_query_len

                attn_out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=seq_start,
                    cu_seqlens_k=seq_start,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    softmax_scale=self.scale,
                    causal=True,
                )
                output[:num_tokens] = attn_out.to(output.dtype)
                return output
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        # === DECODE: v4 fused kernel ===
        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]
        if output is None:
            output = torch.empty_like(q)

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens

        try:
            self._ensure_kernels_loaded()

            if not self._is_fp8:
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, **kwargs)

            block_size = kv_cache.shape[2]

            # Page table: strided layout so all ops are static-shape and
            # capturable under CUDA graphs. kv_indices = block_table flattened;
            # kv_indptr steps by max_pages_per_seq. The kernel reads only up to
            # kv_len[b] tokens per sequence, so trailing invalid entries are
            # never touched.
            num_pages = (seq_lens + block_size - 1) // block_size
            max_pages = block_table.shape[1]
            kv_indptr = (torch.arange(seq_lens.shape[0] + 1,
                                      dtype=torch.int32, device=q.device)
                         * max_pages)
            kv_last_page_len = seq_lens - (num_pages - 1) * block_size
            kv_indices = block_table.reshape(-1)

            q_fp16 = q.to(torch.float16)
            if self.head_size < self._pd:
                q_fp16 = F.pad(q_fp16, (0, self._pd - self.head_size))

            # v5 tensor-core decode allocates gather buffers internally,
            # which is illegal during CUDA graph capture. Use v4 under
            # graphs (graph-safe, no allocations) and v5 under eager
            # (2.5× faster kernel).
            is_capturing = torch.cuda.is_current_stream_capturing()
            if is_capturing:
                result = torch.ops.turboquant.decode_v4_from_cache(
                    q_fp16, kv_cache,
                    kv_indices, kv_indptr, kv_last_page_len,
                    seq_lens,
                    self.num_kv_heads, block_size,
                    self.head_size, self._pd, self.scale,
                    self._signs,
                    self._qbytes, self._nbytes,
                )
            else:
                result = self._v5_module.decode_v5_from_cache(
                    q_fp16, kv_cache,
                    kv_indices, kv_indptr, kv_last_page_len,
                    seq_lens,
                    self.num_kv_heads, block_size,
                    self.head_size, self._pd, self.scale,
                    self._signs if self._signs is not None else torch.empty(0, device=q.device),
                    self._qbytes, self._nbytes,
                )

            if self.head_size < self._pd:
                output[:num_actual] = result[:num_actual, :, :self.head_size]
            else:
                output[:num_actual] = result[:num_actual]

        except Exception as e:
            import sys, traceback
            print(f"[TQ] Decode kernel failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            # Do NOT fall back to FA for fp8 cache — FA can't read uint8.
            # Re-raise so the error is visible.
            if self._is_fp8:
                raise
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        return output

