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

    def _get_decode_module(self):
        if self._decode_module is None:
            from turboquant.decode_kernel_v4 import _get_module
            self._decode_module = _get_module()
        return self._decode_module

    def _get_write_module(self):
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
        return self._write_module

    def _write_to_cache(self, key, value, kv_cache, slot_mapping):
        """Quantize K,V and write to uint8 cache."""
        num_tokens = slot_mapping.shape[0]
        block_size = kv_cache.shape[2]
        bids = slot_mapping // block_size
        boffs = slot_mapping % block_size

        wm = self._get_write_module()
        kq, vq, kn, vn = wm.quantize_write_kv(
            key[:num_tokens].float(), value[:num_tokens].float(),
            self._signs, self.head_size, self._pd)

        kn_u8 = kn.view(torch.uint8)
        vn_u8 = vn.view(torch.uint8)
        kv_cache[0, bids, boffs, :, :self._qbytes] = kq
        kv_cache[0, bids, boffs, :, self._qbytes:self._qbytes + self._nbytes] = kn_u8
        kv_cache[1, bids, boffs, :, :self._qbytes] = vq
        kv_cache[1, bids, boffs, :, self._qbytes:self._qbytes + self._nbytes] = vn_u8

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self._ensure(key.device)

        if self._is_fp8:
            # Single-cache mode: quantize and write directly to uint8 cache.
            # No fp16 copy — the cache IS the quantized storage.
            try:
                self._write_to_cache(key, value, kv_cache, slot_mapping)
            except Exception as e:
                import sys
                print(f"[TQ] Write kernel failed: {e}", file=sys.stderr)
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
                # Prefill: FA needs fp16 cache tensor. Allocate a SMALL dummy —
                # just enough blocks for this batch. FA writes here but we discard.
                # Real quantized data goes to the uint8 cache via do_kv_cache_update.
                bs, nkv = kv_cache.shape[2], kv_cache.shape[3]
                num_tokens = attn_metadata.num_actual_tokens
                dummy_blocks = (num_tokens + bs - 1) // bs + 1  # +1 safety margin
                if not hasattr(self, '_dummy_cache') or self._dummy_cache.shape[1] < dummy_blocks:
                    self._dummy_cache = torch.zeros(
                        2, dummy_blocks, bs, nkv, self.head_size,
                        dtype=torch.float16, device=kv_cache.device)
                # vLLM pre-quantizes query to fp8 when kv_cache_dtype="fp8".
                # Cast back to fp16 for FA prefill (FA only accepts fp16/bf16).
                query_fp16 = query.to(torch.float16) if query.dtype != torch.float16 else query
                key_fp16 = key.to(torch.float16) if key.dtype != torch.float16 else key
                value_fp16 = value.to(torch.float16) if value.dtype != torch.float16 else value
                saved = self.kv_cache_dtype
                self.kv_cache_dtype = "auto"
                try:
                    result = super().forward(layer, query_fp16, key_fp16, value_fp16,
                                            self._dummy_cache, attn_metadata,
                                            output, output_scale, **kwargs)
                finally:
                    self.kv_cache_dtype = saved
                return result
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
            module = self._get_decode_module()

            if self._is_fp8:
                # Parse uint8 cache: vLLM layout [blocks, entries, heads, bytes]
                # Kernel expects HND: [blocks, heads, entries, bytes]
                # Transpose dims 1,2 to convert NHD → HND
                block_size = kv_cache.shape[2]
                k_hnd = kv_cache[0].transpose(1, 2)  # [blocks, heads, entries, bytes]
                v_hnd = kv_cache[1].transpose(1, 2)
                k_q = k_hnd[..., :self._qbytes].contiguous().view(-1)
                v_q = v_hnd[..., :self._qbytes].contiguous().view(-1)
                k_n = k_hnd[..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16).view(-1)
                v_n = v_hnd[..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16).view(-1)
            else:
                # Legacy: shouldn't reach here without fp8 cache
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, **kwargs)

            # Page table on GPU
            num_pages = (seq_lens + block_size - 1) // block_size
            kv_indptr = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device)
            torch.cumsum(num_pages, dim=0, out=kv_indptr[1:])
            kv_last_page_len = (seq_lens - (num_pages - 1) * block_size).to(torch.int32)
            page_offsets = torch.arange(block_table.shape[1], device=q.device)
            kv_indices = block_table[page_offsets.unsqueeze(0) < num_pages.unsqueeze(1)].to(torch.int32)

            # v4 kernel with fused Hadamard
            q_fp16 = q.to(torch.float16)
            if self.head_size < self._pd:
                q_fp16 = F.pad(q_fp16, (0, self._pd - self.head_size))

            result = module.decode_v4(
                q_fp16, k_q, v_q, k_n, v_n,
                kv_indices, kv_indptr, kv_last_page_len,
                self.num_kv_heads, block_size,
                self.head_size, self._pd, self.scale,
                self._signs,
            )

            if self.head_size < self._pd:
                output[:num_actual] = result[:num_actual, :, :self.head_size]
            else:
                output[:num_actual] = result[:num_actual]

        except Exception as e:
            import sys
            print(f"[TQ] Decode kernel failed ({e}), falling back to FA", file=sys.stderr)
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        return output
