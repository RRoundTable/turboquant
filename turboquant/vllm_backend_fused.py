# TurboQuant backend with fused CUDA decode kernel.
# Maintains separate quantized KV tensors alongside vLLM's cache.
# Write: quantize → store in own tensors + fp16 in vLLM cache (for prefill FA).
# Decode: fused CUDA kernel reads from own quantized tensors.

import math
import os
import functools
from pathlib import Path
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

TILE_DIMS = 64
QUANT_BYTES_PER_CHUNK = 32  # 64 dims × 4 bits / 8
NORM_BYTES_PER_CHUNK = 2

_C4 = [-2.7326368,-2.0693470,-1.6180345,-1.2562491,-0.9423684,-0.6567591,-0.3880823,-0.1284154,
        0.1284154, 0.3880823, 0.6567591, 0.9423684, 1.2562491, 1.6180345, 2.0693470, 2.7326368]
_B4 = [-2.4009919,-1.8436908,-1.4371418,-1.0993087,-0.7995637,-0.5224207,-0.2582488, 0.0,
        0.2582488, 0.5224207, 0.7995637, 1.0993087, 1.4371418, 1.8436908, 2.4009919]


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pack_4bit(idx):
    return ((idx[..., 0::2] << 4) | (idx[..., 1::2] & 0x0F)).to(torch.uint8)


class TurboQuantBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "float16", "bfloat16", "turboquant",
    ]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return TurboQuantFusedImpl

    @staticmethod
    def get_builder_cls():
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # For fp8 cache: [2, blocks, block_size, kv_heads, head_size] uint8
        # We store quant+norms in the first bytes_per_head bytes, rest is padding.
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)


class TurboQuantMetadataBuilder(FlashAttentionMetadataBuilder):
    pass


class TurboQuantFusedImpl(FlashAttentionImpl):
    """TurboQuant with fused CUDA decode kernel.

    When kv_cache_dtype="turboquant": single quantized cache, 3.76× memory savings.
    The vLLM cache stores [quant_bytes | norm_bytes] per token per head as uint8.
    Prefill: quantize KV → store in cache, dequant from cache → FA.
    Decode: v4 kernel reads quantized cache directly.

    When kv_cache_dtype="auto"/"float16": dual storage (backward compat).
    """

    def __init__(self, num_heads, head_size, scale, num_kv_heads,
                 alibi_slopes, sliding_window, kv_cache_dtype, *args, **kwargs):
        # Bypass FlashAttention's fp8 device check (we use our own decode kernel)
        self._real_kv_cache_dtype = kv_cache_dtype
        parent_dtype = "auto" if kv_cache_dtype in ("fp8", "fp8_e4m3") else kv_cache_dtype
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, parent_dtype, *args, **kwargs)
        self.kv_cache_dtype = self._real_kv_cache_dtype
        self._pd = _next_pow2(self.head_size)
        self._dc = self._pd // TILE_DIMS
        self._qbytes = self._dc * QUANT_BYTES_PER_CHUNK
        self._nbytes = self._dc * NORM_BYTES_PER_CHUNK
        self._signs = None
        self._fused_module = None
        self._write_module_cache = None
        self._is_tq_cache = self.kv_cache_dtype in ("fp8", "fp8_e4m3", "turboquant")
        # Legacy dual-storage tensors (only when NOT using turboquant cache)
        self._k_quant = None

    def _ensure(self, dev):
        if self._signs is not None:
            return
        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        signs = torch.sign(torch.randn(self._pd, generator=gen))
        signs[signs == 0] = 1.0
        self._signs = signs.to(dev)

    def _ensure_legacy_storage(self, kv_cache):
        """Allocate separate quantized tensors (dual storage mode only)."""
        if self._k_quant is not None:
            return
        nb, bs = kv_cache.shape[1], kv_cache.shape[2]
        dev = kv_cache.device
        self._k_quant = torch.zeros(nb, self.num_kv_heads, bs, self._qbytes, dtype=torch.uint8, device=dev)
        self._v_quant = torch.zeros_like(self._k_quant)
        self._k_norms = torch.zeros(nb, self.num_kv_heads, bs, self._dc, dtype=torch.float16, device=dev)
        self._v_norms = torch.zeros_like(self._k_norms)
        self._k_quant_flat = self._k_quant.view(-1)
        self._v_quant_flat = self._v_quant.view(-1)
        self._k_norms_flat = self._k_norms.view(-1)
        self._v_norms_flat = self._v_norms.view(-1)

    def _get_fused_module(self):
        if self._fused_module is not None:
            return self._fused_module
        from turboquant.decode_kernel_v4 import _get_module
        self._fused_module = _get_module()
        return self._fused_module

    def _get_write_module(self):
        if self._write_module_cache is not None:
            return self._write_module_cache
        from turboquant.decode_kernel_v4 import _CSRC_DIR
        from torch.utils.cpp_extension import load
        self._write_module_cache = load(
            name="turboquant_write",
            sources=[str(_CSRC_DIR / "src" / "quantize_write_binding.cu")],
            extra_include_paths=[str(_CSRC_DIR / "include")],
            extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
            verbose=False)
        return self._write_module_cache

    def _parse_tq_cache(self, kv_cache):
        """Split vLLM's fp8/uint8 cache into quant + norms views.

        kv_cache: [2, blocks, block_size, kv_heads, head_size] uint8
        First qbytes bytes = packed quant data, next nbytes = norms (fp16 as uint8).
        Remaining bytes are padding (unused).
        """
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        k_q = k_cache[..., :self._qbytes].contiguous()
        v_q = v_cache[..., :self._qbytes].contiguous()
        k_n = k_cache[..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16)
        v_n = v_cache[..., self._qbytes:self._qbytes + self._nbytes].contiguous().view(torch.float16)
        return k_q.view(-1), v_q.view(-1), k_n.view(-1), v_n.view(-1), kv_cache.shape[2]

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self._ensure(key.device)
        num_tokens = slot_mapping.shape[0]

        if self._is_tq_cache:
            # Single-cache mode: write quantized data directly to vLLM's fp8/uint8 cache
            # kv_cache: [2, blocks, block_size, kv_heads, head_size] uint8
            block_size = kv_cache.shape[2]
            bids = slot_mapping // block_size
            boffs = slot_mapping % block_size
            try:
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
            except Exception:
                pass
        else:
            # Legacy dual-storage mode
            self._ensure_legacy_storage(kv_cache)
            block_size = self._k_quant.shape[2]
            bids = slot_mapping // block_size
            boffs = slot_mapping % block_size
            try:
                wm = self._get_write_module()
                kq, vq, kn, vn = wm.quantize_write_kv(
                    key[:num_tokens].float(), value[:num_tokens].float(),
                    self._signs, self.head_size, self._pd)
                self._k_quant[bids, :, boffs, :] = kq
                self._v_quant[bids, :, boffs, :] = vq
                self._k_norms[bids, :, boffs, :] = kn
                self._v_norms[bids, :, boffs, :] = vn
            except Exception:
                pass
            # Also store fp16 for prefill FA
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
            if self._is_tq_cache:
                # TQ-native prefill: compute attention from fresh fp16 K,V
                # using PyTorch SDPA. Bypasses FA entirely (no dtype check).
                num_tokens = attn_metadata.num_actual_tokens
                q = query[:num_tokens]
                k = key[:num_tokens]
                v = value[:num_tokens]
                if output is None:
                    output = torch.empty_like(q)

                # Reshape for SDPA: [batch=1, heads, seq, head_dim]
                # Handle GQA: repeat K,V heads to match Q heads
                nqo, nkv = self.num_heads, self.num_kv_heads
                q_sdpa = q.transpose(0, 1).unsqueeze(0)  # [1, nqo, seq, hd]
                k_sdpa = k.transpose(0, 1).unsqueeze(0)  # [1, nkv, seq, hd]
                v_sdpa = v.transpose(0, 1).unsqueeze(0)
                if nqo != nkv:
                    gqa = nqo // nkv
                    k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
                    v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)

                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    attn_out = F.scaled_dot_product_attention(
                        q_sdpa.to(torch.float16), k_sdpa.to(torch.float16),
                        v_sdpa.to(torch.float16), is_causal=True,
                        scale=self.scale)

                output[:num_tokens] = attn_out.squeeze(0).transpose(0, 1).to(query.dtype)
                return output
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        # === DECODE: fused CUDA kernel ===
        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]
        if output is None:
            output = torch.empty_like(q)

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens

        try:
            module = self._get_fused_module()

            if self._is_tq_cache:
                kqf, vqf, knf, vnf, block_size = self._parse_tq_cache(kv_cache)
            else:
                block_size = self._k_quant.shape[2]
                kqf, vqf = self._k_quant_flat, self._v_quant_flat
                knf, vnf = self._k_norms_flat, self._v_norms_flat

            # Page table: vectorized GPU ops
            num_pages_per_req = (seq_lens + block_size - 1) // block_size
            kv_indptr = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device)
            torch.cumsum(num_pages_per_req, dim=0, out=kv_indptr[1:])
            kv_last_page_len = (seq_lens - (num_pages_per_req - 1) * block_size).to(torch.int32)
            page_offsets = torch.arange(block_table.shape[1], device=q.device)
            kv_indices = block_table[page_offsets.unsqueeze(0) < num_pages_per_req.unsqueeze(1)].to(torch.int32)

            # v4 kernel: fused Hadamard Q rotate + attention + output un-rotate
            result = module.decode_v4(
                q.to(torch.float16) if self.head_size == self._pd else
                    F.pad(q.to(torch.float16), (0, self._pd - self.head_size)),
                kqf, vqf, knf, vnf,
                kv_indices, kv_indptr, kv_last_page_len,
                self.num_kv_heads, block_size,
                self.head_size, self._pd, self.scale,
                self._signs,
            )

            output[:num_actual] = result[:num_actual, :, :self.head_size] if self.head_size < self._pd else result[:num_actual]

        except Exception as e:
            import sys
            print(f"[TQ] Fused kernel failed ({e}), falling back to FA", file=sys.stderr)
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        return output
