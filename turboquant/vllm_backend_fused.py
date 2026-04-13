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
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # HND physical layout matches paged_kv_turbo_t strides.
        # With uint8 view, writes go to correct HND positions.
        if include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        return (0, 1, 3, 2, 4)


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

        # View as uint8 to avoid float8 type-casting corruption.
        # vLLM allocates fp8 cache as float8_e4m3fn — raw byte writes
        # get type-cast, destroying our packed quantization bytes.
        cache_u8 = kv_cache.view(torch.uint8)
        kn_u8 = kn.view(torch.uint8)
        vn_u8 = vn.view(torch.uint8)
        cache_u8[0, bids, boffs, :, :self._qbytes] = kq
        cache_u8[0, bids, boffs, :, self._qbytes:self._qbytes + self._nbytes] = kn_u8
        cache_u8[1, bids, boffs, :, :self._qbytes] = vq
        cache_u8[1, bids, boffs, :, self._qbytes:self._qbytes + self._nbytes] = vn_u8

    _update_count = 0

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self._ensure(key.device)
        TurboQuantFusedImpl._update_count += 1
        if False and TurboQuantFusedImpl._update_count <= 3:
            import sys
            print(f"[TQ] do_kv_cache_update #{TurboQuantFusedImpl._update_count}: "
                  f"key={key.shape} dt={key.dtype} cache={kv_cache.shape} "
                  f"slots={slot_mapping[:5].tolist()} bs={kv_cache.shape[2]} fp8={self._is_fp8}",
                  flush=True, file=sys.stderr)

        if self._is_fp8:
            try:
                self._write_to_cache(key, value, kv_cache, slot_mapping)
                if False and TurboQuantFusedImpl._update_count <= 3:
                    import sys
                    num_tokens = slot_mapping.shape[0]
                    bs = kv_cache.shape[2]
                    bids = slot_mapping[:5] // bs
                    boffs = slot_mapping[:5] % bs
                    # Check what's at the first write location
                    sample = kv_cache[0, bids[0], boffs[0], 0, :8].tolist()
                    total_nz = (kv_cache[0] != 0).sum().item()
                    print(f"[TQ] Write OK. nz={total_nz} bids={bids.tolist()} "
                          f"sample={sample}", flush=True, file=sys.stderr)
            except Exception as e:
                import sys, traceback
                print(f"[TQ] Write kernel failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
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
                # Cannot use super().forward() — FA's internal reshape_and_cache_flash
                # writes to slot_mapping positions that exceed the dummy cache size.
                from vllm.vllm_flash_attn import flash_attn_varlen_func
                num_tokens = attn_metadata.num_actual_tokens
                q = query[:num_tokens].to(torch.float16)
                k = key[:num_tokens].to(torch.float16)
                v = value[:num_tokens].to(torch.float16)

                if output is None:
                    output = torch.empty_like(query[:num_tokens])

                # Build cu_seqlens for varlen FA from attn_metadata
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
            module = self._get_decode_module()

            if self._is_fp8:
                # Physical = HND [blocks, heads, entries, head_size] via stride order.
                # ZERO COPY: pass flat pointers directly into HND memory.
                # Kernel uses entry_byte_stride=head_size to skip norms+padding.
                # Norms pointer = quant pointer + qbytes offset (same underlying buffer).
                cache_u8 = kv_cache.view(torch.uint8)
                block_size = kv_cache.shape[2]
                # permute NHD→HND (no copy, just reinterprets strides)
                k_flat = cache_u8[0].permute(0, 2, 1, 3).contiguous().view(-1)
                v_flat = cache_u8[1].permute(0, 2, 1, 3).contiguous().view(-1)
                # k_flat is HND with head_size bytes per entry.
                # Quant data at offset 0, norms at offset qbytes.
                k_q = k_flat
                v_q = v_flat
                # Norms: offset by qbytes, viewed as fp16
                k_n = k_flat[self._qbytes:].view(torch.float16)
                v_n = v_flat[self._qbytes:].view(torch.float16)
                entry_stride = self.head_size
            else:
                # Legacy: shouldn't reach here without fp8 cache
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, **kwargs)

            # Page table: build once per decode step, cache for reuse across layers.
            # attn_metadata is the same object for all 28 layers in one step.
            _meta_id = id(attn_metadata)
            if not hasattr(self, '_pt_cache_id') or self._pt_cache_id != _meta_id:
                num_pages = (seq_lens + block_size - 1) // block_size
                self._pt_indptr = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32, device=q.device)
                torch.cumsum(num_pages, dim=0, out=self._pt_indptr[1:])
                self._pt_last = (seq_lens - (num_pages - 1) * block_size).to(torch.int32)
                page_offsets = torch.arange(block_table.shape[1], device=q.device)
                self._pt_indices = block_table[page_offsets.unsqueeze(0) < num_pages.unsqueeze(1)].to(torch.int32)
                self._pt_cache_id = _meta_id
            kv_indptr = self._pt_indptr
            kv_last_page_len = self._pt_last
            kv_indices = self._pt_indices

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
                entry_stride if self._is_fp8 else 0,
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
