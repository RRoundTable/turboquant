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


class TurboQuantMetadataBuilder(FlashAttentionMetadataBuilder):
    pass


class TurboQuantFusedImpl(FlashAttentionImpl):
    """TurboQuant with fused CUDA decode kernel.

    Maintains separate quantized KV tensors for the fused decode kernel.
    Prefill uses FlashAttention (parent) with quantize-dequant simulation.
    Decode uses the fused CUDA kernel via JIT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pd = _next_pow2(self.head_size)
        self._dc = self._pd // TILE_DIMS
        self._quant_bytes_per_head = self._dc * QUANT_BYTES_PER_CHUNK
        self._hi_b = None
        self._hi_c = None
        self._signs = None
        self._fused_module = None

        # Separate quantized storage (allocated on first use)
        # HND Layout: [num_pages, num_kv_heads, page_size, quant_bytes_per_head] uint8
        # Norms: [num_pages, num_kv_heads, page_size, dim_chunks] fp16
        self._k_quant = None
        self._v_quant = None
        self._k_norms = None
        self._v_norms = None

    def _ensure(self, dev):
        if self._hi_b is not None:
            return
        s = 1.0 / math.sqrt(self._pd)
        self._hi_b = torch.tensor([b*s for b in _B4], device=dev, dtype=torch.float32)
        self._hi_c = torch.tensor([c*s for c in _C4], device=dev, dtype=torch.float32)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        signs = torch.sign(torch.randn(self._pd, generator=gen))
        signs[signs == 0] = 1.0
        self._signs = signs.to(dev)
        # Pre-compute dense Hadamard matrix for fast matmul rotation.
        # H is symmetric and orthogonal: H @ H = I (when normalized).
        d = self._pd
        H = torch.eye(d, device='cpu', dtype=torch.float32)
        h = 1
        while h < d:
            H = H.view(d, d // (2 * h), 2, h)
            a = H[..., 0, :].clone(); b = H[..., 1, :].clone()
            H[..., 0, :] = a + b; H[..., 1, :] = a - b
            H = H.view(d, d)
            h *= 2
        self._H = (H * (1.0 / math.sqrt(d))).to(dev)

    def _ensure_quant_storage(self, kv_cache):
        """Allocate separate quantized tensors matching cache dimensions."""
        if self._k_quant is not None:
            return
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        dev = kv_cache.device
        self._k_quant = torch.zeros(num_blocks, self.num_kv_heads, block_size,
                                     self._quant_bytes_per_head, dtype=torch.uint8, device=dev)
        self._v_quant = torch.zeros_like(self._k_quant)
        self._k_norms = torch.zeros(num_blocks, self.num_kv_heads, block_size,
                                     self._dc, dtype=torch.float16, device=dev)
        self._v_norms = torch.zeros_like(self._k_norms)

    def _hadamard_rotate(self, x):
        d = x.shape[-1]
        if d < self._pd:
            x = F.pad(x, (0, self._pd - d))
        return torch.matmul(x * self._signs, self._H)

    def _hadamard_inverse(self, y):
        x = torch.matmul(y, self._H) * self._signs
        if self.head_size < self._pd:
            x = x[..., :self.head_size]
        return x

    @torch.no_grad()
    def _quantize_and_store(self, key_or_value, slot_mapping, is_key=True):
        """Quantize and store in HND layout: [pages, heads, entries, bytes].

        Uses CUDA write kernel when available, falls back to Python.
        """
        self._ensure(key_or_value.device)
        num_tokens = slot_mapping.shape[0]
        x = key_or_value[:num_tokens].float()
        block_size = self._k_quant.shape[2]

        # Normalize and rotate
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = x / norms
        rotated = self._hadamard_rotate(normalized)

        # Prepare input for CUDA kernel: rotated * norm (kernel re-normalizes)
        kv_ready = (rotated * norms).to(torch.float16)

        quant_store = self._k_quant if is_key else self._v_quant
        norm_store = self._k_norms if is_key else self._v_norms

        try:
            write_mod = self._get_write_module()
            # CUDA kernel: normalize + quantize + pack → [tokens, heads, bytes]
            quant_out, norms_out = write_mod.quantize_write(
                kv_ready, self.head_size, self._pd)

            # Scatter to paged HND layout using slot_mapping
            bids = slot_mapping // block_size
            boffs = slot_mapping % block_size
            quant_store[bids, :, boffs, :] = quant_out
            norm_store[bids, :, boffs, :] = norms_out
        except Exception:
            # Fallback: Python quantize path
            norms_fp16 = norms.squeeze(-1).to(torch.float16)
            for chunk in range(self._dc):
                ds = chunk * TILE_DIMS
                cd = rotated[..., ds:ds + TILE_DIMS]
                indices = torch.bucketize(cd.contiguous(), self._hi_b).to(torch.uint8)
                packed = _pack_4bit(indices)

                byte_off = chunk * QUANT_BYTES_PER_CHUNK
                for t in range(num_tokens):
                    slot = slot_mapping[t].item()
                    bid = slot // block_size
                    boff = slot % block_size
                    quant_store[bid, :, boff, byte_off:byte_off + QUANT_BYTES_PER_CHUNK] = packed[t]
                    norm_store[bid, :, boff, chunk] = norms_fp16[t]

    def _get_fused_module(self):
        """JIT-compile the v4 fused decode kernel."""
        if self._fused_module is not None:
            return self._fused_module

        from turboquant.decode_kernel_v4 import _get_module
        self._fused_module = _get_module()
        return self._fused_module

    def _get_write_module(self):
        """JIT-compile the CUDA quantize-write kernel."""
        if not hasattr(self, '_write_module') or self._write_module is None:
            from turboquant.decode_kernel_v4 import _CSRC_DIR

            from torch.utils.cpp_extension import load
            self._write_module = load(
                name="turboquant_write",
                sources=[str(_CSRC_DIR / "src" / "quantize_write_binding.cu")],
                extra_include_paths=[str(_CSRC_DIR / "include")],
                extra_cuda_cflags=[
                    "-std=c++17", "-O3", "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
                verbose=False,
            )
        return self._write_module

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Quantize-dequant for FA cache + store quantized bytes for fused kernel."""
        self._ensure_quant_storage(kv_cache)
        num_tokens = slot_mapping.shape[0]

        # Store quantized bytes in separate tensors (for fused decode kernel)
        self._quantize_and_store(key, slot_mapping, is_key=True)
        self._quantize_and_store(value, slot_mapping, is_key=False)

        # Store ORIGINAL fp16 K,V in vLLM cache (for prefill FlashAttention).
        # No quantize-dequant: prefill uses exact values, decode uses quantized.
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
            # Prefill: use parent FlashAttention (reads fp16 from vLLM cache)
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        # === DECODE: fused CUDA kernel ===
        self._ensure(query.device)
        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]
        if output is None:
            output = torch.empty_like(q)

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        block_size = self._k_quant.shape[2]  # HND: [pages, heads, entries, bytes]

        try:
            module = self._get_fused_module()
            num_reqs = seq_lens.shape[0]

            # Build page table on GPU (no Python loops, no CPU→GPU copies)
            num_pages_per_req = (seq_lens + block_size - 1) // block_size
            total_pages = num_pages_per_req.sum().item()
            kv_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=q.device)
            torch.cumsum(num_pages_per_req, dim=0, out=kv_indptr[1:])
            kv_last_page_len = (seq_lens - (num_pages_per_req - 1) * block_size).to(torch.int32)

            # Gather page indices from block_table (GPU tensor op)
            page_offsets = torch.arange(block_table.shape[1], device=q.device)
            mask = page_offsets.unsqueeze(0) < num_pages_per_req.unsqueeze(1)
            kv_indices = block_table[mask].to(torch.int32)

            # Pad Q to padded_dim if needed (kernel expects padded_dim)
            q_fp16 = q.to(torch.float16)
            if self.head_size < self._pd:
                q_fp16 = F.pad(q_fp16, (0, self._pd - self.head_size))

            # v4 kernel with fused Hadamard: rotates Q, computes attention,
            # un-rotates output — all inside the kernel via warp shuffles.
            result = module.decode_v4(
                q_fp16,
                self._k_quant.view(-1),
                self._v_quant.view(-1),
                self._k_norms.view(-1),
                self._v_norms.view(-1),
                kv_indices, kv_indptr, kv_last_page_len,
                self.num_kv_heads, block_size,
                self.head_size, self._pd, self.scale,
                self._signs,
            )

            # Output is already un-rotated by the kernel
            if self.head_size < self._pd:
                output[:num_actual] = result[:num_actual, :, :self.head_size]
            else:
                output[:num_actual] = result[:num_actual]

        except Exception as e:
            import sys
            print(f"[TQ] Fused kernel failed ({e}), falling back to FA", file=sys.stderr)
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        return output
