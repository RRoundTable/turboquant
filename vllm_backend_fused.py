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
        return "TURBOQUANT"

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
        # Layout: [num_pages, page_size, num_kv_heads, quant_bytes_per_head] uint8
        # Norms: [num_pages, page_size, num_kv_heads, dim_chunks] fp16
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

    def _ensure_quant_storage(self, kv_cache):
        """Allocate separate quantized tensors matching cache dimensions."""
        if self._k_quant is not None:
            return
        # kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
        num_blocks = kv_cache.shape[1]
        block_size = kv_cache.shape[2]
        dev = kv_cache.device
        self._k_quant = torch.zeros(num_blocks, block_size, self.num_kv_heads,
                                     self._quant_bytes_per_head, dtype=torch.uint8, device=dev)
        self._v_quant = torch.zeros_like(self._k_quant)
        self._k_norms = torch.zeros(num_blocks, block_size, self.num_kv_heads,
                                     self._dc, dtype=torch.float16, device=dev)
        self._v_norms = torch.zeros_like(self._k_norms)

    def _fwht(self, x):
        d = x.shape[-1]; x = x.clone(); shape = x.shape; h = 1
        while h < d:
            x = x.view(*shape[:-1], d // (2 * h), 2, h)
            a = x[..., 0, :].clone(); b = x[..., 1, :].clone()
            x[..., 0, :] = a + b; x[..., 1, :] = a - b
            x = x.view(shape); h *= 2
        return x * (1.0 / math.sqrt(d))

    def _hadamard_rotate(self, x):
        d = x.shape[-1]
        if d < self._pd:
            x = F.pad(x, (0, self._pd - d))
        return self._fwht(x * self._signs)

    def _hadamard_inverse(self, y):
        x = self._fwht(y) * self._signs
        if self.head_size < self._pd:
            x = x[..., :self.head_size]
        return x

    def _quantize_dequantize(self, x):
        """Quantize-dequant simulation (for prefill write path + FA read)."""
        self._ensure(x.device)
        xf = x.float()
        norms = xf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = xf / norms
        norms = norms.to(torch.float16).float()
        rotated = self._hadamard_rotate(normalized)
        indices = torch.bucketize(rotated.contiguous(), self._hi_b)
        quantized = self._hi_c[indices]
        reconstructed = self._hadamard_inverse(quantized)
        return (reconstructed * norms).to(x.dtype)

    @torch.no_grad()
    def _quantize_and_store(self, key_or_value, slot_mapping, is_key=True):
        """Quantize and store in separate quantized tensors."""
        self._ensure(key_or_value.device)
        num_tokens = slot_mapping.shape[0]
        x = key_or_value[:num_tokens].float()
        block_size = self._k_quant.shape[1]

        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, heads, 1]
        normalized = x / norms
        norms_fp16 = norms.squeeze(-1).to(torch.float16)  # [N, heads]
        rotated = self._hadamard_rotate(normalized)  # [N, heads, padded_dim]

        quant_store = self._k_quant if is_key else self._v_quant
        norm_store = self._k_norms if is_key else self._v_norms

        for chunk in range(self._dc):
            ds = chunk * TILE_DIMS
            cd = rotated[..., ds:ds + TILE_DIMS]
            indices = torch.bucketize(cd.contiguous(), self._hi_b).to(torch.uint8)
            packed = _pack_4bit(indices)  # [N, heads, 32]

            byte_off = chunk * QUANT_BYTES_PER_CHUNK
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                bid = slot // block_size
                boff = slot % block_size
                quant_store[bid, boff, :, byte_off:byte_off + QUANT_BYTES_PER_CHUNK] = packed[t]
                norm_store[bid, boff, :, chunk] = norms_fp16[t]

    def _get_fused_module(self):
        """JIT-compile the fused decode kernel."""
        if self._fused_module is not None:
            return self._fused_module

        csrc = Path(__file__).parent.parent / "turboquant" / "csrc"
        if not csrc.exists():
            csrc = Path(os.environ.get("TURBOQUANT_CSRC",
                        os.path.expanduser("~/workdir/turboquant/csrc")))

        from turboquant.decode_kernel import _get_module
        self._fused_module = _get_module()
        return self._fused_module

    @torch.no_grad()
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Quantize-dequant for FA cache + store quantized bytes for fused kernel."""
        self._ensure_quant_storage(kv_cache)
        num_tokens = slot_mapping.shape[0]

        # Store quantized bytes in separate tensors (for fused decode kernel)
        self._quantize_and_store(key, slot_mapping, is_key=True)
        self._quantize_and_store(value, slot_mapping, is_key=False)

        # Also store quantize-dequanted fp16 in vLLM cache (for prefill FA)
        key_qd = self._quantize_dequantize(key[:num_tokens])
        value_qd = self._quantize_dequantize(value[:num_tokens])
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(
            key_qd, value_qd, key_cache, value_cache, slot_mapping,
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
        num_actual = attn_metadata.num_actual_tokens
        q = query[:num_actual]  # [N, num_qo_heads, head_dim]
        if output is None:
            output = torch.empty_like(q)

        block_table = attn_metadata.block_table  # [num_reqs, max_blocks]
        seq_lens = attn_metadata.seq_lens         # [num_reqs]
        block_size = self._k_quant.shape[1]

        try:
            module = self._get_fused_module()

            # Build paged KV arrays for the fused kernel
            # The kernel expects: k_quant flat, v_quant flat, k_norms flat, v_norms flat
            # + page indices, indptr, last_page_len
            num_reqs = seq_lens.shape[0]

            # Build indptr and page indices from block_table
            page_indices_list = []
            indptr = [0]
            last_page_lens = []

            for req in range(num_reqs):
                sl = seq_lens[req].item()
                num_pages = (sl + block_size - 1) // block_size
                for p in range(num_pages):
                    page_indices_list.append(block_table[req, p].item())
                indptr.append(indptr[-1] + num_pages)
                last_page_lens.append(sl - (num_pages - 1) * block_size if num_pages > 0 else 0)

            kv_indices = torch.tensor(page_indices_list, dtype=torch.int32, device=q.device)
            kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=q.device)
            kv_last_page_len = torch.tensor(last_page_lens, dtype=torch.int32, device=q.device)

            # Kernel expects HND layout: [num_pages, num_heads, page_size, ...]
            # Our tensors are NHD: [num_pages, page_size, num_heads, ...]
            # Transpose to HND before flattening
            k_q_hnd = self._k_quant.permute(0, 2, 1, 3).contiguous()
            v_q_hnd = self._v_quant.permute(0, 2, 1, 3).contiguous()
            k_n_hnd = self._k_norms.permute(0, 2, 1, 3).contiguous()
            v_n_hnd = self._v_norms.permute(0, 2, 1, 3).contiguous()

            result = module.decode_attention(
                q, k_q_hnd.view(-1),
                v_q_hnd.view(-1),
                k_n_hnd.view(torch.uint8).view(-1).view(torch.float16),
                v_n_hnd.view(torch.uint8).view(-1).view(torch.float16),
                kv_indices, kv_indptr, kv_last_page_len,
                self.num_heads, self.num_kv_heads, block_size,
                self.head_size, self._pd, self.scale,
            )
            output[:num_actual] = result[:num_actual]

        except Exception as e:
            # Fallback to parent FlashAttention if fused kernel fails
            import sys
            print(f"[TQ] Fused kernel failed ({e}), falling back to FA", file=sys.stderr)
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, **kwargs)

        return output
