"""TurboQuant KV cache for LLM inference.

Drop-in replacement for HuggingFace's DynamicCache that stores
KV entries in quantized form, reducing memory by 4-6x.

Supports:
  - MSE-optimized quantization (simple, good at 3+ bits)
  - Inner-product-optimized quantization (unbiased, better at low bits)
  - Outlier-aware mixed precision (higher bits for outlier channels)
"""

from __future__ import annotations

from typing import Optional

import torch

from .quantizer import TurboQuantMSE, TurboQuantProd


class QuantizedEntry:
    """Stores a single quantized tensor (one append call)."""

    __slots__ = ("data", "mode")

    def __init__(self, data: tuple, mode: str):
        self.data = data
        self.mode = mode


class TurboQuantCache:
    """Quantized KV cache compatible with HuggingFace transformers.

    Usage with HuggingFace generate():
        cache = TurboQuantCache(bit_width=3, head_dim=128)
        outputs = model.generate(inputs, past_key_values=cache)

    Usage with manual attention loop:
        cache = TurboQuantCache(bit_width=3, head_dim=128)
        for layer_idx, (q, k, v) in enumerate(layer_outputs):
            k_full, v_full = cache.update(k, v, layer_idx)
            attn_out = attention(q, k_full, v_full)
    """

    is_sliding_window = False

    def __init__(
        self,
        head_dim: int,
        bit_width: int = 3,
        mode: str = "mse",
        num_outlier_channels: int = 0,
        outlier_bit_width: Optional[int] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """
        Args:
            head_dim: dimension per attention head
            bit_width: quantization bits per coordinate (1-4)
            mode: 'mse' (Algorithm 1) or 'prod' (Algorithm 2, unbiased)
            num_outlier_channels: number of channels to quantize at higher precision
            outlier_bit_width: bit-width for outlier channels (default: bit_width + 1)
            device: torch device
            seed: random seed for reproducibility
        """
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.mode = mode
        self.device = device
        self.seed = seed

        # Outlier config
        self.num_outlier_ch = num_outlier_channels
        self.outlier_bw = outlier_bit_width or min(bit_width + 1, 4)
        self.num_regular_ch = head_dim - num_outlier_channels
        self.outlier_indices: Optional[torch.Tensor] = None
        self._regular_mask: Optional[torch.Tensor] = None

        # Quantizers (lazily initialized so device matches first tensor)
        self._q_main: Optional[TurboQuantMSE | TurboQuantProd] = None
        self._q_outlier: Optional[TurboQuantMSE | TurboQuantProd] = None

        # Per-layer storage: list of (key_entries, value_entries)
        self.key_cache: list[list[QuantizedEntry]] = []
        self.value_cache: list[list[QuantizedEntry]] = []
        self._seen_tokens = 0

    # ------------------------------------------------------------------ #
    #  Lazy init — first call determines device and dtype                 #
    # ------------------------------------------------------------------ #
    def _ensure_quantizers(self, device: torch.device):
        if self._q_main is not None:
            return
        Cls = TurboQuantProd if self.mode == "prod" else TurboQuantMSE
        if self.num_outlier_ch > 0:
            self._q_main = Cls(self.num_regular_ch, self.bit_width, device=device, seed=self.seed)
            self._q_outlier = Cls(self.num_outlier_ch, self.outlier_bw, device=device, seed=self.seed + 1)
        else:
            self._q_main = Cls(self.head_dim, self.bit_width, device=device, seed=self.seed)
        self.device = device

    # ------------------------------------------------------------------ #
    #  Public API — mirrors HuggingFace Cache interface                   #
    # ------------------------------------------------------------------ #
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize new KV states, append, and return full dequantized cache.

        Args:
            key_states:   [batch, num_heads, seq_len, head_dim]
            value_states: same shape
            layer_idx:    transformer layer index
        Returns:
            (all_keys, all_values) — dequantized, concatenated along seq_len dim
        """
        self._ensure_quantizers(key_states.device)

        # Grow layer lists if needed
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Quantize and store
        self.key_cache[layer_idx].append(self._quantize(key_states))
        self.value_cache[layer_idx].append(self._quantize(value_states))

        # Dequantize all entries for this layer
        all_keys = torch.cat([self._dequantize(e) for e in self.key_cache[layer_idx]], dim=-2)
        all_vals = torch.cat([self._dequantize(e) for e in self.value_cache[layer_idx]], dim=-2)
        return all_keys, all_vals

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache) or not self.key_cache[layer_idx]:
            return 0
        return self._seen_tokens

    def get_max_cache_length(self) -> Optional[int]:
        return None

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        """Return (key, value) for a layer — fully dequantized."""
        if layer_idx >= len(self.key_cache) or not self.key_cache[layer_idx]:
            return (torch.tensor([]), torch.tensor([]))
        keys = torch.cat([self._dequantize(e) for e in self.key_cache[layer_idx]], dim=-2)
        vals = torch.cat([self._dequantize(e) for e in self.value_cache[layer_idx]], dim=-2)
        return keys, vals

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search — not yet supported."""
        raise NotImplementedError("Beam search with TurboQuantCache is not yet supported")

    # ------------------------------------------------------------------ #
    #  Outlier channels                                                   #
    # ------------------------------------------------------------------ #
    def set_outlier_channels(self, indices: torch.Tensor):
        """Specify which head_dim channels are outliers (higher precision)."""
        assert len(indices) == self.num_outlier_ch
        self.outlier_indices = indices
        mask = torch.ones(self.head_dim, dtype=torch.bool)
        mask[indices] = False
        self._regular_mask = mask

    def calibrate_outliers(self, sample_kv: torch.Tensor, top_k: Optional[int] = None):
        """Auto-detect outlier channels from a sample of KV vectors.

        Selects channels with the largest variance across tokens.
        """
        top_k = top_k or self.num_outlier_ch
        if top_k == 0:
            return
        var = sample_kv.float().var(dim=tuple(range(sample_kv.dim() - 1)))
        _, indices = var.topk(top_k)
        self.set_outlier_channels(indices.sort().values)

    # ------------------------------------------------------------------ #
    #  Effective bit-width / compression info                             #
    # ------------------------------------------------------------------ #
    def effective_bit_width(self) -> float:
        if self.num_outlier_ch == 0:
            return float(self.bit_width)
        n_o, n_r = self.num_outlier_ch, self.num_regular_ch
        return (n_o * self.outlier_bw + n_r * self.bit_width) / self.head_dim

    def compression_ratio(self) -> float:
        """Compression vs fp16 (approximate — ignores norm storage overhead)."""
        return 16.0 / self.effective_bit_width()

    # ------------------------------------------------------------------ #
    #  Internal quantize / dequantize                                     #
    # ------------------------------------------------------------------ #
    def _quantize(self, x: torch.Tensor) -> QuantizedEntry:
        if self.num_outlier_ch > 0 and self.outlier_indices is not None:
            return self._quantize_split(x)
        data = self._q_main.quantize(x)
        return QuantizedEntry(data, self.mode)

    def _dequantize(self, entry: QuantizedEntry) -> torch.Tensor:
        if entry.mode == "split":
            return self._dequantize_split(entry)
        return self._q_main.dequantize(*entry.data)

    def _quantize_split(self, x: torch.Tensor) -> QuantizedEntry:
        mask = self._regular_mask.to(x.device)
        x_reg = x[..., mask]
        x_out = x[..., ~mask]
        reg_data = self._q_main.quantize(x_reg)
        out_data = self._q_outlier.quantize(x_out)
        return QuantizedEntry((reg_data, out_data, mask), "split")

    def _dequantize_split(self, entry: QuantizedEntry) -> torch.Tensor:
        reg_data, out_data, mask = entry.data
        x_reg = self._q_main.dequantize(*reg_data)
        x_out = self._q_outlier.dequantize(*out_data)
        out = torch.empty(
            *x_reg.shape[:-1], self.head_dim,
            dtype=x_reg.dtype, device=x_reg.device,
        )
        out[..., mask] = x_reg
        out[..., ~mask] = x_out
        return out

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #
    def clear(self):
        self.key_cache.clear()
        self.value_cache.clear()
        self._seen_tokens = 0
