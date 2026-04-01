"""Kernel configuration and model presets for TurboQuant v4 decode.

Maps model architecture → optimal kernel launch parameters.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    """Model attention architecture."""
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    name: str = ""

    @property
    def gqa_ratio(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    @property
    def padded_dim(self) -> int:
        """Next power of 2 >= head_dim, rounded up to multiple of 64."""
        d = self.head_dim
        # Round up to multiple of 64 (our chunk size)
        d = ((d + 63) // 64) * 64
        return d


@dataclass(frozen=True)
class KernelConfig:
    """CUDA kernel launch parameters."""
    bdx: int       # threads per x-dim (head_dim / vec_size)
    bdy: int       # threads per y-dim (GQA ratio)
    bdz: int       # threads per z-dim (KV token parallelism)
    tile_size_per_bdx: int = 4  # rows per thread per tz group
    vec_size: int = 8           # elements per thread

    @property
    def threads_per_block(self) -> int:
        return self.bdx * self.bdy * self.bdz

    @property
    def tile_tokens(self) -> int:
        return self.tile_size_per_bdx * self.bdy * self.bdz

    @property
    def head_dim(self) -> int:
        return self.bdx * self.vec_size


def compute_kernel_config(model: ModelConfig, max_threads: int = 1024) -> KernelConfig:
    """Compute optimal kernel config for a model architecture.

    Maximizes bdz (occupancy) within the thread limit.
    """
    vec_size = 8
    padded = model.padded_dim
    bdx = padded // vec_size
    bdy = model.gqa_ratio

    # Maximize bdz within thread limit
    max_bdz = max_threads // (bdx * bdy)
    # Cap at 64 — diminishing returns beyond 16, and smem grows with bdz
    bdz = min(max_bdz, 64)
    # Ensure at least 1
    bdz = max(bdz, 1)

    return KernelConfig(bdx=bdx, bdy=bdy, bdz=bdz)


# ═══ Model presets ═══

MODEL_PRESETS: Dict[str, ModelConfig] = {
    # Llama family
    "llama-2-7b":   ModelConfig(32, 32, 128, "Llama-2-7B"),
    "llama-2-13b":  ModelConfig(40, 40, 128, "Llama-2-13B"),
    "llama-2-70b":  ModelConfig(64,  8, 128, "Llama-2-70B"),
    "llama-3-8b":   ModelConfig(32,  8, 128, "Llama-3-8B"),
    "llama-3-70b":  ModelConfig(64,  8, 128, "Llama-3-70B"),
    "llama-3.1-8b": ModelConfig(32,  8, 128, "Llama-3.1-8B"),
    "llama-3.1-70b":ModelConfig(64,  8, 128, "Llama-3.1-70B"),

    # Qwen family
    "qwen3-0.6b":  ModelConfig(16, 8,  64,  "Qwen3-0.6B"),
    "qwen3-1.7b":  ModelConfig(16, 8,  128, "Qwen3-1.7B"),
    "qwen3-4b":    ModelConfig(32, 8,  128, "Qwen3-4B"),
    "qwen3-8b":    ModelConfig(32, 8,  128, "Qwen3-8B"),
    "qwen3-14b":   ModelConfig(40, 8,  128, "Qwen3-14B"),
    "qwen3-32b":   ModelConfig(40, 8,  128, "Qwen3-32B"),

    # Mistral / Mixtral
    "mistral-7b":   ModelConfig(32, 8, 128, "Mistral-7B"),
    "mixtral-8x7b": ModelConfig(32, 8, 128, "Mixtral-8x7B"),
    "mistral-nemo": ModelConfig(32, 8, 128, "Mistral-Nemo-12B"),

    # Phi
    "phi-3-mini":   ModelConfig(32, 32, 96,  "Phi-3-mini"),
    "phi-3-small":  ModelConfig(32,  8, 96,  "Phi-3-small"),

    # Gemma
    "gemma-2-2b":   ModelConfig( 8, 4, 256, "Gemma-2-2B"),
    "gemma-2-9b":   ModelConfig(16, 8, 256, "Gemma-2-9B"),

    # Yi
    "yi-6b":        ModelConfig(32, 4, 128, "Yi-6B"),
    "yi-34b":       ModelConfig(56, 8, 128, "Yi-34B"),
}


def print_config_table():
    """Print kernel config for all model presets."""
    print(f"{'Model':<20} {'QO':>3} {'KV':>3} {'HD':>4} {'GQA':>4} "
          f"{'bdx':>4} {'bdy':>4} {'bdz':>4} {'threads':>7} {'tile':>5}")
    print("-" * 72)
    for name, model in sorted(MODEL_PRESETS.items()):
        kc = compute_kernel_config(model)
        print(f"{model.name:<20} {model.num_qo_heads:>3} {model.num_kv_heads:>3} "
              f"{model.head_dim:>4} {model.gqa_ratio:>3}:1 "
              f"{kc.bdx:>4} {kc.bdy:>4} {kc.bdz:>4} {kc.threads_per_block:>7} {kc.tile_tokens:>5}")


if __name__ == "__main__":
    print_config_table()
