"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Near-optimal KV cache quantization for LLM inference, providing 4-6x
compression with minimal quality loss. Integrates with vLLM and SGLang.

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, 2025.
"""

from .quantizer import TurboQuantMSE, TurboQuantProd
from .kv_cache import TurboQuantCache
from .hadamard import RandomHadamardRotation, fwht
from .codebook import Codebook
from .qjl import QJL

__version__ = "0.1.0"
__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantCache",
    "RandomHadamardRotation",
    "Codebook",
    "QJL",
    "fwht",
]
