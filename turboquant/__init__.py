"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Near-optimal KV cache quantization for LLM inference, providing 4-6x
compression with minimal quality loss.

Reference: Zandieh et al., arXiv:2504.19874, 2025.
"""

from .quantizer import TurboQuantMSE, TurboQuantProd
from .hadamard import RandomHadamardRotation, fwht
from .codebook import Codebook
from .qjl import QJL

__version__ = "0.1.0"
__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "RandomHadamardRotation",
    "Codebook",
    "QJL",
    "fwht",
]
