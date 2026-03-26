"""Precomputed optimal Lloyd-Max codebooks for TurboQuant.

After a random rotation, each coordinate of a unit vector in R^d follows
a Beta distribution that converges to N(0, 1/d) in high dimensions.
We store optimal Lloyd-Max quantizer centroids for N(0,1) and scale
by 1/sqrt(d) at runtime.

Reference: Lloyd (1982), Max (1960)
"""

import math
import torch

# Optimal Lloyd-Max centroids for N(0, 1), precomputed to high precision.
_CENTROIDS_N01 = {
    1: [
        -0.7978845608028654,
        0.7978845608028654,
    ],
    2: [
        -1.5104176088887660,
        -0.4527800398860679,
        0.4527800398860679,
        1.5104176088887660,
    ],
    3: [
        -2.1519775207392190,
        -1.3439709227384000,
        -0.7560052489261838,
        -0.2451209601065855,
        0.2451209601065855,
        0.7560052489261838,
        1.3439709227384000,
        2.1519775207392190,
    ],
    4: [
        -2.7326368237440100,
        -2.0693470280837980,
        -1.6180344937899930,
        -1.2562490887894980,
        -0.9423683558764893,
        -0.6567591430551046,
        -0.3880823046410021,
        -0.1284153814744918,
        0.1284153814744918,
        0.3880823046410021,
        0.6567591430551046,
        0.9423683558764893,
        1.2562490887894980,
        1.6180344937899930,
        2.0693470280837980,
        2.7326368237440100,
    ],
}


def _midpoints(centroids):
    """Compute decision boundaries as midpoints between consecutive centroids."""
    return [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(len(centroids) - 1)]


_BOUNDARIES_N01 = {b: _midpoints(c) for b, c in _CENTROIDS_N01.items()}


class Codebook:
    """Scalar quantization codebook for TurboQuant.

    Quantizes coordinates of a randomly rotated unit vector using
    optimal Lloyd-Max centroids for the Beta/Gaussian distribution.
    """

    def __init__(self, dim: int, bit_width: int, device: torch.device = None):
        assert 1 <= bit_width <= 4, f"Supported bit-widths: 1-4, got {bit_width}"

        self.dim = dim
        self.bit_width = bit_width
        self.num_levels = 1 << bit_width
        self.scale = 1.0 / math.sqrt(dim)

        # Scaled centroids for N(0, 1/d)
        self.centroids = torch.tensor(
            [c * self.scale for c in _CENTROIDS_N01[bit_width]],
            dtype=torch.float32,
        )

        # Decision boundaries for torch.bucketize (excluding +/-inf)
        self.boundaries = torch.tensor(
            [b * self.scale for b in _BOUNDARIES_N01[bit_width]],
            dtype=torch.float32,
        )

        if device is not None:
            self.centroids = self.centroids.to(device)
            self.boundaries = self.boundaries.to(device)

    def to(self, device):
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)
        return self

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map each value to its nearest centroid index via bucketize.

        Returns uint8 tensor of same shape as x.
        """
        self.boundaries = self.boundaries.to(x.device)
        return torch.bucketize(x, self.boundaries).to(torch.uint8)

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map indices back to centroid values."""
        self.centroids = self.centroids.to(indices.device)
        return self.centroids[indices.long()]
