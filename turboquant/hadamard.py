"""Fast Walsh-Hadamard Transform and randomized Hadamard rotation.

The randomized Hadamard rotation is used in TurboQuant to map any
worst-case input vector to a pseudo-random point on the unit sphere,
inducing a concentrated Beta distribution on each coordinate.

This runs in O(d log d) time, compared to O(d^2) for a full random
rotation matrix, making it efficient for GPU-accelerated inference.
"""

import math
import torch
import torch.nn.functional as F


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Fast Walsh-Hadamard Transform along the last dimension.

    Computes y = (1/sqrt(d)) * H * x where H is the Hadamard matrix.
    The last dimension must be a power of 2.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Last dim must be power of 2, got {d}"

    orig_shape = x.shape
    h = 1
    while h < d:
        x = x.view(*orig_shape[:-1], d // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(orig_shape)
        h *= 2

    return x * (1.0 / math.sqrt(d))


class RandomHadamardRotation:
    """Pseudo-random orthogonal rotation using randomized Hadamard transform.

    Forward rotation:  y = FWHT(diag(signs) * pad(x))
    Inverse rotation:  x = unpad(diag(signs) * FWHT(y))

    This is equivalent to multiplying by a random orthogonal matrix
    but runs in O(d log d) instead of O(d^2).
    """

    def __init__(self, dim: int, device: torch.device = None, seed: int = 42):
        self.orig_dim = dim
        self.padded_dim = next_power_of_2(dim)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = torch.sign(torch.randn(self.padded_dim, generator=gen))
        signs[signs == 0] = 1.0
        self.signs = signs

        if device is not None:
            self.signs = self.signs.to(device)

    def to(self, device):
        self.signs = self.signs.to(device)
        return self

    @torch.no_grad()
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward rotation: y = FWHT(signs * pad(x))."""
        self.signs = self.signs.to(x.device)
        d = x.shape[-1]
        if d < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - d))
        x = x * self.signs
        return fwht(x)

    @torch.no_grad()
    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse rotation: x = unpad(signs * FWHT(y))."""
        self.signs = self.signs.to(y.device)
        x = fwht(y)
        x = x * self.signs
        if self.orig_dim < self.padded_dim:
            x = x[..., : self.orig_dim]
        return x
