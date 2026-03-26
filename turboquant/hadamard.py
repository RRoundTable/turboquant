"""Fast Walsh-Hadamard Transform and randomized Hadamard rotation.

The randomized Hadamard rotation maps any worst-case input vector to a
pseudo-random point on the unit sphere, inducing a concentrated Beta
distribution on each coordinate. Runs in O(d log d) time.
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

    x = x.clone()
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

    Forward:  y = FWHT(diag(signs) * pad(x))
    Inverse:  x = unpad(diag(signs) * FWHT(y))
    """

    def __init__(self, dim: int, device: torch.device = "cuda", seed: int = 42):
        self.orig_dim = dim
        self.padded_dim = next_power_of_2(dim)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        signs = torch.sign(torch.randn(self.padded_dim, generator=gen))
        signs[signs == 0] = 1.0
        self.signs = signs.to(device)

    @torch.no_grad()
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward rotation: y = FWHT(signs * pad(x))."""
        d = x.shape[-1]
        if d < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - d))
        x = x * self.signs
        return fwht(x)

    @torch.no_grad()
    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse rotation: x = unpad(signs * FWHT(y))."""
        x = fwht(y)
        x = x * self.signs
        if self.orig_dim < self.padded_dim:
            x = x[..., : self.orig_dim]
        return x
