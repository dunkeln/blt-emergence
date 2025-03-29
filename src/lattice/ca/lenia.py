import torch
from torch import no_grad
import torch.nn.functional as F
from typing import Tuple

from .core import BaseModel

# FIX: lenia is messed up
class Lenia(BaseModel):
    def __init__(self, shape: Tuple[int, int] = (64, 64), device='cpu', dtype=torch.float32) -> None:
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) for i in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.R = 5
        self.T = 10
        with torch.no_grad():
            self.lattice = torch.rand(self.shape, dtype=self.dtype, device=device)
            self.kernel = self._create_kernel(self.R, device, dtype)

    def _create_kernel(self, R, device, dtype):
        K = torch.ones((2 * R + 1, 2 * R + 1), dtype=dtype, device=device)
        K[R, R] = 0
        K /= K.sum()
        return K.unsqueeze(0).unsqueeze(0)

    def _growth(self, U):
        return (U >= 0.12) & (U <= 0.15) - ((U < 0.12) | (U > 0.15))

    @no_grad
    def __next__(self):
        lattice = self.lattice.to(self.dtype).unsqueeze(0).unsqueeze(0)
        toroid = F.pad(lattice, (self.kernel.shape[-1] // 2,) * 4, mode='circular')
        neighbors = F.conv2d(toroid, self.kernel).squeeze()
        new_lattice = torch.clamp(self.lattice + (1 / self.T) * self._growth(neighbors), 0, 1)
        self.lattice = new_lattice
        return self

    def time_steps(self, count=32):
        batch = [next(self).lattice for _ in range(count)]
        batch = torch.stack(batch).unsqueeze(dim=1)
        return batch
