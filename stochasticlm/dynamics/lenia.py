import torch
import torch.nn.functional as F
from typing import Tuple
from .core import BaseModel


class StochasticLenia(BaseModel):
    def __init__(
        self, 
        shape: Tuple[int, int] = (64, 64), 
        device: str = 'cpu', 
        dtype: torch.dtype = torch.float32,
        dt: float = 0.1,
        noise_strength: float = 0.0,
        mu: float = 0.5,
        sigma: float = 0.05,
        kernel_size: int = 7,
        kernel_sigma: float = 0.5
    ) -> None:
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) for i in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.dt = dt
        self.noise_strength = noise_strength
        self.mu = mu
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma

        with torch.no_grad():
            self.lattice = torch.rand(self.shape, dtype=self.dtype, device=device)
            coords = torch.linspace(-1, 1, steps=kernel_size, device=device)
            grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
            d = torch.sqrt(grid_x**2 + grid_y**2)
            kernel = torch.exp(- (d**2) / (2 * (kernel_sigma**2)))
            center = kernel_size // 2
            kernel[center, center] = 0
            kernel = kernel / kernel.sum()
            self.kernel = kernel.to(dtype=self.dtype).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def __next__(self):
        lattice = self.lattice.to(self.dtype).unsqueeze(0).unsqueeze(0)
        pad_size = self.kernel_size // 2
        toroid = F.pad(lattice, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        potential = F.conv2d(toroid, self.kernel).squeeze()
        growth = 2 * torch.exp(- ((potential - self.mu) / self.sigma) ** 2) - 1
        noise = self.noise_strength * torch.randn_like(self.lattice)
        new_lattice = torch.clamp(self.lattice + self.dt * (growth + noise), 0, 1)
        self.lattice = new_lattice
        return self

    def time_steps(self, count: int = 32):
        batch = [next(self).lattice for _ in range(count)]
        batch = torch.stack(batch).unsqueeze(dim=1)
        return batch
