import torch
from torch import no_grad
import torch.nn.functional as F
from typing import Tuple
import logging

class BaseModel:
    def __init__(self, shape: Tuple[int, int] =(20, 20), device='cpu', dtype=torch.bfloat16) -> None:
        # INFO: type checkings
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) for i in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self.shape = shape
        self.dtype = dtype
        with torch.no_grad():
            self.lattice = torch.randint(0, 2, self.shape, dtype=self.dtype, device=device)
            self.kernel = torch.tensor([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ], dtype=self.dtype, device=device).unsqueeze(0).unsqueeze(0)

    @no_grad
    def __next__(self):
        lattice = self.lattice.to(self.dtype).unsqueeze(0).unsqueeze(0)
        toroid = F.pad(lattice, (1, 1, 1, 1), mode='circular')
        neighbors = F.conv2d(toroid, self.kernel).squeeze()
        new_lattice = ((neighbors == 3) | ((self.lattice == 1) & (neighbors == 2))).to(self.dtype)
        self.lattice = new_lattice
        return self

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model = BaseModel(shape=(4, 4))
