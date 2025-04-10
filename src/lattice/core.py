import torch
from torch import no_grad
from typing import Tuple

class BaseModel:
    def __init__(self, shape: Tuple[int, int] =(20, 20), device='cpu', dtype=torch.float16) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.lattice = torch.zeros(self.shape, dtype=self.dtype, device=device)

    @no_grad
    def __next__(self):
        raise NotImplementedError("method `__next__` not implemented")

    def time_steps(self):
        raise NotImplementedError("method `get_series` not implemented")

    def byte_encode(self):
        flat_tensor = self.lattice.flatten()
        encoded = flat_tensor.numpy().tobytes()
        return encoded

    def byte_decode(self, stream: bytes):
        return torch.frombuffer(stream, dtype=self.dtype).clone().reshape(*self.shape)

    def size(self, idx = None):
        if idx is None:
            return self.lattice.size()
        else:
            return self.lattice.size(idx)
