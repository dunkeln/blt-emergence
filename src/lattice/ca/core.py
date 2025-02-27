import torch
import torch.nn.functional as f
from typing import Tuple
import logging

class BaseModel:
    def __init__(self, shape: Tuple[int, int] =(20, 20)) -> None:
        # INFO: type checkings
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) for i in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self.shape = shape
        self.lattice = torch.randint(0, 2, self.shape, dtype=torch.int8)
        self.kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=torch.int8)
        self.sum_kernel = self.kernel.sum()

    def __next__(self):
        ker = self.kernel.unsqueeze(0).unsqueeze(0)
        return self

    def __call__(self):
        pass

    def show(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model = BaseModel(shape=(4, 4))
