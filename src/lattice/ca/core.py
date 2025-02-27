from typing import Tuple
import logging

class BaseModel:
    def __init__(self, shape: Tuple[int, int] =(20, 20)) -> None:
        # INFO: type checkings
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(i, int) for i in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self._shape = shape
        logging.debug("hi there")

    def shape(self):
        return self._shape

    def __call__(self):
        pass

    def show(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model = BaseModel(shape=(4, 4))
