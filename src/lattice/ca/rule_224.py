import numpy as np
from typing import Tuple
from .core import BaseModel


class Rule_224(BaseModel):
    """
    Rules:
        Any live cell with fewer than two live neighbours dies (referred to as underpopulation or exposure[2]).
        Any live cell with more than three live neighbours dies (referred to as overpopulation or overcrowding).
        Any live cell with two or three live neighbours lives, unchanged, to the next generation.
        Any dead cell with exactly three live neighbours will come to life.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lattice = np.random.choice([0, 1], self.shape())

    def count_neighbors(self, x, y):
        """Count the number of live neighbors around cell (x, y)"""
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        count = 0
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self._shape[0] and 0 <= ny < self._shape[1]:
                count += self.lattice[nx, ny]
        return count

    def __next__(self):
        """Compute the next state of the cellular automaton"""
        new_lattice = np.copy(self.lattice)

        for x in range(self._shape[0]):
            for y in range(self._shape[1]):
                live_neighbors = self.count_neighbors(x, y)

                if self.lattice[x, y] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_lattice[x, y] = 0
                else:
                    if live_neighbors == 3:
                        new_lattice[x, y] = 1

        self.lattice = new_lattice
        return self

    def show(self):
        """Print the current state of the lattice"""
        print("\n".join("".join("█" if cell else "." for cell in row) for row in self.lattice))
        print("\n")
