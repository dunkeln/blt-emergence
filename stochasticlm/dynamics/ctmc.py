import torch
import random
import math
from typing import Tuple
from torch import no_grad
from .core import BaseModel

class CTMCActiveLattice(BaseModel):
    def __init__(
        self,
        shape: Tuple[int,int] = (20,20),
        device: str = 'cpu',
        dtype = torch.long,
        density: float = 0.1,
        D: float = 0.1,
        v_plus: float = 10.0,
        v_zero: float = 1.0,
    ) -> None:
        super().__init__(shape, device=device, dtype=dtype)
        # physical rates
        self.density = density
        self.D = D
        self.v_plus = v_plus
        self.v_zero = v_zero

        # global clock
        self.time = 0.0

        # initialize lattice: 0=empty, 1–4=orientation
        total = shape[0] * shape[1]
        occ = (torch.rand(total, device=device) < density)
        flat = torch.zeros(total, dtype=self.dtype, device=self.device)
        n = int(occ.sum().item())
        if n > 0:
            flat[occ] = torch.randint(1, 5, (n,), dtype=self.dtype, device=self.device)
        self.lattice = flat.view(self.shape)

        # neighbor offsets for Up,Right,Down,Left
        self._dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @no_grad()
    def __next__(self):
        H, W = self.shape
        events = []
        rates = []

        # collect all rotation + translation events
        for i, j in torch.nonzero(self.lattice, as_tuple=False).tolist():
            ori = int(self.lattice[i, j]) - 1  # 0–3

            # rotations
            events.append(('rotate', (i, j, 'cw')))
            rates.append(self.D)
            events.append(('rotate', (i, j, 'ccw')))
            rates.append(self.D)

            # translations
            for d, (di, dj) in enumerate(self._dirs):
                ni, nj = (i + di) % H, (j + dj) % W
                if self.lattice[ni, nj] == 0:
                    rate = self.v_plus if d == ori else self.v_zero
                    events.append(('translate', (i, j, ni, nj)))
                    rates.append(rate)

        if not rates:
            return self  # nothing to do

        # pick one event by rate
        R_tot = sum(rates)
        r = random.random() * R_tot
        cum = 0.0
        for idx, rate in enumerate(rates):
            cum += rate
            if r < cum:
                kind, args = events[idx]
                break
        else:
            kind, args = events[-1]

        # execute it
        if kind == 'rotate':
            i, j, direction = args
            ori = int(self.lattice[i, j]) - 1
            delta = 1 if direction == 'cw' else -1
            self.lattice[i, j] = ((ori + delta) % 4) + 1
        else:  # translate
            i, j, ni, nj = args
            state = int(self.lattice[i, j])
            # double-check empty
            if self.lattice[ni, nj] == 0:
                self.lattice[ni, nj] = state
                self.lattice[i, j] = 0

        # advance time
        U = random.random()
        dt = -math.log(U) / R_tot
        self.time += dt

        return self

    def time_steps(self, count: int = 32):
        """
        Generate `count` successive CTMC states.
        Returns a Tensor of shape (count, 1, H, W).
        """
        series = []
        for _ in range(count):
            next(self)
            series.append(self.lattice.clone())
        return torch.stack(series).unsqueeze(dim=1)
