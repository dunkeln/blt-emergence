import torch
import random
from typing import Tuple
from .core import BaseModel

class MCSTLattice(BaseModel):
    def __init__(
        self, 
        shape: Tuple[int, int] = (30, 30), 
        device: str = 'cpu', 
        dtype=torch.int64,
        density: float = 0.1,
        D: float = 0.1,           # Rotation rate (per move attempt)
        v_plus: float = 10.0,     # Translation rate for a forward move
        v_zero: float = 1.0       # Translation rate for lateral moves
    ) -> None:
        # Type check for shape
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(x, int) for x in shape)):
            raise TypeError(f"Expected shape to be a tuple of 2 integers, got {shape}")

        self.shape = shape
        self.dtype = dtype
        self.device = device

        # Physical parameters
        self.density = density  # fraction of sites initially occupied
        self.D = D              # rotational (angular) move rate
        self.v_plus = v_plus    # translation rate if moving forward (in the particle's orientation)
        self.v_zero = v_zero    # translation rate if moving in any other direction

        # Initialize the lattice.
        # We use an integer tensor where:
        #   0   -> empty site
        #   1-4 -> particle with one of 4 orientations:
        #         1: Up, 2: Right, 3: Down, 4: Left.
        total_cells = shape[0] * shape[1]
        r = torch.rand(total_cells, device=device)
        occupancy = (r < density)
        lattice_flat = torch.zeros(total_cells, dtype=self.dtype, device=device)
        num_particles = int(occupancy.sum().item())
        if num_particles > 0:
            # For each occupied site, assign a random orientation from 1 to 4.
            lattice_flat[occupancy] = torch.randint(1, 5, (num_particles,), dtype=self.dtype, device=device)
        self.lattice = lattice_flat.reshape(shape)

    @torch.no_grad()
    def __next__(self):
        # Perform one Monte Carlo sweep:
        # Gather the coordinates of all occupied sites (particles).
        particle_coords = torch.nonzero(self.lattice != 0, as_tuple=False).tolist()
        random.shuffle(particle_coords)

        # Mapping for the four principal directions:
        # 0: Up (-1,  0)
        # 1: Right (0, +1)
        # 2: Down (+1, 0)
        # 3: Left (0, -1)
        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        # Loop over each particle (in random order) and attempt an update.
        for coord in particle_coords:
            i, j = coord
            state = self.lattice[i, j].item()  # Value in {1,2,3,4}
            if state == 0:
                continue  # This particle may already have moved.
            orientation = state - 1  # Convert to 0-indexed: 0=Up, 1=Right, 2=Down, 3=Left.

            actions = []
            rates = []
            # Rotation moves: always allowed.
            actions.append(('rotate', 'cw'))   # Clockwise rotation.
            rates.append(self.D)
            actions.append(('rotate', 'ccw'))  # Counterclockwise rotation.
            rates.append(self.D)

            # Translation moves: check each neighbor (with periodic boundaries).
            for d in range(4):
                di, dj = directions[d]
                ni = (i + di) % self.shape[0]
                nj = (j + dj) % self.shape[1]
                # Allow translation only if the neighbor is empty.
                if self.lattice[ni, nj].item() == 0:
                    actions.append(('translate', (ni, nj), d))
                    # If moving in the direction the particle is facing, assign rate v_plus; otherwise, v_zero.
                    if d == orientation:
                        rates.append(self.v_plus)
                    else:
                        rates.append(self.v_zero)

            # If no moves are available (can happen if a particle is fully blocked), skip this particle.
            total_rate = sum(rates)
            if total_rate == 0:
                continue

            # Select one move with probability proportional to its rate.
            r = random.uniform(0, total_rate)
            cumulative = 0.0
            chosen_action = None
            for act, rate in zip(actions, rates):
                cumulative += rate
                if r < cumulative:
                    chosen_action = act
                    break

            # Execute the chosen action.
            if chosen_action is None:
                continue
            if chosen_action[0] == 'rotate':
                # Update orientation based on rotation direction.
                if chosen_action[1] == 'cw':
                    new_orientation = (orientation + 1) % 4
                elif chosen_action[1] == 'ccw':
                    new_orientation = (orientation - 1) % 4
                else:
                    raise ValueError(f"Unexpected rotation direction: {chosen_action[1]}")
                self.lattice[i, j] = new_orientation + 1  # Convert back to 1-indexed.
            elif chosen_action[0] == 'translate':
                # Attempt to move the particle: check again that the target remains empty.
                target = chosen_action[1]
                ni, nj = target
                if self.lattice[ni, nj].item() == 0:
                    self.lattice[i, j] = 0  # Vacate original cell.
                    self.lattice[ni, nj] = state  # Keep the orientation the same.
        return self

    def time_steps(self, count: int = 32):
        # Record successive lattice configurations.
        batch = [next(self).lattice.clone() for _ in range(count)]
        batch = torch.stack(batch).unsqueeze(dim=1)  # Optionally add a channel dimension.
        return batch
