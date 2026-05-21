from typing import Sequence

import mesa
import numpy as np

from grps import Specie


class Grid(mesa.Model):
    def __init__(
        self,
        grid_dims: np.ndarray,
        proportions: np.ndarray,
    ) -> None:
        # ensure valid shape and probabilities
        assert proportions.shape == (3,) and sum(proportions) == 1.0

        # build the grid
        self.grid = np.random.choice([0, 1, 2], size=grid_dims, p=proportions)

    def step(self, n: int) -> None:  # ty:ignore[invalid-method-override]
        """Perform one epoch"""
        for e in range(n):
            # get individual1
            x = np.random.choice(self.grid.shape[0])
            y = np.random.choice(self.grid.shape[1])
            ind1 = self.grid[x][y]

            # get individual2 from neighborhood of ind1
            xoffset = np.random.choice([-1, 0, 1])
            yoffset = np.random.choice([-1, 0, 1])
            ind2 = self.grid[x + xoffset][y + yoffset]

            # predation
