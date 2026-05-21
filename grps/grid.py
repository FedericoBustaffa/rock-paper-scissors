from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from grps import Agent


class Grid:
    def __init__(
        self,
        grid_dims: np.ndarray,
        proportions: np.ndarray,
        epoch_length: int = 100,
    ) -> None:
        # ensure valid shape and probabilities
        assert proportions.shape == (3,) and sum(proportions) == 1.0

        # build the grid
        self.grid = np.random.choice([0, 1, 2], size=grid_dims, p=proportions)

        self.epoch_length = epoch_length

    def step(self) -> None:
        """Perform one epoch"""
        for e in range(self.epoch_length):
            # get individual1
            x = np.random.choice(self.grid.shape[0])
            y = np.random.choice(self.grid.shape[1])
            ind1 = self.grid[x][y]

            # get individual2
            xoffset = np.random.choice([-1, 0, 1])
            yoffset = np.random.choice([-1, 0, 1])
            ind2 = self.grid[x + xoffset][y + yoffset]
