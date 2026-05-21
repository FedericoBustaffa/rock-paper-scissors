import numpy as np

from grps import Agent, Grid

if __name__ == "__main__":
    proportions = np.ones(3) / 3
    grid_dims = np.array([10, 10])
    model = Grid(grid_dims, proportions)
    print(model.grid)

    model.step()
