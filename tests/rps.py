import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from grps import Genetic, Inheritance, RPSModel

if __name__ == "__main__":
    policies = {
        "rock": Inheritance(),
        "paper": Inheritance(),
        "scissors": Genetic(sigma=0.01),
    }
    model = RPSModel(dim=25, policies=policies, rng=9951)

    n_epochs = 200
    model.run_for(n_epochs)

    model_df = model.datacollector.get_model_vars_dataframe()
    print(model_df)

    # plot population density
    plt.figure(dpi=200)
    plt.title("Population Density")

    plt.plot(model_df["R_density"], c="red", label="rock")
    plt.plot(model_df["P_density"], c="blue", label="paper")
    plt.plot(model_df["S_density"], c="green", label="scissors")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot invasion rate
    plt.figure(dpi=200)
    plt.title("Average Invasion Rate")

    plt.plot(model_df["R_invasion"], c="red", label="rock")
    plt.plot(model_df["P_invasion"], c="blue", label="paper")
    plt.plot(model_df["S_invasion"], c="green", label="scissors")
    plt.plot([0, n_epochs], [1.0, 1.0], "r--", label="maximal invasion")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot average age
    plt.figure(dpi=200)
    plt.title("Average Age")

    plt.plot(model_df["R_age"], c="red", label="rock")
    plt.plot(model_df["P_age"], c="blue", label="paper")
    plt.plot(model_df["S_age"], c="green", label="scissors")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot the grid
    grid = np.full((model.grid.width, model.grid.height), -1)
    color_map = {
        "rock": 0,
        "paper": 1,
        "scissors": 2,
    }

    for agent in model.agents:  # oppure model.schedule.agents
        x, y = agent.cell.coordinate  # dipende dalla tua implementazione
        grid[x, y] = color_map[agent.specie]

    plt.figure(dpi=200)
    plt.title("Final Grid State")
    cmap = ListedColormap(["red", "blue", "green"])
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    plt.colorbar(ticks=[0, 1, 2], label="species")
    plt.tight_layout()
    plt.show()
