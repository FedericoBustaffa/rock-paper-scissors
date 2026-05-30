import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from grps import EvolutionPolicy, Genetic, RPSModel, Stochastic
from grps.evolution_policies import Stochastic

if __name__ == "__main__":
    # CLI arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, help="square grid side dimension")
    parser.add_argument(
        "sigma",
        type=float,
        help="standard deviation of the gaussian producing mutations",
    )
    parser.add_argument("radius", type=int, help="distance radius for mating")
    parser.add_argument("epochs", type=int, help="simulation number of epochs")
    parser.add_argument("seed", type=int, help="random seed for reproducibility")
    args = parser.parse_args()

    policies: dict[str, EvolutionPolicy] = {
        "rock": Genetic(sigma=args.sigma, radius=args.radius),
        "paper": Stochastic(sigma=args.sigma),
        "scissors": Stochastic(sigma=args.sigma),
    }
    model = RPSModel(
        dim=args.dim,
        initial_invasions=[0.5, 0.5, 0.5],
        policies=policies,
        rng=args.seed,
    )

    model.run_for(args.epochs)
    df = model.datacollector.get_model_vars_dataframe()

    # Population density
    plt.figure(dpi=150)
    plt.title("Population Density")
    plt.plot(df["R_density"], c="red", label="rock")
    plt.plot(df["P_density"], c="blue", label="paper")
    plt.plot(df["S_density"], c="green", label="scissors")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("population_density.png")
    plt.show()

    # Average invasion rate
    plt.figure(dpi=150)
    plt.title("Average Invasion Rate")
    plt.plot(df["R_invasion"], c="red", label="rock")
    plt.plot(df["P_invasion"], c="blue", label="paper")
    plt.plot(df["S_invasion"], c="green", label="scissors")
    plt.axhline(1.0, ls="--", c="gray", label="max invasion")
    plt.xlabel("Epoch")
    plt.ylabel("Mean invasion probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("invasion_rate.png")
    plt.show()

    # Average age
    plt.figure(dpi=150)
    plt.title("Average Age")
    plt.plot(df["R_age"], c="red", label="rock")
    plt.plot(df["P_age"], c="blue", label="paper")
    plt.plot(df["S_age"], c="green", label="scissors")
    plt.xlabel("Epoch")
    plt.ylabel("Mean age (epochs)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("average_age.png")
    plt.show()

    # Final grid state
    color_map = {"rock": 0, "paper": 1, "scissors": 2}
    grid_arr = np.full((model.grid.width, model.grid.height), -1)
    for agent in model.agents:
        x, y = agent.cell.coordinate
        grid_arr[x, y] = color_map[agent.specie]

    plt.figure(dpi=150)
    plt.title("Final Grid State")
    cmap = ListedColormap(["red", "blue", "green"])
    im = plt.imshow(grid_arr.T, cmap=cmap, interpolation="nearest", origin="lower")
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["rock", "paper", "scissors"])
    plt.tight_layout()
    plt.savefig("final_grid.png")
    plt.show()
