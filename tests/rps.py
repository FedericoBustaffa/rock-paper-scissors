import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

from grps import RPSAgent, RPSModel

if __name__ == "__main__":
    model = RPSModel(100, 10, 10, rng=42)
    model.run_for(100)

    df = model.datacollector.get_model_vars_dataframe()

    # population density trend
    plt.figure(dpi=200)
    densities = df[["rock_density", "paper_density", "scissors_density"]]
    g = sns.lineplot(data=densities, dashes=False)
    g.set(xlabel="Epoch", ylabel="Population Density")
    plt.tight_layout()
    plt.show()

    # invasion rate trend
    plt.figure(dpi=200)
    invasions = df[["rock_invasion", "paper_invasion", "scissors_invasion"]]
    print(invasions)
    g = sns.lineplot(data=invasions, dashes=False)
    g.set(xlabel="Epoch", ylabel="Invasion Rate")
    plt.tight_layout()
    plt.show()
