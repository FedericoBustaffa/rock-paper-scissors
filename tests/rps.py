import matplotlib.pyplot as plt

from grps import RPSModel

if __name__ == "__main__":
    model = RPSModel(50, 50, rng=42)
    model.run_for(1000)

    model_df = model.datacollector.get_model_vars_dataframe()
    print(model_df)

    plt.figure(dpi=200)
    plt.title("Population Density")

    plt.plot(model_df["R_density"], label="rock")
    plt.plot(model_df["P_density"], label="paper")
    plt.plot(model_df["S_density"], label="scissors")

    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(dpi=200)
    plt.title("Average Invasion Rate")

    plt.plot(model_df["R_invasion"], label="rock")
    plt.plot(model_df["P_invasion"], label="paper")
    plt.plot(model_df["S_invasion"], label="scissors")

    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(dpi=200)
    plt.title("Average Age")

    plt.plot(model_df["R_age"], label="rock")
    plt.plot(model_df["P_age"], label="paper")
    plt.plot(model_df["S_age"], label="scissors")

    plt.grid()
    plt.tight_layout()
    plt.show()
