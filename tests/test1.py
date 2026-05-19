import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mesa import Agent, DataCollector, Model
from mesa.space import SingleGrid

ROCK = 0
SCISSORS = 1
PAPER = 2


def dominates(a, b):
    return (
        (a == ROCK and b == SCISSORS)
        or (a == SCISSORS and b == PAPER)
        or (a == PAPER and b == ROCK)
    )


class SpeciesAgent(Agent):
    def __init__(self, model, species):
        super().__init__(model)
        self.species = species

    def step(self):

        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        if not neighbors:
            return

        target = random.choice(neighbors)

        if dominates(self.species, target.species):
            target.species = self.species


class RPSModel(Model):
    def __init__(self, width=50, height=50):

        super().__init__()

        self.grid = SingleGrid(width, height, torus=True)

        self.datacollector = DataCollector(
            model_reporters={
                "Rock": lambda m: sum(a.species == ROCK for a in m.agents),
                "Scissors": lambda m: sum(a.species == SCISSORS for a in m.agents),
                "Paper": lambda m: sum(a.species == PAPER for a in m.agents),
            }
        )

        for x in range(width):
            for y in range(height):
                species = random.choice([ROCK, SCISSORS, PAPER])

                a = SpeciesAgent(self, species)

                self.grid.place_agent(a, (x, y))

    def advance(self) -> None:
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

    def to_numpy(self):
        arr = np.zeros((self.grid.height, self.grid.width))
        for agent in self.agents:
            x, y = agent.pos
            arr[y, x] = agent.species

        return arr


# ============================================================
# VISUALIZATION
# ============================================================

model = RPSModel()
fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(model.to_numpy(), interpolation="nearest", vmin=-1, vmax=2)

ax.set_title("Rock Paper Scissors CA")


def update(frame):
    model.advance()
    img.set_array(model.to_numpy())
    print(frame)
    return [img]


ani = FuncAnimation(fig, update, frames=100, interval=3, blit=True)
plt.show()


# ============================================================
# POPULATION PLOT
# ============================================================
data = model.datacollector.get_model_vars_dataframe()

plt.figure(figsize=(10, 5))

plt.plot(data["Rock"], label="Rock")
plt.plot(data["Scissors"], label="Scissors")
plt.plot(data["Paper"], label="Paper")

plt.xlabel("Time")
plt.ylabel("Population")

plt.legend()

plt.show()
