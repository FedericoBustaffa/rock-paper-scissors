from typing import Sequence

import mesa
import numpy as np
from mesa.discrete_space import OrthogonalMooreGrid

from grps import RPSAgent


def mean_invasion(agentset):
    values = agentset.get("invasion", handle_missing="default")
    return np.mean(values) if len(values) > 0 else 0.0


class RPSModel(mesa.Model):
    def __init__(
        self,
        n: int,
        width: int,
        height: int,
        rng: None | float = None,
    ) -> None:
        super().__init__(rng=rng)

        # create grid
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)

        # create agents
        species = ["rock", "paper", "scissors"]
        invasion_probas = [self.random.uniform(0, 1) for _ in range(n)]
        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=self.random.choices(self.grid.all_cells.cells, k=n),
            specie=self.random.choices(species, k=n),
            invasion=invasion_probas,
        )

        # data collection
        model_reporters = {
            "rock_density": lambda m: len(
                m.agents.select(lambda a: a.specie == "rock")
            ),
            "paper_density": lambda m: len(
                m.agents.select(lambda a: a.specie == "paper")
            ),
            "scissors_density": lambda m: len(
                m.agents.select(lambda a: a.specie == "scissors")
            ),
            "rock_invasion": lambda m: mean_invasion(
                m.agents.select(lambda a: a.specie == "rock")
            ),
            "paper_invasion": lambda m: mean_invasion(
                m.agents.select(lambda a: a.specie == "paper")
            ),
            "scissors_invasion": lambda m: mean_invasion(
                m.agents.select(lambda a: a.specie == "scissors")
            ),
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)

    def step(self) -> None:  # ty:ignore[invalid-method-override]
        self.datacollector.collect(self)
        self.agents.shuffle_do("hunt")
