from functools import partial

import mesa
import numpy as np
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import RNGLike, SeedLike

from grps import RPSAgent
from grps.evolution_policies import EvolutionPolicy


def population_density(specie: str, model: mesa.Model) -> int:
    return len(model.agents.select(lambda a: a.specie == specie))


def specie_invasion(specie: str, model: mesa.Model) -> float:
    avg_invasion = model.agents.select(lambda a: a.specie == specie).agg(
        "invasion", np.mean
    )
    assert isinstance(avg_invasion, float)
    return avg_invasion


def specie_age(specie: str, model: mesa.Model) -> int:
    avg_age = model.agents.select(lambda a: a.specie == specie).agg("age", np.mean)
    assert isinstance(avg_age, float)
    return int(avg_age)


class RPSModel(mesa.Model):
    def __init__(
        self,
        width: int,
        height: int,
        policies: dict[str, EvolutionPolicy],
        rng: RNGLike | SeedLike | None = None,
    ) -> None:
        super().__init__(rng=rng)

        n = width * height  # number of individuals induced by grid dimensions
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)
        self.epoch_length = n  # same as the paper

        # create agents
        species = self.random.choices(["rock", "paper", "scissors"], k=n)
        invasion_probas = [self.random.uniform(0, 1) for _ in range(n)]
        individual_policies = [policies[s] for s in species]

        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=self.grid.all_cells.cells,
            specie=self.random.choices(species, k=n),
            invasion=invasion_probas,
            evo_policy=individual_policies,
        )

        # data collectors
        model_reporters = {
            "R_density": partial(population_density, "rock"),
            "P_density": partial(population_density, "paper"),
            "S_density": partial(population_density, "scissors"),
            "R_invasion": partial(specie_invasion, "rock"),
            "P_invasion": partial(specie_invasion, "paper"),
            "S_invasion": partial(specie_invasion, "scissors"),
            "R_age": partial(specie_age, "rock"),
            "P_age": partial(specie_age, "paper"),
            "S_age": partial(specie_age, "scissors"),
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)

        self.epoch = 0

    def step(self) -> None:
        # custom scheduler like in the paper in which not every agent hunts
        # every step/epoch
        self.datacollector.collect(self)
        r = population_density("rock", self)
        p = population_density("paper", self)
        s = population_density("scissors", self)

        ri = specie_invasion("rock", self)
        pi = specie_invasion("paper", self)
        si = specie_invasion("scissors", self)

        ra = specie_age("rock", self)
        pa = specie_age("paper", self)
        sa = specie_age("scissors", self)
        print(
            f"""epoch: {self.epoch},\
R_d: {r}, P_d: {p}, S_d: {s},\
R_i: {ri:.2f}, P_i: {pi:.2f}, S_i: {si:.2f},\
R_a: {ra}, P_a: {pa}, S_a: {sa}
            """
        )
        self.epoch += 1

        for _ in range(self.epoch_length):
            ind = self.random.choice(self.agents)
            ind.hunt()

        self.agents.do("get_older")
