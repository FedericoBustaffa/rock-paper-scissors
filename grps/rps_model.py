from functools import partial

import mesa
import numpy as np
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import RNGLike, SeedLike

from grps.rps_agent import RPSAgent
from grps.evolution_policies import EvolutionPolicy

# ---------------------------------------------------------------------------
# Data-collector reporter functions
# ---------------------------------------------------------------------------

def population_density(specie: str, model: mesa.Model) -> int:
    """Number of living agents of the given species."""
    return len(model.agents.select(lambda a: a.specie == specie))


def specie_invasion(specie: str, model: mesa.Model) -> float:
    """Mean invasion rate of the given species (0.0 if extinct)."""
    agents = model.agents.select(lambda a: a.specie == specie)
    if len(agents) == 0:
        return 0.0
    avg = agents.agg("invasion", np.mean)
    return float(avg)


def specie_age(specie: str, model: mesa.Model) -> float:
    """Mean age of the given species (0.0 if extinct)."""
    agents = model.agents.select(lambda a: a.specie == specie)
    if len(agents) == 0:
        return 0.0
    avg = agents.agg("age", np.mean)
    return float(avg)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RPSModel(mesa.Model):
    """Spatial Rock-Paper-Scissors model on a toroidal Moore-neighbourhood grid.

    The grid is fully occupied (one agent per cell) at all times.  Predation
    coincides with reproduction: a successful hunt converts the prey cell
    rather than removing it, preserving the total population.

    Args:
        dim:      side length of the square grid (total agents = dim²).
        policies: mapping from species name to an :class:`EvolutionPolicy`
                  that determines the offspring's invasion rate.
        rng:      seed or RNG for reproducibility.
    """

    SPECIES = ("rock", "paper", "scissors")

    def __init__(
        self,
        dim: int,
        policies: dict[str, EvolutionPolicy],
        rng: "RNGLike | SeedLike | None" = None,
    ) -> None:
        super().__init__(rng=rng)

        n = dim * dim  # total number of cells / agents
        self.grid = OrthogonalMooreGrid((dim, dim), torus=True, random=self.random)
        self.epoch_length = n  # one epoch = N attempted hunts (as in the paper)
        self.policies = policies
        self.epoch = 0

        # ------------------------------------------------------------------
        # Populate the grid: one agent per cell, random species & invasion
        # ------------------------------------------------------------------
        cells = list(self.grid.all_cells)
        species_list = self.random.choices(self.SPECIES, k=n)
        invasion_list = [self.random.uniform(0.0, 1.0) for _ in range(n)]

        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=cells,
            specie=species_list,
            invasion=invasion_list,
        )

        # ------------------------------------------------------------------
        # Data collectors
        # ------------------------------------------------------------------
        model_reporters = {
            "R_density":  partial(population_density, "rock"),
            "P_density":  partial(population_density, "paper"),
            "S_density":  partial(population_density, "scissors"),
            "R_invasion": partial(specie_invasion, "rock"),
            "P_invasion": partial(specie_invasion, "paper"),
            "S_invasion": partial(specie_invasion, "scissors"),
            "R_age":      partial(specie_age, "rock"),
            "P_age":      partial(specie_age, "paper"),
            "S_age":      partial(specie_age, "scissors"),
        }
        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the model by one epoch.

        Each epoch consists of ``epoch_length`` randomly-chosen hunt attempts,
        followed by aging all agents by one step.  Data are collected at the
        *beginning* of each epoch (i.e. before the hunts).
        """
        self.datacollector.collect(self)
        self._log_epoch()
        self.epoch += 1

        # Randomly activate agents (with replacement, as in the reference paper)
        for _ in range(self.epoch_length):
            ind = self.random.choice(self.agents)
            ind.hunt()

        # Every agent ages by 1 at the end of the epoch
        self.agents.do("get_older")

    def run_for(self, n_epochs: int) -> None:
        """Run the model for *n_epochs* epochs."""
        for _ in range(n_epochs):
            self.step()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_epoch(self) -> None:
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
            f"epoch {self.epoch:>4d} | "
            f"R: {r:>4d} (inv={ri:.3f}, age={ra:.1f})  "
            f"P: {p:>4d} (inv={pi:.3f}, age={pa:.1f})  "
            f"S: {s:>4d} (inv={si:.3f}, age={sa:.1f})"
        )
