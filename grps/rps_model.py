import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import RNGLike, SeedLike

from grps import RPSAgent


class RPSModel(mesa.Model):
    def __init__(
        self,
        width: int,
        height: int,
        rng: RNGLike | SeedLike | None = None,
    ) -> None:
        super().__init__(rng=rng)

        n = width * height  # number of individuals induced by grid dimensions
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)
        self.epoch_length = n  # same as the paper

        # create agents
        species = ["rock", "paper", "scissors"]
        invasion_probas = [self.random.uniform(0, 1) for _ in range(n)]

        RPSAgent.create_agents(
            model=self,
            n=n,
            cell=self.grid.all_cells.cells,
            specie=self.random.choices(species, k=n),
            invasion=invasion_probas,
        )

    def step(self) -> None:
        # custom scheduler like in the paper in which not every agent hunts every step/epoch
        for e in range(self.epoch_length):
            ind = self.random.choice(self.agents)
            ind.hunt()
