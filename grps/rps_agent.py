import mesa
from mesa.discrete_space import Cell, CellAgent

from grps.evolution_policies import EvolutionPolicy


def get_prey(specie: str) -> str:
    if specie == "rock":
        return "paper"
    elif specie == "paper":
        return "scissors"
    elif specie == "scissors":
        return "rock"
    else:
        raise ValueError(f"Invalid specie: {specie}")


class RPSAgent(CellAgent):
    def __init__(
        self,
        model: mesa.Model,
        cell: Cell,
        specie: str,
        invasion: float,
        evo_policy: EvolutionPolicy,
    ) -> None:
        super().__init__(model)
        self.cell = cell
        self.specie = specie
        self.prey = get_prey(specie)
        self.age = 0

        assert invasion >= 0.0 and invasion <= 1.0  # ensure valid probability
        self.invasion = invasion
        self.evo_policy = evo_policy

    def hunt(self) -> None:
        assert self.cell is not None

        # select a random nearby cell and its agent
        nearby_cell = self.random.choice(list(self.cell.neighborhood))
        nearby_agent = nearby_cell.agents[0]
        assert isinstance(nearby_agent, RPSAgent)

        # if the specie of the nearby agent matches the prey of this agent, kill
        # the nearby agent
        if self.prey == nearby_agent.specie:
            if self.model.random.random() < self.invasion:
                nearby_agent.specie = self.specie
                nearby_agent.prey = self.prey
                nearby_agent.age = 0
                nearby_agent.invasion = self.evo_policy(
                    self.invasion, self.model.random
                )

    def get_older(self) -> None:
        self.age += 1
