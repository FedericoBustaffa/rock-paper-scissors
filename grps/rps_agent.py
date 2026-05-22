import mesa
import numpy as np


def get_prey(specie):
    if specie == "rock":
        return "scissors"
    if specie == "paper":
        return "rock"
    if specie == "scissors":
        return "paper"


class RPSAgent(mesa.Agent):
    def __init__(
        self, model: mesa.Model, cell: Cell, specie: str, invasion: float
    ) -> None:
        super().__init__(model)
        self.cell = cell
        self.specie = specie
        self.prey = get_prey(specie)
        assert invasion >= 0.0 and invasion <= 1.0  # ensure valid probability
        self.invasion = invasion

    def hunt(self) -> None:
        cellmates = [a for a in self.cell.agents if a is not self]
        other = self.random.choice(cellmates)

        if other.specie == self.prey:
            if self.random.uniform(0, 1) <= self.invasion:
                other.specie = self.specie
                other.invasion = min(self.invasion + self.random.gauss(0, 0.01), 1)
