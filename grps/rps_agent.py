import mesa
from mesa.discrete_space import Cell, CellAgent


def get_prey(specie: str) -> str:
    """Return the species that *specie* predates (i.e. beats in RPS)."""
    match specie:
        case "rock":
            return "scissors"
        case "paper":
            return "rock"
        case "scissors":
            return "paper"
        case _:
            raise ValueError(f"Invalid specie: {specie}")


class RPSAgent(CellAgent):
    """A single individual in the Rock-Paper-Scissors spatial model.

    Attributes:
        specie:    one of ``"rock"``, ``"paper"``, ``"scissors"``.
        prey:      the species this agent predates.
        age:       number of epochs survived (used as fitness proxy).
        invasion:  probability [0, 1] of successfully converting a neighbour
                   when predation is attempted.
    """

    def __init__(
        self,
        model: mesa.Model,
        cell: Cell,
        specie: str,
        invasion: float,
    ) -> None:
        super().__init__(model)
        self.cell = cell
        self.specie = specie
        self.prey = get_prey(specie)
        self.age = 0

        if not (0.0 <= invasion <= 1.0):
            raise ValueError(f"invasion must be in [0, 1], got {invasion}")
        self.invasion = invasion

    # ------------------------------------------------------------------
    # scheduled methods
    # ------------------------------------------------------------------

    def hunt(self) -> None:
        """Try to predate a random neighbour.

        If a neighbour of the prey species is found and the invasion roll
        succeeds, the neighbour is *converted* (not removed): it takes on
        the predator's species and is assigned a new invasion rate according
        to the model's evolution policy.
        """
        assert self.cell is not None

        # Pick a random neighbouring cell (Moore neighbourhood, torus).
        nearby_cell = self.random.choice(list(self.cell.neighborhood))
        if not nearby_cell.agents:
            return  # empty cell – nothing to hunt (shouldn't happen on a full grid)

        nearby_agent = nearby_cell.agents[0]
        assert isinstance(nearby_agent, RPSAgent)

        # Predation: this agent hunts its prey species.
        if nearby_agent.specie == self.prey:
            if self.model.random.random() < self.invasion:
                # Convert the prey cell to the predator's species.
                nearby_agent.specie = self.specie
                nearby_agent.prey = get_prey(self.specie)
                nearby_agent.age = 0

                # Compute offspring's invasion rate via the evolution policy.
                evo_policy = self.model.policies[self.specie]
                nearby_agent.invasion = evo_policy(self)

    def get_older(self) -> None:
        """Increment age by one epoch."""
        self.age += 1
