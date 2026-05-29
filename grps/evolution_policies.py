import numpy as np

from grps import RPSAgent


class EvolutionPolicy:
    def __init__(self) -> None:
        pass

    def __call__(self, agent: RPSAgent) -> float: ...


class Inheritance(EvolutionPolicy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, agent: RPSAgent) -> float:
        return agent.invasion


class Stochastic(EvolutionPolicy):
    def __init__(self, sigma: float = 0.01) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, agent: RPSAgent) -> float:
        offset = agent.model.random.gauss(0, self.sigma)
        return np.clip(agent.invasion + offset, 0, 1)


class Genetic(EvolutionPolicy):
    def __init__(self, sigma: float = 0.01) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, agent: RPSAgent) -> float:
        return agent.invasion
