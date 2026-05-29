from abc import ABC, abstractmethod

import mesa
import numpy as np
from mesa.agent import Random


class EvolutionPolicy(ABC):
    def __init__(self, model: mesa.Model) -> None:
        self.model = model

    @abstractmethod
    def __call__(self, invasion: float, rng: Random | None = None) -> float:
        pass


class Inheritance(EvolutionPolicy):
    def __init__(self, model: mesa.Model) -> None:
        super().__init__(model)

    def __call__(self, invasion: float, rng: Random | None = None) -> float:
        return invasion


class Stochastic(EvolutionPolicy):
    def __init__(self, model: mesa.Model, sigma: float = 0.01) -> None:
        super().__init__(model)
        self.sigma = sigma

    def __call__(self, invasion: float, rng: Random | None = None) -> float:
        assert rng is not None
        offset = rng.gauss(0, self.sigma)
        return np.clip(invasion + offset, 0, 1)


class Genetic(EvolutionPolicy):
    def __init__(self, model: mesa.Model, sigma: float = 0.01) -> None:
        super().__init__(model)
        self.sigma = sigma

    def __call__(self, invasion: float, rng: Random | None = None) -> float:
        assert rng is not None
        offset = rng.gauss(0, self.sigma)
        return np.clip(invasion + offset, 0, 1)
