import mesa
import numpy as np


class Specie(mesa.Agent):
    def __init__(self, invasion: float) -> None:
        assert abs(invasion) <= 1.0  # ensure valid probability
        self.invasion = invasion
