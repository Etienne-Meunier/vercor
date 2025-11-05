import numpy as np
from vercor.grid import RectilinearGrid
from vercor.components.base import Component


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses LHF sign as proxy).
    Outputs: SOILM [0..1]
    Inputs: LHF (proxy for evaporation)
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler) -> None:
        nlat, nlon = self.grid.shape
        self.state["SOILM"] = 0.3 * np.ones((nlat, nlon))

    def step(self, dt, time, coupler) -> None:
        LHF = self.state.get("LHF")
        soil = self.state["SOILM"]
        evap = 1e-9 * (LHF if LHF is not None else 0.0)  # tiny dt scaling
        soil = np.clip(soil - evap * dt.total_seconds(), 0.0, 1.0)
        self.state["SOILM"] = soil
