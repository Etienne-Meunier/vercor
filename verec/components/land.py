import numpy as np

from verec.grid import RectilinearGrid
from verec.components.base import Component
from verec.fields import Field


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses LHF sign as proxy).
    Outputs: SOILM [0..1]
    Inputs: LHF (proxy for evaporation)
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        # super().__init__(name, grid, inputs=["LHF"], outputs=["SOILM"])
        super().__init__(name, grid)

    def initialize(self, coupler) -> None:
        nlat, nlon = self.grid.shape
        self.state["SOILM"] = Field(
            "SOILM", 0.3 * np.ones((nlat, nlon)), self.grid, units="1"
        )

    def step(self, dt, time, coupler) -> None:
        LHF = self.state.get("LHF")
        soil = self.state["SOILM"].data
        evap = 1e-9 * (LHF.data if LHF is not None else 0.0)  # tiny dt scaling
        soil = np.clip(soil - evap * dt.total_seconds(), 0.0, 1.0)
        self.state["SOILM"].data = soil
