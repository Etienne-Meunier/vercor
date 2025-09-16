import numpy as np

from verec.grid import RectilinearGrid
from verec.components.base import Component
from verec.fields import Field


class Ocean(Component):
    """Toy slab ocean: updates SST using SHF (sensible) + LHF (latent).
    Outputs: SST [K]
    Inputs: SHF, LHF
    """

    def __init__(self, name: str, grid: RectilinearGrid, H: float = 30.0) -> None:
        super().__init__(name, grid, inputs=["SHF", "LHF"], outputs=["SST"])
        self.H = H  # mixed-layer depth [m]
        self.rho = 1025.0
        self.cp = 3990.0
        self.lambda_relax = 1.0 / (
            30.0 * 86400.0
        )  # weak restoring to 15C over ~30 days

    def initialize(self, coupler) -> None:
        ny, nx = self.grid.shape
        self.state["SST"] = Field(
            "SST", 273.15 + 15.0 * np.ones((ny, nx)), self.grid, units="K"
        )

    def step(self, dt, t, coupler) -> None:
        ny, nx = self.grid.shape
        SST = self.state["SST"].data
        SHF = self.state.get("SHF")
        LHF = self.state.get("LHF")
        Qnet = np.zeros_like(SST)
        if SHF is not None:
            Qnet += SHF.data
        if LHF is not None:
            Qnet += LHF.data
        T0 = 273.15 + 15.0
        dTdt = Qnet / (self.rho * self.cp * self.H) - self.lambda_relax * (SST - T0)
        self.state["SST"].data = SST + dTdt * dt.total_seconds()
