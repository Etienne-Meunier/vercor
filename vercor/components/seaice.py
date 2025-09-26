import numpy as np
from vercor.grid import RectilinearGrid
from vercor.components.base import Component


class SeaIce(Component):
    """Toy thermodynamic sea-ice: diagnostic concentration from SST.
    Outputs: ICEFRAC [0..1]
    Inputs: SST [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler) -> None:
        nlat, nlon = self.grid.shape
        self.state["ICEFRAC"] = np.zeros((nlat, nlon))

    def step(self, dt, time, coupler) -> None:
        SST = self.state.get("SST")
        if SST is None:
            return
        Tfreeze = 273.15 - 1.8
        # Smooth step: more ice when colder than freezing
        x = (Tfreeze - SST) / 2.0
        ice = 1.0 / (1.0 + np.exp(-x))
        self.state["ICEFRAC"] = ice

