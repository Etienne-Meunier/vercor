from datetime import datetime, timedelta
import numpy as np

from vercor.components.base import Component
from vercor.grid import RectilinearGrid


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vercor.coupler import Coupler


class SeaIce(Component):
    """Toy thermodynamic sea-ice: diagnostic concentration from SST.
    Outputs: ICEFRAC [0..1]
    Inputs: SST [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        nlat, nlon = self.grid.shape
        self.shared_fields["ICEFRAC"] = np.zeros((nlat, nlon))

    def step(self, dt: timedelta, time: datetime, coupler: "Coupler") -> None:
        SST = self.shared_fields.get("SST")
        if SST is None:
            return
        Tfreeze = 273.15 - 1.8
        # Smooth step: more ice when colder than freezing
        x = (Tfreeze - SST) / 2.0
        ice = 1.0 / (1.0 + np.exp(-x))
        self.shared_fields["ICEFRAC"] = ice

    def finalize(self, coupler: "Coupler") -> None:
        pass