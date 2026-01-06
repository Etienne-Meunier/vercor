from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class SeaIce(Component):
    """Toy thermodynamic sea-ice: diagnostic concentration from sst.
    Outputs: ICEFRAC [0..1]
    Inputs: sst [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)
        self._fields2import = [
            "sst",
        ]
        self._fields2export = [
            "ICEFRAC",
        ]

    def initialize(self, coupler: "Coupler") -> None:
        self.data["ICEFRAC"] = np.zeros(self.grid.shape)

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        sst = self.data.get("sst", None)
        if sst is None:
            return

        Tfreeze = 273.15 - 1.8
        # Smooth step: more ice when colder than freezing
        x = (Tfreeze - sst) / 2.0
        ice = 1.0 / (1.0 + np.exp(-x))

        self.data["ICEFRAC"] = ice
