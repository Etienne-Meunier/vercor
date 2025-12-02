from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component
from vercor.components.base import TimedNamedArray as TNA
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses LHF sign as proxy).
    Outputs: SOILM [0..1]
    Inputs: LHF (proxy for evaporation)
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)
        self._fields2import = ["LHF", "SHF"]
        self._fields2export = [
            "SOILM",
        ]

    def initialize(self, coupler: "Coupler") -> None:
        self.cdata["SOILM"] = 0.3 * np.ones(self.grid.shape)

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        LHF = self.cdata["LHF"]
        soil = self.cdata["SOILM"]

        evap = 1e-9 * (LHF if LHF is not None else 0.0)  # tiny dt scaling
        soil = np.clip(soil - evap * dt.total_seconds(), 0.0, 1.0)
        self.cdata["SOILM"] = soil
