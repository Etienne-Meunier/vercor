from datetime import datetime, timedelta
import numpy as np

from vercor.components.base import Component
from vercor.components.base import TimedNamedArray as TNA
from vercor.grid import RectilinearGrid


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses LHF sign as proxy).
    Outputs: SOILM [0..1]
    Inputs: LHF (proxy for evaporation)
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        nlat, nlon = self.grid.shape
        self.outgoing_fields.SOILM = TNA(
            0.3 * np.ones((nlat, nlon)), coupler.clock.start, self.name
        )

    def step(self, dt: timedelta, time: datetime, coupler: "Coupler") -> None:
        LHF = self.incoming_fields.LHF.data
        soil = self.outgoing_fields.SOILM.data
        evap = 1e-9 * (LHF if LHF is not None else 0.0)  # tiny dt scaling
        soil = np.clip(soil - evap * dt.total_seconds(), 0.0, 1.0)
        self.outgoing_fields.SOILM.data = soil
