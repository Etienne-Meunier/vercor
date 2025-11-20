from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component
from vercor.components.base import TimedNamedArray as TNA
from vercor.grid import RectilinearGrid

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
        self.outgoing_fields.ICEFRAC = TNA(
            np.zeros((nlat, nlon)), coupler.clock.start, self.name
        )

    def step(
        self,
        dt: Optional[timedelta] = None,
        time: Optional[datetime] = None,
        coupler: Optional["Coupler"] = None,
    ) -> None:
        if dt is None:
            raise ValueError(
                f"A 'dt' instance is required to advance {self.__class__.__name__}."
            )

        SST = self.incoming_fields.SST.data
        if SST is None:
            return
        Tfreeze = 273.15 - 1.8
        # Smooth step: more ice when colder than freezing
        x = (Tfreeze - SST) / 2.0
        ice = 1.0 / (1.0 + np.exp(-x))
        self.outgoing_fields.ICEFRAC.data = ice
