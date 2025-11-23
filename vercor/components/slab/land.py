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
        self._fields2export = ["SOILM",]

    def initialize(self, coupler: "Coupler") -> None:
        self._cdata["SOILM"] = 0.3 * np.ones(self.grid.shape)

        self.send_fields_for_export(
            coupler.clock.start, coupler
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
        if time is None:
            raise ValueError(
                f"A 'time' instance is required to advance {self.__class__.__name__}."
            )
        if coupler is None:
            raise ValueError(
                f"A 'Coupler' instance is required to advance {self.__class__.__name__}."
            )

        self.receive_fields_from_import()

        LHF = self._cdata["LHF"]
        soil = self._cdata["SOILM"]

        evap = 1e-9 * (LHF if LHF is not None else 0.0)  # tiny dt scaling
        soil = np.clip(soil - evap * dt.total_seconds(), 0.0, 1.0)
        self._cdata["SOILM"] = soil

        self.send_fields_for_export(time, coupler)
