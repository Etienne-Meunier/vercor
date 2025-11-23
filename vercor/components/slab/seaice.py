from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component
from vercor.components.base import TimedNamedArray as TNA
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
        self._fields2import = ["sst",]
        self._fields2export = ["ICEFRAC",]

    def initialize(self, coupler: "Coupler") -> None:
        self._cdata["ICEFRAC"] = np.zeros(self.grid.shape)

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

        sst = self._cdata.get("sst", None)
        if sst is None:
            return

        Tfreeze = 273.15 - 1.8
        # Smooth step: more ice when colder than freezing
        x = (Tfreeze - sst) / 2.0
        ice = 1.0 / (1.0 + np.exp(-x))

        self._cdata["ICEFRAC"] = ice

        self.send_fields_for_export(time, coupler)