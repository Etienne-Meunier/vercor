from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from sst.
    Inputs: sst [K]
    Outputs: SHF [W/m2], LHF [W/m2], TA2M [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)
        self._fields2import = ["sst"]
        self._fields2export = ["TA2M", "SHF", "LHF", "u10m", "v10m"]

    def initialize(self, coupler: "Coupler") -> None:
        grid_shape = self.grid.shape
        zeros = np.zeros(grid_shape)

        self._cdata["TA2M"] = 273.15 + 15.0 * np.ones(grid_shape)
        self._cdata["SHF"] = zeros
        self._cdata["LHF"] = zeros
        self._cdata["u10m"] = zeros
        self._cdata["v10m"] = zeros

        self.send_fields_for_export(
            coupler.clock.start, coupler
        )

    def step(
        self,
        dt: Optional[timedelta] = None,
        time: Optional[datetime] = None,
        coupler: Optional["Coupler"] = None,
    ) -> None:

        if time is None:
            raise ValueError(
                f"A 'time' instance is required to advance {self.__class__.__name__}."
            )
        if coupler is None:
            raise ValueError(
                f"A 'Coupler' instance is required to advance {self.__class__.__name__}."
            )

        # Bulk formula toy: flux proportional to (TA2M - sst)
        self.receive_fields_from_import()

        sst = self._cdata.get("sst", None)

        if sst is None:
            sst = 273.15 + 15.0 * np.ones(self.grid.shape)

        TA = self._cdata["TA2M"]
        dT = TA - sst
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when sst < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        self._cdata["u10m"] = np.cos(
            np.deg2rad(latitudes)
        )  # zonal flow varying with latitude
        self._cdata["v10m"] = 0.5 * np.sin(
            np.deg2rad(longitudes)
        )  # small meridional perturbation

        self._cdata["SHF"] = SHF
        self._cdata["LHF"] = LHF

        # Relax TA2M toward sst weakly (toy boundary layer)
        self._cdata["TA2M"] = TA - 0.01 * dT

        self.send_fields_for_export(time, coupler)
