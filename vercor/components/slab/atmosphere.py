from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components import Component
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

        self.cdata["TA2M"] = 273.15 + 15.0 * np.ones(grid_shape)
        self.cdata["SHF"] = zeros
        self.cdata["LHF"] = zeros
        self.cdata["u10m"] = zeros
        self.cdata["v10m"] = zeros

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        # Bulk formula toy: flux proportional to (TA2M - sst)
        sst = self.cdata.get("sst", None)

        if sst is None:
            sst = 273.15 + 15.0 * np.ones(self.grid.shape)

        TA = self.cdata["TA2M"]
        dT = TA - sst
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when sst < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        self.cdata["u10m"] = np.cos(
            np.deg2rad(latitudes)
        )  # zonal flow varying with latitude
        self.cdata["v10m"] = 0.5 * np.sin(
            np.deg2rad(longitudes)
        )  # small meridional perturbation

        self.cdata["SHF"] = SHF
        self.cdata["LHF"] = LHF

        # Relax TA2M toward sst weakly (toy boundary layer)
        self.cdata["TA2M"] = TA - 0.01 * dT
