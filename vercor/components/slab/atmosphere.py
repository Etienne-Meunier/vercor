from datetime import datetime, timedelta
from typing import TYPE_CHECKING

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

    def initialize(self, coupler: "Coupler") -> None:
        grid_shape = self.grid.shape
        zeros = np.zeros(grid_shape)

        self.data["TA2M"] = 273.15 + 15.0 * np.ones(grid_shape)
        self.data["SHF"] = zeros
        self.data["LHF"] = zeros
        self.data["u10m"] = zeros
        self.data["v10m"] = zeros

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        # Bulk formula toy: flux proportional to (TA2M - sst)
        sst = self.data.get("sst", None)

        if sst is None:
            sst = 273.15 + 15.0 * np.ones(self.grid.shape)

        TA = self.data["TA2M"]
        dT = TA - sst
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when sst < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        self.data["u10m"] = np.cos(
            np.deg2rad(latitudes)
        )  # zonal flow varying with latitude
        self.data["v10m"] = 0.5 * np.sin(
            np.deg2rad(longitudes)
        )  # small meridional perturbation

        self.data["SHF"] = SHF
        self.data["LHF"] = LHF

        # Relax TA2M toward sst weakly (toy boundary layer)
        self.data["TA2M"] = TA - 0.01 * dT
