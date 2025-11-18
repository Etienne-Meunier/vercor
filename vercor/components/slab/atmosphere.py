from datetime import datetime, timedelta
import numpy as np

from vercor.components.base import TimedNamedArray as TNA
from vercor.components.base import Component
from vercor.grid import RectilinearGrid


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from SST.
    Inputs: SST [K]
    Outputs: SHF [W/m2], LHF [W/m2], TA2M [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        grid_shape = self.grid.shape
        clock_start = coupler.clock.start
        zeros = np.zeros(grid_shape)
        self.outgoing_fields.TA2M = TNA(
            273.15 + 15.0 * np.ones(grid_shape), clock_start, self.name
        )
        self.outgoing_fields.SHF = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.LHF = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.u10m = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.v10m = TNA(zeros, clock_start, self.name)

    def step(
        self,
        dt: Optional[timedelta] = None,
        time: Optional[datetime] = None,
        coupler: Optional["Coupler"] = None,
    ) -> None:
        # Bulk formula toy: flux proportional to (TA2M - SST)
        SST = self.incoming_fields.SST.data
        if SST is None:
            SST = 273.15 + 15.0 * np.ones(self.grid.shape)

        TA = self.outgoing_fields.TA2M.data
        dT = TA - SST
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when SST < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        u10m = np.cos(np.deg2rad(latitudes))  # zonal flow varying with latitude
        v10m = 0.5 * np.sin(np.deg2rad(longitudes))  # small meridional perturbation

        self.outgoing_fields.SHF.data = SHF
        self.outgoing_fields.LHF.data = LHF

        self.outgoing_fields.u10m.data = u10m
        self.outgoing_fields.v10m.data = v10m

        # Relax TA2M toward SST weakly (toy boundary layer)
        self.outgoing_fields.TA2M.data = TA - 0.01 * dT
