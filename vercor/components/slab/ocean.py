from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Ocean(Component):
    """Toy slab ocean: updates sst using SHF (sensible) + LHF (latent).
    Outputs: sst [K]
    Inputs: SHF, LHF
    """

    def __init__(self, name: str, grid: RectilinearGrid, H: float = 30.0) -> None:

        super().__init__(name, grid)

        self._fields2import = ["u10m", "v10m"]
        self._fields2export = [
            "sst",
        ]

        self.H = H  # mixed-layer depth [m]
        self.rho = 1025.0
        self.cp = 3990.0
        self.lambda_relax = 1.0 / (
            30.0 * 86400.0
        )  # weak restoring to 15C over ~30 days

    def initialize(self, coupler: "Coupler") -> None:
        self.cdata["sst"] = 273.15 + 15.0 * np.ones(self.grid.shape)

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        sst = self.cdata.get("sst", None)
        if sst is None:
            return

        SHF = self.cdata.get("SHF", None)
        LHF = self.cdata.get("LHF", None)
        Qnet = np.zeros_like(sst)
        if SHF is not None:
            Qnet += SHF
        if LHF is not None:
            Qnet += LHF
        T0 = 273.15 + 15.0
        dTdt = - Qnet / (self.rho * self.cp * self.H) - self.lambda_relax * (sst - T0)
        self.cdata["sst"] = sst + dTdt * dt.total_seconds()
