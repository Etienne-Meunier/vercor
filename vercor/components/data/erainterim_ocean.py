from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component, ForcingData
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERAInterimOcean(Component, ForcingData):
    def __init__(
        self,
        name: str = "ERAINTERIM-OCN",
        model_level_file: Path = (
            Path(__file__).parent.parent.parent
            / ".."
            / "forcing"
            / "forcing_4deg_global_open_itf.nc"
        ).resolve(),
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            model_level_file (Path): path to netCDF file with data at model levels
            surface_file (Path): path to netCDF file with data at surface level

        Attributes of parent classes to be initialized:
            ForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
                shared_fields: Dict[str, NDArray] = field(default_factory=dict)
        """

        self.DATA_FILES = {
            "model_level": str(model_level_file),
        }

        longitude = self._read_forcing("xt", where="model_level")
        latitude = self._read_forcing("yt", where="model_level")
        sss = self._read_forcing("sss", where="model_level")
        binary_mask = np.where(sss > 0.0, 1.0, 0.0)[..., 0].T

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self._settings["apply_time_interpolation"] = True
        self._fields2import = [
            "zbot",
            "ubot",
            "vbot",
            "thbot",
            "qbot",
            "tbot",
            "rbot",
            "swr_net",
            "lwr_dw",
        ]
        self._fields2export = [
            "sst",
        ]

        self.cdata["sst"] = self._read_forcing("sst", where="model_level")
        self.cdata["sst"] *= np.where(binary_mask > 0.0, 1.0, np.nan).T[..., np.newaxis]

    def initialize(self, coupler: "Coupler") -> None:
        pass

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        """
        Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        pass
