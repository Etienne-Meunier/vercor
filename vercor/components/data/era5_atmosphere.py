from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component, ForcingData
from vercor.fluxes.utilities import (
    compute_air_density,
    compute_levels_altitudes,
    compute_pressure_levels,
    compute_potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERA5Atmosphere(Component, ForcingData):
    def __init__(
        self,
        name: str = "ERA5-ATM",
        model_level_file: Path = get_forcing_data("model_level"),
        surface_file: Path = get_forcing_data("surface"),
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            model_level_file (Path): path to netCDF file with data at model levels
            surface_file (Path): path to netCDF file with data at surface level

        Data description:
            Only the lowest to the ground model levels are available and read (L136, L137)
            See ECMWF IFS documentation on vertical model resolution for more details:
            https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

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
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("longitude", where="model_level")
        latitude = self._read_forcing("latitude", where="model_level")[::-1]

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
        )

        super().__init__(name, grid=self.grid)

        self._settings["apply_time_interpolation"] = True
        self._fields2import = [
            "sst",
        ]
        self._fields2export = [
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

        self._cdata["hyai"] = self._read_forcing("hyai", where="model_level")[
            -3:
        ]  # L135-L137
        self._cdata["hybi"] = self._read_forcing("hybi", where="model_level")[
            -3:
        ]  # L135-L137
        self._cdata["hyam"] = self._read_forcing("hyam", where="model_level")[
            -2:
        ]  # L136-L137
        self._cdata["hybm"] = self._read_forcing("hybm", where="model_level")[
            -2:
        ]  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        self._cdata["surf_pressure"] = np.exp(lnsp)
        self._cdata["specific_humidity"] = self._read_forcing(
            "q", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        self._cdata["temperature"] = self._read_forcing(
            "t", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137

        self._cdata["ubot"] = self._read_forcing("u", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136
        self._cdata["vbot"] = self._read_forcing("v", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", where="surface", flip_y=True)
        self._cdata["swr_net"] = self._read_forcing(
            "msnswrf", where="surface", flip_y=True
        )
        self._cdata["lwr_dw"] = self._read_forcing(
            "msdwlwrf", where="surface", flip_y=True
        )

        self._cdata["qbot"] = self._cdata["specific_humidity"][..., 0, :]  # L136
        self._cdata["tbot"] = self._cdata["temperature"][..., 0, :]  # L136

    def initialize(self, coupler: "Coupler") -> None:
        nlat, nlon = self.grid.shape
        settings = coupler.settings
        ds = self._cdata

        self._cdata["zbot"] = np.zeros((nlon, nlat, 12))
        self._cdata["rbot"] = np.zeros((nlon, nlat, 12))
        self._cdata["thbot"] = np.zeros((nlon, nlat, 12))

        for m in range(12):
            ph = compute_pressure_levels(
                ds["surf_pressure"][..., m], ds["hyai"], ds["hybi"]
            )
            pf = compute_pressure_levels(
                ds["surf_pressure"][..., m], ds["hyam"], ds["hybm"]
            )
            self._cdata["zbot"][..., m] = compute_levels_altitudes(
                settings,
                ds["temperature"][..., m],
                ds["specific_humidity"][..., m],
                ph[:, :],
            )[
                ..., 1
            ]  # L136
            self._cdata["rbot"][..., m] = compute_air_density(
                settings, ds["tbot"][:, :, m], pf[:, :, 0]
            )
            self._cdata["thbot"][..., m] = compute_potential_temperature(
                settings, ds["tbot"][:, :, m], pf[:, :, 0]
            )

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
