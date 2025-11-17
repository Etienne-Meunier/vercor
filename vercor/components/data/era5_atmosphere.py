from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vercor.components.base import Component, ForcingData, write_shared_to_netcdf
from vercor.components.base import TimedNamedArray as TNA
from vercor.fluxes.utilities import (
    air_density,
    compute_z_level,
    get_press_levs,
    potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.tools import get_field_at_specific_time, get_forcing_data


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

        self.fields2share = (
            "zbot",
            "ubot",
            "vbot",
            "thbot",
            "qbot",
            "tbot",
            "rbot",
        )

        self._state = {}

        longitude = self._read_forcing("longitude", where="model_level")
        latitude = self._read_forcing("latitude", where="model_level")[::-1]

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
        )

        super().__init__(name, grid=self.grid)

        self._state["hyai"] = self._read_forcing("hyai", where="model_level")[
            -3:
        ]  # L135-L137
        self._state["hybi"] = self._read_forcing("hybi", where="model_level")[
            -3:
        ]  # L135-L137
        self._state["hyam"] = self._read_forcing("hyam", where="model_level")[
            -2:
        ]  # L136-L137
        self._state["hybm"] = self._read_forcing("hybm", where="model_level")[
            -2:
        ]  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        self._state["surf_pressure"] = np.exp(lnsp)
        self._state["specific_humidity"] = self._read_forcing(
            "q", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        self._state["temperature"] = self._read_forcing(
            "t", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137

        self._state["ubot"] = self._read_forcing("u", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136
        self._state["vbot"] = self._read_forcing("v", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", where="surface", flip_y=True)
        self._state["swr_net"] = self._read_forcing(
            "msnswrf", where="surface", flip_y=True
        )
        self._state["lwr_dw"] = self._read_forcing(
            "msdwlwrf", where="surface", flip_y=True
        )

        self._state["qbot"] = self._state["specific_humidity"][..., 0, :]  # L136
        self._state["tbot"] = self._state["temperature"][..., 0, :]  # L136

    def initialize(self, coupler: "Coupler") -> None:
        nlat, nlon = self.grid.shape
        settings = coupler.settings
        dataset = self._state

        # Values (local) to be used for time interpolation
        self._state["zbot"] = np.zeros((nlon, nlat, 12))
        self._state["rbot"] = np.zeros((nlon, nlat, 12))
        self._state["thbot"] = np.zeros((nlon, nlat, 12))

        for m in range(12):
            ph = get_press_levs(
                dataset["surf_pressure"][..., m], dataset["hyai"], dataset["hybi"]
            )
            pf = get_press_levs(
                dataset["surf_pressure"][..., m], dataset["hyam"], dataset["hybm"]
            )
            self._state["zbot"][..., m] = compute_z_level(
                settings,
                dataset["temperature"][..., m],
                dataset["specific_humidity"][..., m],
                ph[:, :],
            )[
                ..., 1
            ]  # L136
            self._state["rbot"][..., m] = air_density(
                settings, dataset["tbot"][:, :, m], pf[:, :, 0]
            )
            self._state["thbot"][..., m] = potential_temperature(
                settings, dataset["tbot"][:, :, m], pf[:, :, 0]
            )

        for field in self.fields2share:
            setattr(
                self.outgoing_fields,
                field,
                TNA(
                    get_field_at_specific_time(field, self._state, coupler),
                    coupler.clock.start,
                    self.name,
                ),
            )

    def step(self, dt: timedelta, time: datetime, coupler: "Coupler") -> None:
        """Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """

        for field in self.fields2share:
            setattr(
                self.outgoing_fields,
                field,
                TNA(
                    get_field_at_specific_time(field, self._state, coupler),
                    time,
                    self.name,
                ),
            )
