from pathlib import Path
from typing import Tuple

import numpy as np

from vercor.components.base import Component, ForcingData
from vercor.fluxes.utilities import (
    air_density,
    compute_z_level,
    get_press_levs,
    potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.tools import datetime_to_seconds_in_year, get_periodic_interval


def get_data_dir() -> Tuple[Path, Path]:
    """Return the absolute Paths to the ../data directory relative to this file."""
    return (
        (
            Path(__file__).parent.parent.parent
            / "data"
            / "era5_198x_ml_4x4deg_monthly_mean.nc"
        ).resolve(),
        (
            Path(__file__).parent.parent.parent
            / "data"
            / "era5_198x_sfc_4x4deg_monthly_mean.nc"
        ).resolve(),
    )


class ERA5Atmosphere(Component, ForcingData):
    def __init__(
        self,
        name: str = "ERA5-ATM",
        model_level_file: Path = get_data_dir()[0],
        surface_file: Path = get_data_dir()[1],
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            model_level_file (Path): path to netCDF file with data at model levels
            surface_file (Path): path to netCDF file with data at surface level

        Logic:
            - only the lowest to the ground model levels are available and read (L136, L137)

        Attributes from base classes to be initialized:
            ForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
                state: Dict[str, NDArray] = field(default_factory=dict)
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

        self.state["hyai"] = self._read_forcing("hyai", where="model_level")[
            -3:
        ]  # L135-L137
        self.state["hybi"] = self._read_forcing("hybi", where="model_level")[
            -3:
        ]  # L135-L137
        self.state["hyam"] = self._read_forcing("hyam", where="model_level")[
            -2:
        ]  # L136-L137
        self.state["hybm"] = self._read_forcing("hybm", where="model_level")[
            -2:
        ]  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        self.state["surf_pressure_data"] = np.exp(lnsp)
        self.state["specific_humidity_data"] = self._read_forcing("q", where="model_level", flip_y=True)[
            ..., 1:, :
        ]  # L136-L137
        self.state["temperature_data"] = self._read_forcing("t", where="model_level", flip_y=True)[
            ..., 1:, :
        ]  # L136-L137

        self.state["ubot_data"] = self._read_forcing("u", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136
        self.state["vbot_data"] = self._read_forcing("v", where="model_level", flip_y=True)[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", where="surface", flip_y=True)
        self.state["swr_net_data"] = self._read_forcing(
            "msnswrf", where="surface", flip_y=True
        )
        self.state["lwr_dw_data"] = self._read_forcing(
            "msdwlwrf", where="surface", flip_y=True
        )

        self.state["qbot_data"] = self.state["specific_humidity_data"][..., 0, :]  # L136
        self.state["tbot_data"] = self.state["temperature_data"][..., 0, :]  # L136

    def initialize(self, coupler) -> None:
        ny, nx = self.grid.shape
        settings = coupler.settings
        dataset = self.state

        # Values (local) to be used for time interpolation
        self.state["zbot_data"] = np.zeros((nx, ny, 12))
        self.state["rbot_data"] = np.zeros((nx, ny, 12))
        self.state["thbot_data"] = np.zeros((nx, ny, 12))

        # Values at current time step (to be exchanged with other components)
        self.state["ubot"] = np.zeros((ny, nx))
        self.state["vbot"] = np.zeros((ny, nx))
        self.state["zbot"] = np.zeros((ny, nx))
        self.state["thbot"] = np.zeros((ny, nx))
        self.state["qbot"] = np.zeros((ny, nx))
        self.state["tbot"] = np.zeros((ny, nx))
        self.state["rbot"] = np.zeros((ny, nx))

        for m in range(12):
            ph = get_press_levs(
                dataset["surf_pressure_data"][..., m], dataset["hyai"], dataset["hybi"]
            )
            pf = get_press_levs(
                dataset["surf_pressure_data"][..., m], dataset["hyam"], dataset["hybm"]
            )

            self.state["zbot_data"][..., m] = compute_z_level(
                settings,
                dataset["temperature_data"][..., m],
                dataset["specific_humidity_data"][..., m],
                ph[:, :],
            )[..., 0]  # L136
            self.state["rbot_data"][..., m] = air_density(
                settings, dataset["tbot_data"][:, :, m], pf[:, :, 0]
            )
            self.state["thbot_data"][..., m] = potential_temperature(
                settings, dataset["tbot_data"][:, :, m], pf[:, :, 0]
            )

    def step(self, dt, time, coupler) -> None:
        """Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """

        for variable in (
            "zbot",
            "ubot",
            "vbot",
            "thbot",
            "qbot",
            "tbot",
            "rbot",
        ):
            total_seconds = datetime_to_seconds_in_year(time)

            (n1, f1), (n2, f2) = get_periodic_interval(
                current_time=total_seconds,
                cycle_length=coupler.settings.year_in_seconds,
                rec_spacing=coupler.settings.year_in_seconds / 12.0,
                n_rec=12,
            )

            # Use transpose to have (lat, lon) ordering
            self.state[variable][...] = (
                f1 * self.state[f"{variable}_data"][..., n1].T
                + f2 * self.state[f"{variable}_data"][..., n2].T
            )
