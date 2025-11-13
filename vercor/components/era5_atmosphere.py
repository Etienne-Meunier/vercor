from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from vercor.components.base import Component, ForcingData
from vercor.fluxes.utilities import (
    air_density,
    compute_z_level,
    get_press_levs,
    potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.tools import datetime_to_seconds_in_year, get_periodic_interval


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vercor.coupler import Coupler


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


def get_values_at_specific_time(
        variable: str, state: Dict, coupler: "Coupler", current_time: Optional[datetime] = None
    ) -> NDArray:

    total_seconds = datetime_to_seconds_in_year(
        coupler.clock.start if current_time is None else current_time
    )

    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    # Use transpose to have (lat, lon) ordering
    return (
        f1 * state[f"{variable}"][..., n1].T
        + f2 * state[f"{variable}"][..., n2].T
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
                shared_fields: Dict[str, NDArray] = field(default_factory=dict)
        """

        self.DATA_FILES = {
            "model_level": str(model_level_file),
            "surface": str(surface_file),
        }

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
        self._state["specific_humidity"] = self._read_forcing("q", where="model_level", flip_y=True)[
            ..., 1:, :
        ]  # L136-L137
        self._state["temperature"] = self._read_forcing("t", where="model_level", flip_y=True)[
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
        ny, nx = self.grid.shape
        settings = coupler.settings
        dataset = self._state

        # Values (local) to be used for time interpolation
        self._state["zbot"] = np.zeros((nx, ny, 12))
        self._state["rbot"] = np.zeros((nx, ny, 12))
        self._state["thbot"] = np.zeros((nx, ny, 12))

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
            )[..., 1]  # L136
            self._state["rbot"][..., m] = air_density(
                settings, dataset["tbot"][:, :, m], pf[:, :, 0]
            )
            self._state["thbot"][..., m] = potential_temperature(
                settings, dataset["tbot"][:, :, m], pf[:, :, 0]
            )

        for variable in (
            "zbot",
            "ubot",
            "vbot",
            "thbot",
            "qbot",
            "tbot",
            "rbot",
        ):
            self.shared_fields[variable] = get_values_at_specific_time(
                variable, self._state, coupler
            )

    def step(self, dt: timedelta, time: datetime, coupler: "Coupler") -> None:
        """Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        # This way we keep variables from different components in the same instance
        for variable in (
            "zbot",
            "ubot",
            "vbot",
            "thbot",
            "qbot",
            "tbot",
            "rbot",
        ):
            self.shared_fields[variable][...] = get_values_at_specific_time(
                variable, self._state, coupler, current_time=time
            )

    def finalize(self, coupler: "Coupler") -> None:
        pass