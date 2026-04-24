from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.clock import CustomDateTime
from vercor.components import Component, ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


def _ocean_binary_mask_from_land_fraction(land_fraction: ArrayLike) -> jax.Array:
    """Convert a fractional land mask into a binary ocean mask."""
    land_fraction_array = jnp.asarray(land_fraction)
    return 1.0 - jnp.where(land_fraction_array > 0.0, 1.0, 0.0)


def _mask_sea_surface_temperature(
    sea_surface_temperature: ArrayLike,
    binary_mask: ArrayLike,
) -> jax.Array:
    """Apply the binary ocean mask to a `(nlon, nlat, time)` SST field."""
    return (
        jnp.asarray(sea_surface_temperature)
        * jnp.where(
            jnp.asarray(binary_mask) > 0.0,
            1.0,
            jnp.nan,
        ).T[..., jnp.newaxis]
    )


class ERA5Ocean(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "OCN",
        surface_file: Optional[Path] = None,
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            surface_file (Path): path to netCDF file with data at surface level

        Attributes of parent classes to be initialized:
            ComponentForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
        """

        if surface_file is None:
            surface_file = get_forcing_data("era5_surface")

        self.DATA_FILES = {
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("longitude", where="surface")
        latitude = self._read_forcing("latitude", where="surface")[::-1]
        land_fraction = self._read_forcing("lsm", where="surface", flip_y=True).T[0, ::]
        binary_mask = _ocean_binary_mask_from_land_fraction(land_fraction)

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self.settings.apply_time_interpolation = True

        # Units: [K]
        self.data["sea_surface_temperature"] = _mask_sea_surface_temperature(
            self._read_forcing("sst", where="surface", flip_y=True),
            binary_mask,
        )

    def initialize(self, coupler: "Coupler") -> None:
        pass

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        """
        Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        pass
