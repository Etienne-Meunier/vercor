from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.base import DataComponent
from vercor.forcing_data import ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.assets import get_forcing_data


def _prepare_era5_land_runtime_fields(
    longitude: ArrayLike,
    latitude: ArrayLike,
    binary_mask: ArrayLike,
    land_surface_temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Normalize ERA5 land forcing arrays for JAX-backed runtime storage."""
    return (
        jnp.asarray(longitude),
        jnp.asarray(latitude),
        jnp.asarray(binary_mask).T,
        jnp.asarray(land_surface_temperature),
    )


class ERA5Land(DataComponent, ComponentForcingData):
    def __init__(
        self,
        name: str = "LND",
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
            surface_file = get_forcing_data("era5_land_masked")

        self.DATA_FILES = {
            "surface": str(surface_file),
        }

        (
            longitude,
            latitude,
            binary_mask,
            land_surface_temperature,
        ) = _prepare_era5_land_runtime_fields(
            self._read_forcing("lon", where="surface"),
            self._read_forcing("lat", where="surface"),
            self._read_forcing("mask", where="surface"),
            self._read_forcing("skt", where="surface"),
        )
        grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=grid)

        self.settings.apply_time_interpolation = True

        # Units: [K]
        self.data["land_surface_temperature"] = land_surface_temperature
