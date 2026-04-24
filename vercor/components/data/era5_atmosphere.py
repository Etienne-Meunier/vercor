from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.clock import CustomDateTime
from vercor.components import Component, ComponentForcingData
from vercor.fluxes.utilities import (
    compute_air_density,
    get_altitudes_hybrid_sigma_levels,
    compute_pressure_levels,
    compute_potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


def _decode_surface_pressure(lnsp: ArrayLike) -> jax.Array:
    """Convert log surface pressure to physical pressure in Pascals."""
    return jnp.exp(jnp.asarray(lnsp))


def _compute_monthly_diagnostics(
    settings: VercorSettings,
    surface_pressure: ArrayLike,
    hyai: ArrayLike,
    hybi: ArrayLike,
    hyam: ArrayLike,
    hybm: ArrayLike,
    temperature_3d: ArrayLike,
    specific_humidity_3d: ArrayLike,
    temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ERA5 diagnostics for one monthly slice on the runtime JAX path."""

    surface_pressure_array = jnp.asarray(surface_pressure)
    temperature_3d_array = jnp.asarray(temperature_3d)
    specific_humidity_3d_array = jnp.asarray(specific_humidity_3d)
    temperature_array = jnp.asarray(temperature)
    hyai_array = jnp.asarray(hyai)
    hybi_array = jnp.asarray(hybi)
    hyam_array = jnp.asarray(hyam)
    hybm_array = jnp.asarray(hybm)

    ph = compute_pressure_levels(surface_pressure_array, hyai_array, hybi_array)
    pf = compute_pressure_levels(surface_pressure_array, hyam_array, hybm_array)
    model_level_height = get_altitudes_hybrid_sigma_levels(
        settings,
        temperature_3d_array,
        specific_humidity_3d_array,
        ph,
    )[..., 1]
    density = compute_air_density(settings, pf[..., 0], temperature_array)
    potential_temperature = compute_potential_temperature(
        settings,
        temperature_array,
        pf[..., 0],
    )

    return model_level_height, density, potential_temperature


def _combine_surface_temperatures(
    land_surface_temperature: ArrayLike,
    sea_surface_temperature: ArrayLike,
) -> jax.Array:
    """Merge land and sea surface temperatures while treating NaNs as missing."""
    return jnp.nan_to_num(
        jnp.asarray(land_surface_temperature), nan=0.0
    ) + jnp.nan_to_num(
        jnp.asarray(sea_surface_temperature),
        nan=0.0,
    )


class ERA5Atmosphere(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "ATM",
        model_level_file: Optional[Path] = None,
        surface_file: Optional[Path] = None,
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
            ComponentForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
        """

        if model_level_file is None:
            model_level_file = get_forcing_data("era5_model_levels")
        if surface_file is None:
            surface_file = get_forcing_data("era5_surface")

        self.DATA_FILES = {
            "model_level": str(model_level_file),
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("longitude", where="model_level")
        latitude = self._read_forcing("latitude", where="model_level")[::-1]

        grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
        )

        super().__init__(name, grid=grid)

        self.settings.apply_time_interpolation = True

        self.data["hyai"] = self._read_forcing("hyai", where="model_level")[
            -3:
        ]  # L135-L137
        self.data["hybi"] = self._read_forcing("hybi", where="model_level")[
            -3:
        ]  # L135-L137
        self.data["hyam"] = self._read_forcing("hyam", where="model_level")[
            -2:
        ]  # L136-L137
        self.data["hybm"] = self._read_forcing("hybm", where="model_level")[
            -2:
        ]  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        # Units: [Pa]
        self.data["surface_pressure"] = _decode_surface_pressure(lnsp)
        # Units: [kg/kg]
        self.data["specific_humidity_3d"] = self._read_forcing(
            "q", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        # Units: [K]
        self.data["temperature_3d"] = self._read_forcing(
            "t", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        # Units: [m/s]
        self.data["u_velocity"] = self._read_forcing(
            "u", where="model_level", flip_y=True
        )[
            :, :, 1, :
        ]  # L136
        # Units: [m/s]
        self.data["v_velocity"] = self._read_forcing(
            "v", where="model_level", flip_y=True
        )[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", where="surface", flip_y=True)
        # Units: [W/m²]
        self.data["net_shortwave_radiation_flux"] = self._read_forcing(
            "msnswrf", where="surface", flip_y=True
        )
        # Units: [W/m²]
        self.data["downward_longwave_radiation_flux"] = self._read_forcing(
            "msdwlwrf", where="surface", flip_y=True
        )
        # Units: [kg/kg]
        self.data["specific_humidity"] = self.data["specific_humidity_3d"][
            ..., 0, :
        ]  # L136
        # Units: [K]
        self.data["temperature"] = self.data["temperature_3d"][..., 0, :]  # L136

    def initialize(self, coupler: "Coupler") -> None:
        diagnostics = [
            _compute_monthly_diagnostics(
                coupler.settings,
                self.data["surface_pressure"][..., month_index],
                self.data["hyai"],
                self.data["hybi"],
                self.data["hyam"],
                self.data["hybm"],
                self.data["temperature_3d"][..., month_index],
                self.data["specific_humidity_3d"][..., month_index],
                self.data["temperature"][..., month_index],
            )
            for month_index in range(int(self.data["surface_pressure"].shape[-1]))
        ]
        self.data["model_level_height"] = jnp.stack(
            [item[0] for item in diagnostics],
            axis=-1,
        )
        self.data["density"] = jnp.stack([item[1] for item in diagnostics], axis=-1)
        self.data["potential_temperature"] = jnp.stack(
            [item[2] for item in diagnostics],
            axis=-1,
        )

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
        # Units: [K]
        self.data["total_surface_temperature"] = _combine_surface_temperatures(
            self.data["land_surface_temperature"],
            self.data["sea_surface_temperature"],
        )
