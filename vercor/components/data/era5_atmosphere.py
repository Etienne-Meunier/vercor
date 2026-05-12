from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.base import DataComponent
from vercor.dtypes import as_jax_real_array
from vercor.field_layout import (
    canonicalize_time_last_level_field,
    canonicalize_time_last_surface_field,
)
from vercor.fluxes.utilities import (
    compute_air_density,
    get_altitudes_hybrid_sigma_levels,
    compute_pressure_levels,
    compute_potential_temperature,
)
from vercor.forcing_data import ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext
from vercor.settings import VercorSettings
from vercor.assets import get_forcing_data

_ERA5_ATMOSPHERE_FIELD_NAMES = (
    "surface_pressure",
    "specific_humidity_3d",
    "temperature_3d",
    "u_velocity",
    "v_velocity",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "specific_humidity",
    "temperature",
    "model_level_height",
    "density",
    "potential_temperature",
)


def _decode_surface_pressure(lnsp: ArrayLike) -> jax.Array:
    """Convert log surface pressure to physical pressure in Pascals."""
    return jnp.exp(as_jax_real_array(lnsp))


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

    surface_pressure_array = as_jax_real_array(surface_pressure, settings)
    temperature_3d_array = as_jax_real_array(temperature_3d, settings).transpose(
        (1, 2, 0)
    )
    specific_humidity_3d_array = as_jax_real_array(
        specific_humidity_3d,
        settings,
    ).transpose((1, 2, 0))
    temperature_array = as_jax_real_array(temperature, settings)
    hyai_array = as_jax_real_array(hyai, settings)
    hybi_array = as_jax_real_array(hybi, settings)
    hyam_array = as_jax_real_array(hyam, settings)
    hybm_array = as_jax_real_array(hybm, settings)

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


class ERA5Atmosphere(DataComponent, ComponentForcingData):
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
        self.declare_fields(outputs=_ERA5_ATMOSPHERE_FIELD_NAMES)

        self.update_settings(apply_time_interpolation=True)

        self.hyai: jax.Array = as_jax_real_array(
            self._read_forcing("hyai", where="model_level")[-3:]
        )  # L135-L137
        self.hybi: jax.Array = as_jax_real_array(
            self._read_forcing("hybi", where="model_level")[-3:]
        )  # L135-L137
        self.hyam: jax.Array = as_jax_real_array(
            self._read_forcing("hyam", where="model_level")[-2:]
        )  # L136-L137
        self.hybm: jax.Array = as_jax_real_array(
            self._read_forcing("hybm", where="model_level")[-2:]
        )  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        # Units: [Pa]
        surface_pressure = _decode_surface_pressure(
            canonicalize_time_last_surface_field(lnsp)
        )
        # Units: [kg/kg]
        specific_humidity_3d = canonicalize_time_last_level_field(
            self._read_forcing("q", where="model_level", flip_y=True)[
                ..., 1:, :
            ]  # L136-L137
        )
        # Units: [K]
        temperature_3d = canonicalize_time_last_level_field(
            self._read_forcing("t", where="model_level", flip_y=True)[
                ..., 1:, :
            ]  # L136-L137
        )
        self.seed_fields(
            {
                "surface_pressure": surface_pressure,
                "specific_humidity_3d": specific_humidity_3d,
                "temperature_3d": temperature_3d,
                # Units: [m/s], L136
                "u_velocity": canonicalize_time_last_surface_field(
                    self._read_forcing("u", where="model_level", flip_y=True)[
                        :, :, 1, :
                    ]
                ),
                # Units: [m/s], L136
                "v_velocity": canonicalize_time_last_surface_field(
                    self._read_forcing("v", where="model_level", flip_y=True)[
                        :, :, 1, :
                    ]
                ),
                # Units: [W/m²]
                "net_shortwave_radiation_flux": (
                    canonicalize_time_last_surface_field(
                        self._read_forcing("msnswrf", where="surface", flip_y=True)
                    )
                ),
                # Units: [W/m²]
                "downward_longwave_radiation_flux": (
                    canonicalize_time_last_surface_field(
                        self._read_forcing("msdwlwrf", where="surface", flip_y=True)
                    )
                ),
                # Units: [kg/kg], L136
                "specific_humidity": specific_humidity_3d[:, 0, :, :],
                # Units: [K], L136
                "temperature": temperature_3d[:, 0, :, :],
            }
        )

    def initialize(self, context: ComponentInitContext) -> None:
        diagnostics = [
            _compute_monthly_diagnostics(
                context.settings,
                self.data["surface_pressure"][month_index],
                self.hyai,
                self.hybi,
                self.hyam,
                self.hybm,
                self.data["temperature_3d"][month_index],
                self.data["specific_humidity_3d"][month_index],
                self.data["temperature"][month_index],
            )
            for month_index in range(int(self.data["surface_pressure"].shape[0]))
        ]
        self.seed_fields(
            {
                "model_level_height": jnp.stack(
                    [item[0] for item in diagnostics],
                    axis=0,
                ),
                "density": jnp.stack([item[1] for item in diagnostics], axis=0),
                "potential_temperature": jnp.stack(
                    [item[2] for item in diagnostics],
                    axis=0,
                ),
            }
        )
