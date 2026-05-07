from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0


@jax.jit
def _default_sea_surface_temperature(temperature_2m: object) -> jax.Array:
    return jnp.full_like(
        as_jax_real_array(temperature_2m),
        _REFERENCE_SURFACE_TEMPERATURE,
    )


@jax.jit
def _bulk_flux_step(
    temperature_2m: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    temperature_2m_array = as_jax_real_array(temperature_2m)
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    delta_temperature = temperature_2m_array - sea_surface_temperature_array
    sensible_heat_flux = -10.0 * delta_temperature
    latent_heat_flux = -0.5 * sensible_heat_flux
    updated_temperature_2m = temperature_2m_array - 0.01 * delta_temperature
    return sensible_heat_flux, latent_heat_flux, updated_temperature_2m


@jax.jit
def _surface_wind_10m(
    latitude: object,
    longitude: object,
) -> tuple[jax.Array, jax.Array]:
    latitude_array = as_jax_real_array(latitude)
    longitude_array = as_jax_real_array(longitude) - 180.0
    latitudes, longitudes = jnp.meshgrid(latitude_array, longitude_array, indexing="ij")
    u_velocity_10m = jnp.cos(jnp.deg2rad(latitudes))
    v_velocity_10m = 0.5 * jnp.sin(jnp.deg2rad(longitudes))
    return u_velocity_10m, v_velocity_10m


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from sea_surface_temperature.
    Inputs: sea_surface_temperature [K]
    Outputs: sensible_heat_flux [W/m2], latent_heat_flux [W/m2], temperature_2m [K]
    """

    def __init__(self, grid: RectilinearGrid, name: str = "ATM") -> None:
        super().__init__(name, grid)
        self.declare_fields(
            inputs=("sea_surface_temperature",),
            outputs=(
                "temperature_2m",
                "sensible_heat_flux",
                "latent_heat_flux",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
            default_fields={"temperature_2m": _REFERENCE_SURFACE_TEMPERATURE},
        )

    def initialize(self, context: ComponentInitContext) -> None:
        self.seed_fields(
            {
                "temperature_2m": _REFERENCE_SURFACE_TEMPERATURE,
                "sensible_heat_flux": 0.0,
                "latent_heat_flux": 0.0,
                "u_velocity_10m": 0.0,
                "v_velocity_10m": 0.0,
            },
            context.settings,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance the slab atmosphere on immutable runtime state."""

        _ = context
        temperature_2m = self.runtime_field(component_state, "temperature_2m")
        sea_surface_temperature = self.runtime_field_or(
            component_state,
            "sea_surface_temperature",
            _REFERENCE_SURFACE_TEMPERATURE,
        )

        sensible_heat_flux, latent_heat_flux, updated_temperature_2m = _bulk_flux_step(
            temperature_2m,
            sea_surface_temperature,
        )
        u_velocity_10m, v_velocity_10m = _surface_wind_10m(
            self.grid.latitude,
            self.grid.longitude,
        )
        return self.with_runtime_fields(
            component_state,
            {
                "u_velocity_10m": u_velocity_10m,
                "v_velocity_10m": v_velocity_10m,
                "sensible_heat_flux": sensible_heat_flux,
                "latent_heat_flux": latent_heat_flux,
                "temperature_2m": updated_temperature_2m,
            },
        )
