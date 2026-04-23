from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0


@jax.jit
def _default_sea_surface_temperature(temperature_2m: object) -> jax.Array:
    return jnp.full_like(
        jnp.asarray(temperature_2m, dtype=jnp.float64),
        _REFERENCE_SURFACE_TEMPERATURE,
    )


@jax.jit
def _bulk_flux_step(
    temperature_2m: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    temperature_2m_array = jnp.asarray(temperature_2m, dtype=jnp.float64)
    sea_surface_temperature_array = jnp.asarray(
        sea_surface_temperature, dtype=jnp.float64
    )
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
    latitude_array = jnp.asarray(latitude, dtype=jnp.float64)
    longitude_array = jnp.asarray(longitude, dtype=jnp.float64) - 180.0
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

    def initialize(self, coupler: "Coupler") -> None:
        grid_shape = self.grid.shape
        zeros = jnp.zeros(grid_shape, dtype=jnp.float64)

        self.data["temperature_2m"] = jnp.full(
            grid_shape, _REFERENCE_SURFACE_TEMPERATURE, dtype=jnp.float64
        )
        self.data["sensible_heat_flux"] = zeros
        self.data["latent_heat_flux"] = zeros
        self.data["u_velocity_10m"] = zeros
        self.data["v_velocity_10m"] = zeros

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        sst = self.data.get("sea_surface_temperature", None)
        temperature_2m = jnp.asarray(self.data["temperature_2m"], dtype=jnp.float64)

        if sst is None:
            sst_array = _default_sea_surface_temperature(temperature_2m)
        else:
            sst_array = jnp.asarray(sst, dtype=jnp.float64)

        sensible_heat_flux, latent_heat_flux, updated_temperature_2m = _bulk_flux_step(
            temperature_2m, sst_array
        )
        u_velocity_10m, v_velocity_10m = _surface_wind_10m(
            self.grid.latitude, self.grid.longitude
        )

        self.data["u_velocity_10m"] = u_velocity_10m
        self.data["v_velocity_10m"] = v_velocity_10m
        self.data["sensible_heat_flux"] = sensible_heat_flux
        self.data["latent_heat_flux"] = latent_heat_flux
        self.data["temperature_2m"] = updated_temperature_2m
