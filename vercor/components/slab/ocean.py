from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


_REFERENCE_SEA_SURFACE_TEMPERATURE = 273.15 + 15.0


@jax.jit
def _advance_sea_surface_temperature(
    sea_surface_temperature: object,
    sensible_heat_flux: object,
    latent_heat_flux: object,
    dt_seconds: float,
    rho: float,
    cp: float,
    mixed_layer_depth: float,
    lambda_relax: float,
    reference_temperature: float,
) -> jax.Array:
    sea_surface_temperature_array = jnp.asarray(
        sea_surface_temperature, dtype=jnp.float64
    )
    sensible_heat_flux_array = jnp.asarray(sensible_heat_flux, dtype=jnp.float64)
    latent_heat_flux_array = jnp.asarray(latent_heat_flux, dtype=jnp.float64)
    qnet = sensible_heat_flux_array + latent_heat_flux_array
    tendency = qnet / (rho * cp * mixed_layer_depth) + lambda_relax * (
        sea_surface_temperature_array - reference_temperature
    )
    return sea_surface_temperature_array + tendency * dt_seconds


class Ocean(Component):
    """Toy slab ocean: updates sea_surface_temperature using sensible_heat_flux (sensible) + latent_heat_flux (latent).
    Outputs: sea_surface_temperature [K]
    Inputs: sensible_heat_flux, latent_heat_flux
    """

    def __init__(
        self, grid: RectilinearGrid, name: str = "OCN", H: float = 30.0
    ) -> None:
        super().__init__(name, grid)

        self.H = H  # mixed-layer depth [m]
        self.rho = 1025.0
        self.cp = 3990.0
        self.lambda_relax = 1.0 / (
            30.0 * 86400.0
        )  # weak restoring to 15C over ~30 days

    def initialize(self, coupler: "Coupler") -> None:
        self.data["sea_surface_temperature"] = cast(
            Any,
            jnp.full(
                self.grid.shape,
                _REFERENCE_SEA_SURFACE_TEMPERATURE,
                dtype=jnp.float64,
            ),
        )

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        sst = self.data.get("sea_surface_temperature", None)
        if sst is None:
            return

        sst_array = jnp.asarray(sst, dtype=jnp.float64)
        SHF = self.data.get("sensible_heat_flux", None)
        LHF = self.data.get("latent_heat_flux", None)
        sensible_heat_flux = (
            jnp.zeros_like(sst_array)
            if SHF is None
            else jnp.asarray(SHF, dtype=jnp.float64)
        )
        latent_heat_flux = (
            jnp.zeros_like(sst_array)
            if LHF is None
            else jnp.asarray(LHF, dtype=jnp.float64)
        )

        self.data["sea_surface_temperature"] = cast(
            Any,
            _advance_sea_surface_temperature(
                sst_array,
                sensible_heat_flux,
                latent_heat_flux,
                float(dt.total_seconds()),
                self.rho,
                self.cp,
                self.H,
                self.lambda_relax,
                _REFERENCE_SEA_SURFACE_TEMPERATURE,
            ),
        )
