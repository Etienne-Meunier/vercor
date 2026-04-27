from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.clock import ModelDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import RuntimeComponentState


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
        self.data["sea_surface_temperature"] = jnp.full(
            self.grid.shape,
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
            dtype=jnp.float64,
        )

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: "Coupler",
    ) -> None:
        component_state = self.step_runtime_state(
            self.to_runtime_component_state(prefill_missing=True),
            float(dt.total_seconds()),
            coupler.settings,
            time=time,
            coupler=coupler,
        )
        self.commit_runtime_state(component_state, time)

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate slab-ocean runtime fields."""

        super().validate_runtime_state(component_state, expected_shape)
        self._validate_runtime_grid_data_field(
            component_state,
            "sea_surface_temperature",
            expected_shape,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: datetime | ModelDateTime | None = None,
        coupler: "Coupler | None" = None,
    ) -> "RuntimeComponentState":
        """Advance the slab ocean on immutable runtime state."""

        _ = runtime_settings, time, coupler
        data = component_state.data
        try:
            sea_surface_temperature = data.get("sea_surface_temperature")
        except KeyError:
            return component_state
        try:
            sensible_heat_flux = data.get("sensible_heat_flux")
        except KeyError:
            sensible_heat_flux = jnp.zeros_like(sea_surface_temperature)
        try:
            latent_heat_flux = data.get("latent_heat_flux")
        except KeyError:
            latent_heat_flux = jnp.zeros_like(sea_surface_temperature)

        updated_sst = _advance_sea_surface_temperature(
            sea_surface_temperature,
            sensible_heat_flux,
            latent_heat_flux,
            dt_seconds,
            self.rho,
            self.cp,
            self.H,
            self.lambda_relax,
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
        )
        return component_state.with_data(
            data.set("sea_surface_temperature", updated_sst)
        )
