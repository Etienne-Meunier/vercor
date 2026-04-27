from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import RuntimeComponentState


@jax.jit
def _update_soil_moisture(
    soil_moisture: object,
    latent_heat_flux: object,
    dt_seconds: float,
) -> jax.Array:
    soil_moisture_array = jnp.asarray(soil_moisture, dtype=jnp.float64)
    latent_heat_flux_array = jnp.asarray(latent_heat_flux, dtype=jnp.float64)
    evap = 1e-9 * latent_heat_flux_array
    return jnp.clip(soil_moisture_array - evap * dt_seconds, 0.0, 1.0)


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses latent_heat_flux sign as proxy).
    Outputs: soil_moisture [0..1]
    Inputs: latent_heat_flux (proxy for evaporation)
    """

    def __init__(self, grid: RectilinearGrid, name: str = "LND") -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        self.data["soil_moisture"] = jnp.full(self.grid.shape, 0.3, dtype=jnp.float64)
        self.data["land_surface_temperature"] = jnp.full(
            self.grid.shape, 288.15, dtype=jnp.float64
        )

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        latent_heat_flux = self.data["latent_heat_flux"]
        soil_moisture = self.data["soil_moisture"]

        self.data["soil_moisture"] = _update_soil_moisture(
            soil_moisture,
            latent_heat_flux if latent_heat_flux is not None else 0.0,
            float(dt.total_seconds()),
        )

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate slab-land runtime fields."""

        self._validate_runtime_grid_data_field(
            component_state,
            "soil_moisture",
            expected_shape,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: datetime | CustomDateTime | None = None,
        coupler: "Coupler | None" = None,
    ) -> "RuntimeComponentState":
        """Advance the slab land component on immutable runtime state."""

        _ = runtime_settings, time, coupler
        data = component_state.data
        soil_moisture = data.get("soil_moisture")
        try:
            latent_heat_flux = data.get("latent_heat_flux")
        except KeyError:
            latent_heat_flux = jnp.zeros_like(soil_moisture)
        updated_soil_moisture = _update_soil_moisture(
            soil_moisture,
            latent_heat_flux,
            dt_seconds,
        )
        return component_state.with_data(
            data.set("soil_moisture", updated_soil_moisture)
        )
