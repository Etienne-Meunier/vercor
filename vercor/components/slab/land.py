from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component
from vercor.dtypes import as_jax_real_array, jax_full
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.runtime.components import validate_runtime_grid_data_field

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


@jax.jit
def _update_soil_moisture(
    soil_moisture: object,
    latent_heat_flux: object,
    dt_seconds: float,
) -> jax.Array:
    soil_moisture_array = as_jax_real_array(soil_moisture)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
    evap = 1e-9 * latent_heat_flux_array
    return jnp.clip(soil_moisture_array - evap * dt_seconds, 0.0, 1.0)


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses latent_heat_flux sign as proxy).
    Outputs: soil_moisture [0..1]
    Inputs: latent_heat_flux (proxy for evaporation)
    """

    def __init__(self, grid: RectilinearGrid, name: str = "LND") -> None:
        super().__init__(name, grid)

    def initialize(self, context: ComponentInitContext) -> None:
        self.data["soil_moisture"] = jax_full(self.grid.shape, 0.3, context.settings)
        self.data["land_surface_temperature"] = jax_full(
            self.grid.shape, 288.15, context.settings
        )

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Validate slab-land runtime fields."""

        _ = contract
        validate_runtime_grid_data_field(
            self,
            component_state,
            "soil_moisture",
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance the slab land component on immutable runtime state."""

        dt_seconds = context.dt_seconds
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
