from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component, ComponentFieldSpec
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


_LAND_FIELD_SPEC = ComponentFieldSpec(
    inputs=("latent_heat_flux",),
    outputs=("soil_moisture", "land_surface_temperature"),
    default_fields={"soil_moisture": 0.3, "land_surface_temperature": 288.15},
)


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
        self.declare_fields(_LAND_FIELD_SPEC)

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance the slab land component on immutable runtime state."""

        dt_seconds = context.dt_seconds
        soil_moisture = self.runtime_field(component_state, "soil_moisture")
        latent_heat_flux = self.runtime_field_or_zeros_like(
            component_state,
            "latent_heat_flux",
            soil_moisture,
        )
        updated_soil_moisture = _update_soil_moisture(
            soil_moisture,
            latent_heat_flux,
            dt_seconds,
        )
        return self.with_runtime_fields(
            component_state,
            {"soil_moisture": updated_soil_moisture},
        )
