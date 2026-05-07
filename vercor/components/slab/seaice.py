from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


@jax.jit
def _diagnose_ice_fraction(sea_surface_temperature: object) -> jax.Array:
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    freezing_temperature = 273.15 - 1.8
    x = (freezing_temperature - sea_surface_temperature_array) / 2.0
    return 1.0 / (1.0 + jnp.exp(-x))


class SeaIce(Component):
    """Toy thermodynamic sea-ice: diagnostic concentration from sea_surface_temperature.
    Outputs: ice_fraction [0..1]
    Inputs: sea_surface_temperature [K]
    """

    def __init__(self, grid: RectilinearGrid, name: str = "ICE") -> None:
        super().__init__(name, grid)
        self.declare_fields(
            inputs=("sea_surface_temperature",),
            outputs=("ice_fraction",),
        )

    def initialize(self, context: ComponentInitContext) -> None:
        self.seed_field("ice_fraction", 0.0, context.settings)

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Diagnose slab sea-ice fraction on immutable runtime state."""

        _ = context
        if not self.has_runtime_field(component_state, "sea_surface_temperature"):
            return component_state
        sea_surface_temperature = self.runtime_field(
            component_state,
            "sea_surface_temperature",
        )
        ice_fraction = _diagnose_ice_fraction(sea_surface_temperature)
        return self.with_runtime_fields(
            component_state,
            {"ice_fraction": ice_fraction},
        )
