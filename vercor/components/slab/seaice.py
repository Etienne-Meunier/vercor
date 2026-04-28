from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component, ComponentInitContext, RuntimeStepContext
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


@jax.jit
def _diagnose_ice_fraction(sea_surface_temperature: object) -> jax.Array:
    sea_surface_temperature_array = jnp.asarray(
        sea_surface_temperature, dtype=jnp.float64
    )
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

    def initialize(self, context: ComponentInitContext) -> None:
        _ = context
        self.data["ice_fraction"] = jnp.zeros(self.grid.shape, dtype=jnp.float64)

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract | None" = None,
    ) -> None:
        """Validate slab-sea-ice runtime fields."""

        super().validate_runtime_state(component_state, contract)
        for field_name in ("ice_fraction", "sea_surface_temperature"):
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
            )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Diagnose slab sea-ice fraction on immutable runtime state."""

        _ = context
        data = component_state.data
        try:
            sea_surface_temperature = data.get("sea_surface_temperature")
        except KeyError:
            return component_state
        ice_fraction = _diagnose_ice_fraction(sea_surface_temperature)
        return component_state.with_data(data.set("ice_fraction", ice_fraction))
