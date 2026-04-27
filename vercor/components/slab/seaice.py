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

    def initialize(self, coupler: "Coupler") -> None:
        self.data["ice_fraction"] = jnp.zeros(self.grid.shape, dtype=jnp.float64)

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
        """Validate slab-sea-ice runtime fields."""

        super().validate_runtime_state(component_state, expected_shape)
        for field_name in ("ice_fraction", "sea_surface_temperature"):
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
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
        """Diagnose slab sea-ice fraction on immutable runtime state."""

        _ = dt_seconds, runtime_settings, time, coupler
        data = component_state.data
        try:
            sea_surface_temperature = data.get("sea_surface_temperature")
        except KeyError:
            return component_state
        ice_fraction = _diagnose_ice_fraction(sea_surface_temperature)
        return component_state.with_data(data.set("ice_fraction", ice_fraction))
