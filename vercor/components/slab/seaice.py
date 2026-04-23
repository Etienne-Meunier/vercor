from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


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
        self.data["ice_fraction"] = cast(
            Any, jnp.zeros(self.grid.shape, dtype=jnp.float64)
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

        self.data["ice_fraction"] = cast(Any, _diagnose_ice_fraction(sst))
