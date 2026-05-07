from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vercor.components.base import Component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


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
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    sensible_heat_flux_array = as_jax_real_array(sensible_heat_flux)
    latent_heat_flux_array = as_jax_real_array(latent_heat_flux)
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

    def initialize(self, context: ComponentInitContext) -> None:
        self.seed_constant_field(
            "sea_surface_temperature",
            _REFERENCE_SEA_SURFACE_TEMPERATURE,
            context.settings,
        )

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Validate slab-ocean runtime fields."""

        _ = contract
        self.require_runtime_fields(component_state, "sea_surface_temperature")

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance the slab ocean on immutable runtime state."""

        dt_seconds = context.dt_seconds
        if "sea_surface_temperature" not in component_state.data.field_names:
            return component_state
        sea_surface_temperature = self.runtime_field(
            component_state,
            "sea_surface_temperature",
        )
        if "sensible_heat_flux" in component_state.data.field_names:
            sensible_heat_flux = self.runtime_field(
                component_state,
                "sensible_heat_flux",
            )
        else:
            sensible_heat_flux = jnp.zeros_like(sea_surface_temperature)
        if "latent_heat_flux" in component_state.data.field_names:
            latent_heat_flux = self.runtime_field(component_state, "latent_heat_flux")
        else:
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
        return self.with_runtime_fields(
            component_state,
            {"sea_surface_temperature": updated_sst},
        )
