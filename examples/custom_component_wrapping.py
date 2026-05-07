from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from vercor import Component, ComponentStepResult, DataComponent, HostRuntimeComponent
from vercor.grid import RectilinearGrid


def make_example_grid() -> RectilinearGrid:
    """Return a small grid for custom component wrapper examples."""

    return RectilinearGrid(
        name="example-grid",
        longitude=jnp.asarray([0.0, 90.0]),
        latitude=jnp.asarray([-30.0, 30.0]),
    )


def make_data_forcing(grid: RectilinearGrid) -> DataComponent:
    """Wrap static or time-dependent forcing fields without a runtime step."""

    return DataComponent.wrap(
        name="ATM",
        grid=grid,
        fields={
            "temperature": jnp.full(grid.shape, 288.15),
            "specific_humidity": jnp.full(grid.shape, 0.01),
        },
    )


def make_differentiable_model(grid: RectilinearGrid) -> Component:
    """Wrap a pure JAX callable as a differentiable VerCOR component."""

    def step(
        fields: Mapping[str, Any],
        context: Any,
        payload: Any | None,
    ) -> Mapping[str, Any]:
        _ = payload
        heat_capacity = 1025.0 * 3990.0 * 30.0
        tendency = fields["net_surface_heat_flux"] / heat_capacity
        return {
            "sea_surface_temperature": (
                fields["sea_surface_temperature"] + tendency * context.dt_seconds
            )
        }

    return Component.wrap(
        name="OCN",
        grid=grid,
        step=step,
        fields={"sea_surface_temperature": jnp.full(grid.shape, 288.15)},
        required_fields=("net_surface_heat_flux",),
        prefill_fields=("net_surface_heat_flux",),
    )


@dataclass
class ToyHostModel:
    """Small mutable host-side model used to show host wrapper payloads."""

    offset: float = 0.0

    def advance(self, temperature: Any, dt_seconds: float) -> Any:
        self.offset += 0.001 * dt_seconds
        return jnp.asarray(temperature) + self.offset


def make_host_model(grid: RectilinearGrid) -> HostRuntimeComponent:
    """Wrap a Python host-side model while keeping VerCOR runtime fields explicit."""

    def step(
        fields: Mapping[str, Any],
        context: Any,
        payload: Any | None,
    ) -> ComponentStepResult:
        if not isinstance(payload, ToyHostModel):
            raise TypeError("Host wrapper payload must be a ToyHostModel")
        updated_temperature = payload.advance(fields["temperature"], context.dt_seconds)
        return ComponentStepResult(
            fields={"temperature": updated_temperature},
            payload=payload,
        )

    return HostRuntimeComponent.wrap(
        name="LND",
        grid=grid,
        step=step,
        fields={"temperature": jnp.full(grid.shape, 283.15)},
        payload=ToyHostModel(),
        required_fields=("temperature",),
    )


if __name__ == "__main__":
    example_grid = make_example_grid()
    for component in (
        make_data_forcing(example_grid),
        make_differentiable_model(example_grid),
        make_host_model(example_grid),
    ):
        print(component)
