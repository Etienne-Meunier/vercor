from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vercor import (
    Clock,
    Component,
    Coupler,
    DataComponent,
    ComponentSpec,
    Exchange,
    ExecutionContext,
    HostComponent,
    RectilinearGrid,
    RuntimeDriver,
    RunState,
    StepContext,
    StepResult,
    RuntimeOptions,
)
from vercor.dtypes import as_jax_real_array


def make_example_grid() -> RectilinearGrid:
    """Return a small grid for custom component wrapper examples."""

    return RectilinearGrid.from_coordinates(
        "example-grid",
        longitude=as_jax_real_array([0.0, 90.0]),
        latitude=as_jax_real_array([-30.0, 30.0]),
    )


def make_data_forcing(grid: RectilinearGrid) -> DataComponent:
    """Wrap static or time-dependent forcing fields without a runtime step."""

    return DataComponent.from_fields(
        name="ATM",
        grid=grid,
        fields={
            "temperature": 288.15,
            "specific_humidity": 0.01,
        },
    ).update_settings(identifier="example-forcing")


def make_differentiable_model(grid: RectilinearGrid) -> Component:
    """Wrap a pure JAX callable as a differentiable VerCOR component."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        heat_capacity = 1025.0 * 3990.0 * 30.0
        tendency = fields["net_surface_heat_flux"] / heat_capacity
        return {
            "sea_surface_temperature": (
                fields["sea_surface_temperature"] + tendency * context.dt_seconds
            )
        }

    return Component.from_step(
        name="OCN",
        grid=grid,
        step=step,
        spec=ComponentSpec(
            inputs=("net_surface_heat_flux",),
            outputs=("sea_surface_temperature",),
            defaults={
                "sea_surface_temperature": 288.15,
                "net_surface_heat_flux": 0.0,
            },
        ),
    )


@dataclass
class ToyHostModel:
    """Small mutable host-side model used to show host wrapper payloads."""

    offset: float = 0.0

    def advance(self, temperature: Any, dt_seconds: float) -> Any:
        self.offset += 0.001 * dt_seconds
        return as_jax_real_array(temperature) + self.offset


def make_host_model(grid: RectilinearGrid) -> HostComponent:
    """Wrap a Python host-side model while keeping VerCOR runtime fields explicit."""

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None,
    ) -> StepResult:
        if not isinstance(payload, ToyHostModel):
            raise TypeError("Host wrapper payload must be a ToyHostModel")
        updated_temperature = payload.advance(fields["temperature"], context.dt_seconds)
        return StepResult(
            fields={"temperature": updated_temperature},
            payload=payload,
        )

    return HostComponent.from_step(
        name="LND",
        grid=grid,
        step=step,
        payload=ToyHostModel(),
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 283.15},
        ),
    )


@dataclass
class StructuralFluxModel:
    """Small structural component using the public ComponentLike contract."""

    grid: RectilinearGrid
    name: str = "MODEL"
    spec: ComponentSpec = ComponentSpec(
        inputs=("custom_flux",),
        outputs=("custom_flux",),
        defaults={"custom_flux": 0.0},
        execution="host",
    )

    def initial_fields(self) -> Mapping[str, Any]:
        """Return setup-time field seeds."""

        return {}

    def initialize(self, context: Any) -> None:
        """Perform setup-time initialization."""

        _ = context

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        """Update the custom flux on the host runtime path."""

        _ = payload
        return {"custom_flux": fields["custom_flux"] + context.step}


class SequentialBackend:
    """Minimal custom backend that delegates component stepping to RuntimeDriver."""

    def run(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        driver: RuntimeDriver,
    ) -> RunState:
        """Run components sequentially for every clock step."""

        for step, _, _ in context.clock.iter():
            for component in context.run_order:
                state = driver.step_component(state, component, step=step)
        return state


def make_custom_coupler(grid: RectilinearGrid) -> Coupler:
    """Assemble custom-named components without the built-in surface-mask policy."""

    source = DataComponent.from_fields(
        name="FORCING",
        grid=grid,
        fields={"custom_flux": 1.0},
    )

    model = StructuralFluxModel(grid)
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=3),
        components=(source, model),
        exchanges=(Exchange("FORCING", "MODEL", ("custom_flux",)),),
        run_order=("FORCING", "MODEL"),
        runtime=RuntimeOptions(execution=SequentialBackend()),
    )


if __name__ == "__main__":
    example_grid = make_example_grid()
    for component in (
        make_data_forcing(example_grid),
        make_differentiable_model(example_grid),
        make_host_model(example_grid),
    ):
        print(component)
    print(make_custom_coupler(example_grid))
