"""Exercise VerCOR extension points using only stable public modules."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from vercor.clock import Clock
from vercor.components import (
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupContext,
    StepContext,
    StepResult,
)
from vercor.coupling import Coupler
from vercor.exchanges import Exchange
from vercor.grids import RectilinearGrid
from vercor.output import OutputConfig, SnapshotContext
from vercor.regridding import bilinear
from vercor.runtime import ExecutionContext, RuntimeDriver
from vercor.runtime import RuntimeOptions
from vercor.state import RunState
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
)
from vercor.types import RuntimeArray


def _record_original_component(owner: Any, context: SetupContext) -> None:
    """Record that the lifecycle hook received the original plugin object."""

    if not isinstance(owner, StructuralJaxComponent):
        raise TypeError("lifecycle hook did not receive the original component")
    owner.lifecycle_events.append("hook-initialize")
    owner.lifecycle_owner_ids.append(id(owner))
    if context.run_order != ("FORCING", "JAX", "HOST"):
        raise ValueError("unexpected run order")


def _create_payload(owner: Any) -> RuntimeArray:
    """Create the payload replaced by every public ``StepResult``."""

    if not isinstance(owner, StructuralJaxComponent):
        raise TypeError("payload hook did not receive the original component")
    return jnp.asarray(0, dtype=jnp.int32)


def _write_snapshot(context: SnapshotContext) -> None:
    """Write one public snapshot-context payload for smoke verification."""

    value = float(jnp.asarray(context.state.field("temperature"))[0, 0])
    context.output_path.write_text(
        json.dumps(
            {
                "component": context.component.name,
                "temperature": value,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class StructuralJaxComponent:
    """Structural differentiable component implemented outside VerCOR."""

    name = "JAX"

    def __init__(self) -> None:
        self.grid = RectilinearGrid.uniform(
            "plugin-jax-grid",
            nlon=4,
            nlat=3,
            longitude=(0.0, 360.0),
            latitude=(-90.0, 90.0),
        )
        self.spec = ComponentSpec(
            inputs=("forcing",),
            outputs=("temperature",),
            defaults={"temperature": 0.0},
            lifecycle=LifecycleHooks(
                initialize=_record_original_component,
                create_payload=_create_payload,
            ),
            output=OutputConfig(snapshot_writer=_write_snapshot),
        )
        self.lifecycle_events: list[str] = []
        self.lifecycle_owner_ids: list[int] = []
        self._initial_fields: Mapping[str, RuntimeArray] = {
            "temperature": jnp.zeros(self.grid.shape)
        }

    def initial_fields(self) -> Mapping[str, RuntimeArray]:
        """Return the initial grid-shaped temperature."""

        return self._initial_fields

    def initialize(self, context: SetupContext) -> None:
        """Record structural user initialization."""

        _ = context
        self.lifecycle_events.append("user-initialize")

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> StepResult:
        """Update fields and replace the public runtime payload."""

        _ = context
        if payload is None:
            raise ValueError("JAX component payload was not initialized")
        payload_value = jnp.asarray(payload, dtype=jnp.int32)
        return StepResult(
            fields={
                "temperature": fields["temperature"] + fields["forcing"] + payload_value
            },
            payload=payload_value + 1,
        )


class StructuralHostComponent:
    """Structural host component implemented outside VerCOR."""

    name = "HOST"

    def __init__(self) -> None:
        self.grid = RectilinearGrid.uniform(
            "plugin-host-grid",
            nlon=2,
            nlat=2,
            longitude=(0.0, 360.0),
            latitude=(-90.0, 90.0),
        )
        self.spec = ComponentSpec(
            outputs=("host_value",),
            defaults={"host_value": 10.0},
            execution="host",
        )
        self._initial_fields: Mapping[str, RuntimeArray] = {
            "host_value": jnp.full(self.grid.shape, 10.0)
        }

    def initial_fields(self) -> Mapping[str, RuntimeArray]:
        """Return the initial host field."""

        return self._initial_fields

    def initialize(self, context: SetupContext) -> None:
        """Accept the standard structural initialization context."""

        _ = context

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        """Advance the host field through the public mapping contract."""

        _ = context, payload
        return {"host_value": fields["host_value"] + 2.0}


class SequentialBackend:
    """Sequential custom backend using only the public runtime driver."""

    def __init__(self) -> None:
        self.calls = 0
        self.state_replacement = False

    def run(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        driver: RuntimeDriver,
    ) -> RunState:
        """Advance every configured component for every clock step."""

        initial_temperature = state.component("JAX").field("temperature")
        state = state.replace_fields(
            "JAX",
            {"temperature": jnp.full_like(initial_temperature, 10.0)},
        )
        self.state_replacement = True
        self.calls += 1
        for step in range(context.clock.steps):
            for component in context.run_order:
                state = driver.step_component(state, component, step=step)
        return state


class RecordingTopologyPolicy:
    """Custom topology policy returning an empty public patch."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def applies(self, context: TopologyContext) -> bool:
        """Record policy selection for the plugin component graph."""

        self.events.append("applies")
        return tuple(context.components) == ("FORCING", "JAX", "HOST")

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Record policy construction and return an empty patch."""

        _ = context
        self.events.append("build")
        return ExchangeTopologyPatch()


def run_smoke(output_dir: Path) -> dict[str, object]:
    """Run all required public extension points and return compact evidence."""

    jax_component = StructuralJaxComponent()
    host_component = StructuralHostComponent()
    forcing_component = DataComponent.from_fields(
        "FORCING",
        RectilinearGrid.uniform(
            "plugin-forcing-grid",
            nlon=2,
            nlat=2,
            longitude=(0.0, 360.0),
            latitude=(-90.0, 90.0),
        ),
        {"forcing": 1.0},
    )
    backend = SequentialBackend()
    topology = RecordingTopologyPolicy()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=2),
        components=(forcing_component, jax_component, host_component),
        exchanges=(
            Exchange(
                "FORCING",
                "JAX",
                ("forcing",),
                regrid=bilinear,
            ),
        ),
        run_order=(forcing_component.name, jax_component.name, host_component.name),
        runtime=RuntimeOptions(execution=backend, topology=topology),
    )

    final_state = coupler.run()
    coupler.write_outputs(final_state, output_dir=output_dir)

    temperature = float(
        jnp.asarray(final_state.component("JAX").field("temperature"))[0, 0]
    )
    host_value = float(
        jnp.asarray(final_state.component("HOST").field("host_value"))[0, 0]
    )
    exchange_forcing = float(
        jnp.asarray(final_state.component("JAX").field("forcing", scope="received"))[
            0, 0
        ]
    )
    snapshot_path = output_dir / "jax.snapshot.nc"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    if jax_component.lifecycle_events != ["user-initialize", "hook-initialize"]:
        raise AssertionError("structural lifecycle order was not preserved")
    if jax_component.lifecycle_owner_ids != [id(jax_component)]:
        raise AssertionError("lifecycle hook did not receive the original object")
    if backend.calls != 1:
        raise AssertionError("custom backend was not invoked exactly once")
    if topology.events != ["applies", "build"]:
        raise AssertionError("custom topology policy was not applied")
    if temperature != 13.0 or host_value != 14.0 or exchange_forcing != 1.0:
        raise AssertionError("sequential backend produced unexpected fields")
    if not backend.state_replacement:
        raise AssertionError("public RunState.replace_fields was not exercised")

    return {
        "backend_calls": backend.calls,
        "exchange_forcing": exchange_forcing,
        "host_value": host_value,
        "lifecycle": tuple(jax_component.lifecycle_events),
        "snapshot": snapshot,
        "state_replacement": backend.state_replacement,
        "temperature": temperature,
        "topology": tuple(topology.events),
    }


__all__ = [
    "RecordingTopologyPolicy",
    "SequentialBackend",
    "StructuralHostComponent",
    "StructuralJaxComponent",
    "run_smoke",
]
