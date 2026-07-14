"""Shared construction helpers for the VerCOR 4 workflow tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import vercor.runtime as runtime
from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.output import OutputSpec
from vercor.state import RunState


def make_clock(steps: int = 2, *, dt_seconds: float = 60.0) -> Clock:
    """Build the compact deterministic clock shared by workflow tests."""
    return Clock(
        start=datetime(2000, 1, 1),
        dt_seconds=dt_seconds,
        steps=steps,
    )


def make_component(
    name: str,
    *,
    execution: str = "jax",
    observed: list[tuple[str, int]] | None = None,
    output: OutputSpec | None = None,
) -> CallableComponent:
    """Build the scalar increment component shared by workflow tests."""
    grid = make_test_grid(name=f"workflow-{name}")

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        if observed is not None:
            observed.append((name, int(context.step)))
        return {"value": fields["value"] + 1.0}

    return CallableComponent(
        name,
        grid,
        step,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            initial_fields={"value": 0.0},
            execution=cast(Any, execution),
            output=OutputSpec() if output is None else output,
        ),
    )


class StaticWorkflow:
    """Return a fixed plan while retaining received planning contexts."""

    def __init__(self, steps: tuple[object, ...]) -> None:
        self.steps = steps
        self.contexts: list[runtime.WorkflowContext] = []

    def build(self, context: runtime.WorkflowContext) -> runtime.ExecutionPlan:
        """Return the configured entries as an execution plan."""
        self.contexts.append(context)
        return runtime.ExecutionPlan(steps=cast(Any, self.steps))


class SequentialBackend:
    """Execute each core-owned chunk plan sequentially through its driver."""

    def __init__(self) -> None:
        self.calls: list[tuple[runtime.ExecutionContext, runtime.ExecutionChunk]] = []

    def execute(
        self,
        state: RunState,
        *,
        context: runtime.ExecutionContext,
        chunk: runtime.ExecutionChunk,
        driver: runtime.RuntimeDriver,
    ) -> RunState:
        """Record the chunk and dispatch all of its plans in order."""
        self.calls.append((context, chunk))
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return state
