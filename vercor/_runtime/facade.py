from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import vercor.output._runtime as _runtime_output
from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._run_order import normalize_run_order
from vercor._runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor._runtime.initialization import (
    RuntimeInitializationState,
    initialize_coupler_runtime as _initialize_coupler_runtime,
)
from vercor._runtime.preparation import (
    create_runtime_state,
    prepare_runtime_state,
)
from vercor._runtime.resources import CouplerRuntimeResources
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.runner import (
    run_coupler_runtime,
)
from vercor.state import RunState
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass(frozen=True)
class RuntimeInputs:
    """Repeated static coupler inputs consumed by runtime facade helpers."""

    components: Mapping[str, "Component"]
    exchanges: Sequence[Exchange]
    runtime_resources: CouplerRuntimeResources
    run_order: Sequence[str]
    clock: Clock
    settings: Settings


def create_runtime_resources() -> CouplerRuntimeResources:
    """Return a fresh mutable runtime resource holder for a coupler."""

    return CouplerRuntimeResources()


def initialize_coupler_runtime(
    *,
    inputs: RuntimeInputs,
    logger: LoggerLike,
) -> RuntimeInitializationState:
    """Initialize components, runtime contracts, and exchange topology."""

    initialized = _initialize_coupler_runtime(
        clock=inputs.clock,
        components=dict(inputs.components),
        exchanges=inputs.exchanges,
        topology_maps=inputs.runtime_resources.topology_maps,
        run_order=normalize_run_order(inputs.run_order),
        settings=inputs.settings,
        logger=logger,
    )
    inputs.runtime_resources.runtime_contracts = initialized.runtime_contracts
    inputs.runtime_resources.topology_maps = initialized.topology.topology_maps
    return initialized


def runtime_dispatch_context(
    *,
    inputs: RuntimeInputs,
) -> RuntimeDispatchContext:
    """Return static runtime dispatch plumbing for a configured coupler."""

    return build_runtime_dispatch_context(
        inputs.components,
        inputs.exchanges,
        inputs.runtime_resources.topology_maps.regridders,
        inputs.runtime_resources.runtime_contracts,
        dt_seconds=inputs.clock.dt_seconds,
        settings=inputs.settings,
    )


def runtime_run_context(
    *,
    inputs: RuntimeInputs,
    logger: LoggerLike,
) -> RuntimeRunContext:
    """Return static runtime inputs bundled for execution."""

    return RuntimeRunContext(
        run_order=tuple(normalize_run_order(inputs.run_order)),
        clock=inputs.clock,
        logger=logger,
        dispatch_context=runtime_dispatch_context(
            inputs=inputs,
        ),
        interrupts=inputs.runtime_resources.interrupt_controller,
    )


def run(
    runtime_state: RunState,
    *,
    inputs: RuntimeInputs,
    logger: LoggerLike,
) -> RunState:
    """Run a validated runtime state through the selected runtime path."""

    return run_coupler_runtime(
        runtime_state,
        context=runtime_run_context(
            inputs=inputs,
            logger=logger,
        ),
    )


def finalize(
    *,
    final_state: RunState,
    inputs: RuntimeInputs,
    output_file_mask: Path | None = None,
    output_dir: Path = Path("."),
    filename_template: str = "{component}.runtime_fields.nc",
    write_snapshots: bool = True,
    logger: LoggerLike,
) -> None:
    """Write final runtime output files."""

    topology_maps = inputs.runtime_resources.topology_maps
    _runtime_output.write_coupler_runtime_outputs(
        final_state=final_state,
        components=inputs.components,
        exchanges=inputs.exchanges,
        binary_masks=topology_maps.binary_masks,
        fractional_masks=topology_maps.fractional_masks,
        output_file_mask=output_file_mask,
        output_dir=output_dir,
        filename_template=filename_template,
        logger=logger,
    )
    if write_snapshots:
        _runtime_output.write_coupler_component_snapshots(
            final_state=final_state,
            components=inputs.components,
            output_time=_final_snapshot_time(inputs.clock),
            output_dir=output_dir,
            logger=logger,
        )


def _final_snapshot_time(clock: Clock) -> datetime | ModelDateTime:
    """Return the model time represented by the final runtime state."""

    final_time: datetime | ModelDateTime = clock.start
    for _, time, _ in clock.iter():
        final_time = time
    return final_time


__all__ = [
    "RuntimeInputs",
    "create_runtime_resources",
    "create_runtime_state",
    "finalize",
    "initialize_coupler_runtime",
    "run",
    "prepare_runtime_state",
]
