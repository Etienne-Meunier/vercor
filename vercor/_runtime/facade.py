from __future__ import annotations

from datetime import datetime
from pathlib import Path

import vercor.output._runtime as _runtime_output
from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.jax_logging import LoggerLike
from vercor._runtime.preparation import (
    create_runtime_state,
    prepare_runtime_state,
)
from vercor._runtime.prepared import PreparedCoupling, prepare_coupling
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.runner import (
    run_coupler_runtime,
)
from vercor.state import RunState


def runtime_run_context(
    *,
    prepared: PreparedCoupling,
    logger: LoggerLike,
) -> RuntimeRunContext:
    """Return static runtime inputs bundled for execution."""

    return RuntimeRunContext(
        run_order=prepared.run_order,
        clock=prepared.clock,
        logger=logger,
        dispatch_context=prepared.dispatch_context,
        interrupts=prepared.interrupts,
        options=prepared.runtime,
    )


def run(
    runtime_state: RunState,
    *,
    prepared: PreparedCoupling,
    logger: LoggerLike,
) -> RunState:
    """Run a validated runtime state through the selected runtime path."""

    return run_coupler_runtime(
        runtime_state,
        context=runtime_run_context(
            prepared=prepared,
            logger=logger,
        ),
    )


def finalize(
    *,
    final_state: RunState,
    prepared: PreparedCoupling,
    output_file_mask: Path | None = None,
    output_dir: Path = Path("."),
    filename_template: str = "{component}.runtime_fields.nc",
    write_snapshots: bool = True,
    logger: LoggerLike,
) -> None:
    """Write final runtime output files."""

    topology_maps = prepared.topology_maps
    _runtime_output.write_coupler_runtime_outputs(
        final_state=final_state,
        components=prepared.components,
        exchanges=prepared.exchanges,
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
            components=prepared.components,
            output_time=_final_snapshot_time(prepared.clock),
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
    "create_runtime_state",
    "finalize",
    "prepare_coupling",
    "run",
    "prepare_runtime_state",
]
