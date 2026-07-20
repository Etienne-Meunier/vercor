"""Final runtime-view NetCDF output helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
import re
from typing import TYPE_CHECKING

from vercor.calendar import ModelDateTime
from vercor.exceptions import ComponentError
from vercor.exchanges import Exchange
from vercor.output import SnapshotContext
from vercor.output._dataset import grid_field_dims
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output import OutputVariable
from vercor.state import ComponentState, RunState
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding
    from vercor.jax_logging import LoggerLike


def output_masks_for_component(
    name: str,
    exchanges: Sequence[Exchange],
    binary_masks: Mapping[str, RuntimeArray],
    fractional_masks: Mapping[str, RuntimeArray],
) -> dict[str, RuntimeArray]:
    """Return output mask fields for one destination component."""

    masks: dict[str, RuntimeArray] = {}
    destination_exchanges = tuple(
        exchange for exchange in exchanges if name == exchange.target
    )
    base_tokens = {
        exchange.route_id: (
            re.sub(r"[^A-Za-z0-9_]+", "_", exchange.route_id).strip("_") or "route"
        )
        for exchange in destination_exchanges
    }
    token_counts = Counter(base_tokens.values())
    for exchange in destination_exchanges:

        route_id = exchange.route_id
        route_token = base_tokens[route_id]
        if token_counts[route_token] > 1:
            route_token = f"{route_token}_{route_id.encode('utf-8').hex()}"
        masks["bmask_" + route_token] = binary_masks[route_id]
        masks["fmask_" + route_token] = fractional_masks[route_id]
    return masks


def write_runtime_component_view_to_netcdf(
    view: ComponentState,
    filename: Path,
    *,
    masks: dict[str, RuntimeArray] | None = None,
) -> None:
    """Write final runtime fields from a single runtime component view.

    Arguments:
        view: runtime component view containing fields to write
        filename: path to the output NetCDF file
        masks: optional mask fields to include in the same output
    """

    write_netcdf_dataset(
        output=str(filename),
        coordinate_variables=_runtime_coordinate_variables(view),
        data_variables=_runtime_data_variables(view, masks=masks or {}),
    )


def _runtime_coordinate_variables(
    view: ComponentState,
) -> dict[str, OutputVariable]:
    if view.grid is None:
        raise ValueError("ComponentState grid is required for runtime output.")
    grid = view.grid
    return {
        "latitude": OutputVariable(("nlat",), grid.latitude),
        "longitude": OutputVariable(("nlon",), grid.longitude),
    }


def _runtime_data_variables(
    view: ComponentState,
    *,
    masks: Mapping[str, RuntimeArray],
) -> dict[str, OutputVariable]:
    data_variables: dict[str, OutputVariable] = {}
    for scope, name, value in view.iter_fields("state", "received", "sent"):
        data_variables[f"{scope}_{name}"] = _runtime_output_variable(
            view,
            value,
            runtime_store=scope,
            field_name=name,
        )

    for name, value in masks.items():
        data_variables[name] = _runtime_output_variable(
            view,
            value,
            runtime_store="mask",
            field_name=name,
        )
    return data_variables


def _runtime_output_variable(
    view: ComponentState,
    value: RuntimeArray,
    *,
    runtime_store: str,
    field_name: str,
) -> OutputVariable:
    shape = tuple(value.shape)
    return OutputVariable(
        grid_field_dims(
            field_name,
            shape,
            view.grid.shape if view.grid is not None else None,
        ),
        value,
        {
            "component": view.name,
            "runtime_store": runtime_store,
            "field_name": field_name,
        },
    )


def write_coupler_runtime_outputs(
    *,
    final_state: RunState,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    binary_masks: Mapping[str, RuntimeArray],
    fractional_masks: Mapping[str, RuntimeArray],
    output_dir: Path = Path("."),
    logger: "LoggerLike | None" = None,
) -> None:
    """Write final runtime component views for all configured components."""

    filenames = _component_output_filenames(
        tuple(components),
        suffix="runtime_fields.nc",
    )
    for name, component in components.items():
        filepath = output_dir / filenames[name]
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            view = ComponentState._from_runtime(
                name,
                component.grid,
                final_state._component_state(name),
            )
            write_runtime_component_view_to_netcdf(
                view,
                filepath,
                masks=output_masks_for_component(
                    name,
                    exchanges,
                    binary_masks,
                    fractional_masks,
                ),
            )
        except Exception as exc:
            raise ComponentError(
                "Final-field output for component "
                f"{name!r} at {str(filepath)!r}: {exc}"
            ) from exc
        if logger is not None:
            logger.info(f"Finalized {name}")


def write_coupler_component_snapshots(
    *,
    final_state: RunState,
    components: Mapping[str, "_ComponentBinding"],
    output_time: datetime | ModelDateTime,
    output_dir: Path = Path("."),
    logger: "LoggerLike | None" = None,
) -> None:
    """Write registered native component snapshots for configured components."""

    filenames = _component_output_filenames(
        tuple(components),
        suffix="snapshot.nc",
    )
    for name, component in components.items():
        writer = component.spec.output.snapshot_writer
        if writer is None:
            continue
        output_path = output_dir / filenames[name]
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            runtime_state = final_state._component_state(name)
            writer(
                SnapshotContext(
                    component=component._component,
                    state=ComponentState._from_runtime(
                        name,
                        component.grid,
                        runtime_state,
                    ),
                    payload=runtime_state.payload,
                    output_path=output_path,
                    time=output_time,
                    logger=logger,
                )
            )
        except Exception as exc:
            raise ComponentError(
                "Snapshot output for component "
                f"{name!r} at {str(output_path)!r}: {exc}"
            ) from exc


def _component_output_filenames(
    names: tuple[str, ...],
    *,
    suffix: str,
) -> dict[str, str]:
    """Return deterministic path-safe unique filenames for components."""

    base_tokens = {
        name: (re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-_").lower() or "component")
        for name in names
    }
    counts = Counter(base_tokens.values())
    return {
        name: (
            f"{token}.{suffix}"
            if counts[token] == 1
            else f"{token}.component{index:04d}.{suffix}"
        )
        for index, (name, token) in enumerate(base_tokens.items())
    }


__all__ = [
    "output_masks_for_component",
    "write_coupler_component_snapshots",
    "write_coupler_runtime_outputs",
    "write_runtime_component_view_to_netcdf",
]
