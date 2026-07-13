"""Final runtime-view NetCDF output helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vercor.calendar import ModelDateTime
from vercor.exchanges import Exchange
from vercor._runtime.exchange_keys import exchange_regrid_key
from vercor.output import SnapshotContext
from vercor.components.contracts import ComponentInfo
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output import OutputVariable
from vercor.state import ComponentState, RunState
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.jax_logging import LoggerLike


def output_masks_for_component(
    name: str,
    exchanges: Sequence[Exchange],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
) -> dict[str, RuntimeArray]:
    """Return output mask fields for one destination component."""

    masks = {}
    for exchange in exchanges:
        if name != exchange.target:
            continue

        key = (exchange.source, name, exchange_regrid_key(exchange))
        source_destination_name = "_".join(key)
        masks["bmask_" + source_destination_name] = binary_masks[key]
        masks["fmask_" + source_destination_name] = fractional_masks[key]
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
    for scope, name, value in view.iter_fields("received", "sent"):
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
    return OutputVariable(
        ("nlat", "nlon"),
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
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    output_file_mask: Path | None = None,
    output_dir: Path = Path("."),
    filename_template: str = "{component}.runtime_fields.nc",
    logger: "LoggerLike | None" = None,
) -> None:
    """Write final runtime component views for all configured components."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, component in components.items():
        if output_file_mask is None:
            filepath = output_dir / filename_template.format(
                component=name.lower(),
                component_name=name,
                name=name,
            )
        else:
            filepath = output_dir / Path(f"{name.lower()}_{output_file_mask}.nc")
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
        if logger is not None:
            logger.info(f"Finalized {name}")


def write_coupler_component_snapshots(
    *,
    final_state: RunState,
    components: Mapping[str, "Component"],
    output_time: datetime | ModelDateTime,
    output_dir: Path = Path("."),
    logger: "LoggerLike | None" = None,
) -> None:
    """Write registered native component snapshots for configured components."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, component in components.items():
        writer = component.spec.output.snapshot_writer
        if writer is None:
            continue
        runtime_state = final_state._component_state(name)
        writer(
            SnapshotContext(
                component=ComponentInfo(
                    name=component.name,
                    grid=component.grid,
                    spec=component.spec,
                ),
                state=ComponentState._from_runtime(name, component.grid, runtime_state),
                payload=runtime_state.payload,
                output_path=output_dir / f"{name.lower()}.snapshot.nc",
                time=output_time,
                logger=logger,
            )
        )


__all__ = [
    "output_masks_for_component",
    "write_coupler_component_snapshots",
    "write_coupler_runtime_outputs",
    "write_runtime_component_view_to_netcdf",
]
