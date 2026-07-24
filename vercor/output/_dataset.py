"""Shared dataset assembly helpers for NetCDF output adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.output._period import TIME_NAME, output_time_value_and_attrs
from vercor.output import OutputVariable


def grid_field_dims(
    name: str,
    shape: tuple[int, ...],
    grid_shape: tuple[int, int] | None,
) -> tuple[str, ...]:
    """Return stable generic dimensions for one optional grid-shaped field."""

    if grid_shape is not None and len(shape) >= 2 and shape[-2:] == grid_shape:
        prefix = tuple(f"{name}_dim_{index}" for index in range(len(shape) - 2))
        return (*prefix, "nlat", "nlon")
    return tuple(f"{name}_dim_{index}" for index in range(len(shape)))


def time_coordinate_variable(
    output_time: datetime | ModelDateTime,
    *,
    time_dim: str = TIME_NAME,
) -> OutputVariable:
    """Return the one-step NetCDF time coordinate for an output dataset."""

    values, attrs = output_time_value_and_attrs(output_time)
    return OutputVariable((time_dim,), values, attrs)


def used_dimension_names(
    variables: Mapping[str, OutputVariable],
    *,
    excluded_dims: Sequence[str] = (TIME_NAME,),
) -> tuple[str, ...]:
    """Return non-excluded dimensions in first-use order across variables."""

    excluded = set(excluded_dims)
    used: list[str] = []
    for variable in variables.values():
        for dim in variable.dims:
            if dim not in excluded and dim not in used:
                used.append(dim)
    return tuple(used)


__all__ = ["grid_field_dims", "time_coordinate_variable", "used_dimension_names"]
