from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from vercor.components import (
    ComponentSpec,
    LifecycleHooks,
    DataComponent,
    TransferPolicy,
)
from vercor.grids import RectilinearGrid


def time_interpolated_data_component(
    *,
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object],
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...],
    initial_fields: Mapping[str, object] | None = None,
    data_files: Mapping[str, str],
    lifecycle: LifecycleHooks | None = None,
) -> DataComponent:
    """Create a data component with the standard time-interpolation metadata."""

    component = DataComponent(
        name,
        grid,
        fields,
        spec=ComponentSpec(
            inputs=inputs,
            outputs=outputs,
            initial_fields=initial_fields,
            lifecycle=lifecycle,
            transfer=TransferPolicy(time_selection="linear"),
        ),
    )
    cast(Any, component)._data_files = dict(data_files)
    return component
