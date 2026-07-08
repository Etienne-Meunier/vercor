from __future__ import annotations

from collections.abc import Mapping

from vercor.components import ComponentSpec, LifecycleHooks, DataComponent
from vercor.grids import RectilinearGrid


def time_interpolated_data_component(
    *,
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object],
    outputs: tuple[str, ...],
    data_files: Mapping[str, str],
    hooks: LifecycleHooks | None = None,
) -> DataComponent:
    """Create a data component with the standard time-interpolation metadata."""

    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields=fields,
        spec=ComponentSpec(outputs=outputs, hooks=hooks),
    )
    component.update_settings(apply_time_interpolation=True)
    component._setup_metadata["DATA_FILES"] = dict(data_files)
    return component
