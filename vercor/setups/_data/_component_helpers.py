from __future__ import annotations

from collections.abc import Mapping

from vercor.components import (
    ComponentSpec,
    FieldImportPolicy,
    LifecycleHooks,
    DataComponent,
    SetupContext,
)
from vercor.grids import RectilinearGrid


def time_interpolated_data_component(
    *,
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object],
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...],
    defaults: Mapping[str, object] | None = None,
    data_files: Mapping[str, str],
    lifecycle: LifecycleHooks | None = None,
) -> DataComponent:
    """Create a data component with the standard time-interpolation metadata."""

    def initialize(component: DataComponent, context: SetupContext) -> None:
        if lifecycle is not None and lifecycle.initialize is not None:
            lifecycle.initialize(component, context)

    source_lifecycle = LifecycleHooks() if lifecycle is None else lifecycle
    runtime_lifecycle = LifecycleHooks(
        initialize=initialize,
        create_payload=source_lifecycle.create_payload,
        prefill=source_lifecycle.prefill,
        validate=source_lifecycle.validate,
    )
    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields=fields,
        spec=ComponentSpec(
            inputs=inputs,
            outputs=outputs,
            defaults=defaults,
            lifecycle=runtime_lifecycle,
            import_policy=FieldImportPolicy(time_interpolation=True),
        ),
    )
    component._setup_metadata["DATA_FILES"] = dict(data_files)
    return component
