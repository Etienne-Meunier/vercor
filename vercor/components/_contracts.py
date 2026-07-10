from __future__ import annotations

from collections.abc import Iterable

from vercor.components.contracts import (
    ComponentStepReturn,
    ComponentSpec,
    StepResult,
    _AuthorFieldValues,
    _AuthorStepCallable,
    _ComponentStepCallable,
    _FieldNames,
)
from vercor._field_names import unique_field_names
from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.field_layout import validate_component_data_layout
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


def normalize_author_field_values(
    *,
    component_name: str,
    grid: RectilinearGrid,
    fields: _AuthorFieldValues,
    policy: PrecisionPolicy = None,
) -> dict[str, RuntimeArray] | None:
    """Return author-provided fields as canonical runtime arrays.

    The additive authoring facade accepts scalar defaults for common setup cases.
    Scalars are expanded to grid-shaped constants; array-like values are converted
    to JAX arrays and then validated against VerCOR's canonical component-data
    layouts.
    """

    if fields is None:
        return None

    normalized: dict[str, RuntimeArray] = {}
    for field_name, field_value in fields.items():
        field_array = as_jax_real_array(field_value, policy)
        if field_array.shape == ():
            normalized[field_name] = jax_full(grid.shape, field_value, policy)
        else:
            normalized[field_name] = field_array

    validate_component_data_layout(
        component_name=component_name,
        grid_shape=grid.shape,
        data=normalized,
    )
    return normalized


def declared_runtime_field_names(spec: ComponentSpec) -> tuple[str, ...]:
    """Return all fields that a declaration validates at runtime."""

    return unique_field_names(
        (
            *spec.inputs,
            *spec.outputs,
            *tuple(spec.defaults),
        )
    )


def merge_component_outputs(
    spec: ComponentSpec,
    output_names: Iterable[str],
) -> ComponentSpec:
    """Return ``spec`` with additional output names merged in."""

    return ComponentSpec(
        inputs=spec.inputs,
        outputs=unique_field_names((*spec.outputs, *tuple(output_names))),
        defaults=spec.defaults,
        execution=spec.execution,
        lifecycle=spec.lifecycle,
        output=spec.output,
    )


__all__ = [
    "ComponentSpec",
    "StepResult",
    "_AuthorFieldValues",
    "_AuthorStepCallable",
    "_ComponentStepCallable",
    "ComponentStepReturn",
    "_FieldNames",
    "declared_runtime_field_names",
    "merge_component_outputs",
    "normalize_author_field_values",
    "unique_field_names",
]
