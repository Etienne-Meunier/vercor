"""Private component field normalization and declaration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor._field_names import unique_field_names
from vercor.components.contracts import ComponentSpec
from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.exceptions import ComponentError
from vercor.field_layout import validate_component_data_layout
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


def normalize_field_values(
    *,
    component_name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object] | None,
    policy: PrecisionPolicy = None,
) -> dict[str, RuntimeArray]:
    """Expand scalar author values and validate canonical component layouts."""

    normalized: dict[str, RuntimeArray] = {}
    for field_name, field_value in (fields or {}).items():
        try:
            field_array = as_jax_real_array(field_value, policy)
        except (TypeError, ValueError) as exc:
            raise ComponentError(
                f"Component '{component_name}' data field '{field_name}' must be "
                "a real numeric scalar or array."
            ) from exc
        normalized[field_name] = (
            jax_full(grid.shape, field_value, policy)
            if field_array.shape == ()
            else field_array
        )
    validate_component_data_layout(
        component_name=component_name,
        grid_shape=grid.shape,
        data=normalized,
    )
    return normalized


def declared_runtime_field_names(spec: ComponentSpec) -> tuple[str, ...]:
    """Return all input/output names declared by a component spec."""

    return unique_field_names((*spec.inputs, *spec.outputs))


def validate_declared_updates(
    component_name: str,
    updates: Mapping[str, Any],
    declared: tuple[str, ...],
    *,
    phase: str,
) -> None:
    """Reject the first field update absent from the allowed declaration."""

    allowed = set(declared)
    undeclared = next((name for name in updates if name not in allowed), None)
    if undeclared is not None:
        suffix = " declared output" if phase == "step" else " declared field"
        raise ComponentError(
            f"Component '{component_name}' {phase} returned field '{undeclared}' "
            f"that is not a{suffix}."
        )


__all__: list[str] = []
