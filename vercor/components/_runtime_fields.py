"""Private application of public step results to immutable runtime stores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.components._contracts import validate_declared_updates
from vercor.components.contracts import StepResult, _ComponentStepReturn, _KEEP_PAYLOAD
from vercor.exceptions import ComponentError
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contracts import Component
    from vercor._runtime.state import ComponentRuntimeState


def runtime_fields(
    component_state: "ComponentRuntimeState",
) -> dict[str, RuntimeArray]:
    """Return runtime fields as an ordinary mapping for author callbacks."""

    return component_state.fields.to_mapping()


def _with_runtime_fields(
    component: "Component",
    component_state: "ComponentRuntimeState",
    fields: Mapping[str, RuntimeArray],
) -> "ComponentRuntimeState":
    """Return runtime state with declared existing output fields replaced."""

    validate_declared_updates(
        component.name,
        fields,
        component.spec.outputs,
        phase="step",
    )
    missing = next(
        (
            field_name
            for field_name in fields
            if field_name not in component_state.fields
        ),
        None,
    )
    if missing is not None:
        raise ComponentError(
            f"Component '{component.name}' step returned field '{missing}' that "
            "is not present in runtime state."
        )
    try:
        return component_state.with_fields(component_state.fields.replace_many(fields))
    except ValueError as exc:
        raise ComponentError(
            f"Component '{component.name}' returned an invalid step field update: {exc}"
        ) from exc


def apply_step_result(
    component: "Component",
    component_state: "ComponentRuntimeState",
    result: _ComponentStepReturn,
) -> "ComponentRuntimeState":
    """Validate and apply a mapping or ``StepResult``."""

    if isinstance(result, StepResult):
        updated = _with_runtime_fields(component, component_state, result.fields)
        return (
            updated
            if result.payload is _KEEP_PAYLOAD
            else updated.with_payload(result.payload)
        )
    if not isinstance(result, Mapping):
        raise ComponentError(
            f"Component '{component.name}' step must return a mapping or "
            f"StepResult; got {type(result).__name__}."
        )
    return _with_runtime_fields(component, component_state, result)


__all__: list[str] = []
