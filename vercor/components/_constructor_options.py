from __future__ import annotations

from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentHooks,
    FieldNames,
    FieldSpec,
)


def normalize_field_spec(
    *,
    inputs: FieldNames = (),
    outputs: FieldNames = (),
    defaults: AuthorFieldValues = None,
) -> FieldSpec:
    """Normalize public field declaration options to one ``FieldSpec``."""

    return FieldSpec(
        inputs=inputs,
        outputs=outputs,
        defaults=defaults or {},
    )


def normalize_lifecycle_hooks(
    *,
    hooks: ComponentHooks | None,
) -> ComponentLifecycleHooks:
    """Normalize public lifecycle hook options to one private hook container."""

    return ComponentLifecycleHooks(
        initialize=hooks.initialize if hooks is not None else None,
        create_runtime_payload=hooks.create_payload if hooks is not None else None,
        prefill_runtime_state_fields=hooks.prefill if hooks is not None else None,
        validate_runtime_state=hooks.validate if hooks is not None else None,
    )


__all__ = ["normalize_field_spec", "normalize_lifecycle_hooks"]
