from __future__ import annotations

from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentCreatePayloadHook,
    ComponentHooks,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    FieldNames,
    FieldSpec,
)


def normalize_field_spec(
    *,
    fields: FieldSpec | None,
    inputs: FieldNames = (),
    outputs: FieldNames = (),
    defaults: AuthorFieldValues = None,
) -> FieldSpec:
    """Normalize public field declaration options to one ``FieldSpec``."""

    if fields is not None and (tuple(inputs) or tuple(outputs) or defaults is not None):
        raise TypeError(
            "Use either fields=FieldSpec(...) or inputs/outputs/defaults, not both"
        )

    return fields or FieldSpec(
        inputs=inputs,
        outputs=outputs,
        defaults=defaults or {},
    )


def normalize_lifecycle_hooks(
    *,
    hooks: ComponentHooks | None,
    initialize: ComponentInitializeHook | None,
    create_runtime_payload: ComponentCreatePayloadHook | None,
    prefill_runtime_state_fields: ComponentPrefillHook | None,
    validate_runtime_state: ComponentValidateHook | None,
) -> ComponentLifecycleHooks:
    """Normalize public lifecycle hook options to one private hook container."""

    if hooks is not None and any(
        hook is not None
        for hook in (
            initialize,
            create_runtime_payload,
            prefill_runtime_state_fields,
            validate_runtime_state,
        )
    ):
        raise TypeError(
            "Use either hooks=ComponentHooks(...) or individual hook arguments, not both"
        )

    return ComponentLifecycleHooks(
        initialize=hooks.initialize if hooks is not None else initialize,
        create_runtime_payload=(
            hooks.create_payload if hooks is not None else create_runtime_payload
        ),
        prefill_runtime_state_fields=(
            hooks.prefill if hooks is not None else prefill_runtime_state_fields
        ),
        validate_runtime_state=(
            hooks.validate if hooks is not None else validate_runtime_state
        ),
    )


__all__ = ["normalize_field_spec", "normalize_lifecycle_hooks"]
