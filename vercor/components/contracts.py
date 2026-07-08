from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

from vercor.components.contexts import (
    SetupContext,
    StepContext,
)
from vercor._field_names import unique_field_names as _unique_field_names
from vercor.output import OutputConfig
from vercor.types import RuntimeArray

KEEP_PAYLOAD: Final = object()
"""Sentinel meaning a component step should preserve the existing payload."""


@dataclass(frozen=True)
class StepResult:
    """Result returned by callable component wrappers.

    Attributes:
        fields: Runtime data fields to update.
        payload: Replacement runtime payload, or ``KEEP_PAYLOAD`` to preserve
            the existing payload. Pass ``None`` explicitly to clear the payload.
    """

    fields: Mapping[str, RuntimeArray] = field(default_factory=dict)
    payload: Any = KEEP_PAYLOAD


@dataclass(frozen=True)
class PrefillContext:
    """Read-only public context supplied to component runtime-prefill hooks."""

    fields: Mapping[str, RuntimeArray]
    received: Mapping[str, RuntimeArray]
    sent: Mapping[str, RuntimeArray]
    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrefillResult:
    """Field updates returned by component runtime-prefill hooks."""

    fields: Mapping[str, RuntimeArray] = field(default_factory=dict)
    received: Mapping[str, RuntimeArray] = field(default_factory=dict)
    sent: Mapping[str, RuntimeArray] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationContext:
    """Public context supplied to component runtime-state validation hooks."""

    state: Any
    payload: Any | None = None
    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()


_ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | StepResult
_ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], StepContext, Any | None],
    _ComponentStepReturn,
]
_AuthorStepCallable: TypeAlias = Callable[..., _ComponentStepReturn]
_FieldNames: TypeAlias = Iterable[str]
_AuthorFieldValues: TypeAlias = Mapping[str, object] | None
ComponentInitializeHook = Callable[[Any, SetupContext], None]
ComponentCreatePayloadHook = Callable[[Any], Any | None]
ComponentPrefillHook = Callable[[Any, PrefillContext], PrefillResult | None]
ComponentValidateHook = Callable[[Any, ValidationContext], None]


@dataclass(frozen=True)
class LifecycleHooks:
    """Optional lifecycle hooks for component setup and runtime customization."""

    initialize: ComponentInitializeHook | None = None
    create_payload: ComponentCreatePayloadHook | None = None
    prefill: ComponentPrefillHook | None = None
    validate: ComponentValidateHook | None = None


@dataclass(frozen=True, init=False)
class ComponentSpec:
    """Author-facing declaration of a component's runtime data-field contract.

    Attributes:
        inputs: Fields the model expects to read from runtime data.
        outputs: Fields the model may write. These are pre-seeded as grid-shaped
            zeros before traced runtime execution.
        defaults: Field defaults used when runtime state is created.
    """

    inputs: _FieldNames = ()
    outputs: _FieldNames = ()
    defaults: Mapping[str, object] = field(default_factory=dict)
    lifecycle: LifecycleHooks = field(default_factory=LifecycleHooks)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __init__(
        self,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        defaults: Mapping[str, object] | None = None,
        *,
        lifecycle: LifecycleHooks | None = None,
        output: OutputConfig | None = None,
    ) -> None:
        """Create a field declaration."""

        object.__setattr__(self, "inputs", _unique_field_names(inputs))
        object.__setattr__(self, "outputs", _unique_field_names(outputs))
        object.__setattr__(self, "defaults", dict(defaults or {}))
        object.__setattr__(
            self,
            "lifecycle",
            LifecycleHooks() if lifecycle is None else lifecycle,
        )
        object.__setattr__(
            self,
            "output",
            OutputConfig() if output is None else output,
        )


__all__ = [
    "ComponentCreatePayloadHook",
    "LifecycleHooks",
    "ComponentInitializeHook",
    "ComponentPrefillHook",
    "ComponentValidateHook",
    "ComponentSpec",
    "KEEP_PAYLOAD",
    "PrefillContext",
    "PrefillResult",
    "SetupContext",
    "StepContext",
    "StepResult",
    "ValidationContext",
]
