from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias

from vercor.components.contexts import (
    SetupContext,
    StepContext,
)
from vercor._field_names import unique_field_names as _unique_field_names
from vercor.output import OutputConfig
from vercor.state import ComponentState
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.grids import RectilinearGrid

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

    state: ComponentState
    payload: Any | None = None
    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()


ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | StepResult
_ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], StepContext, Any | None],
    ComponentStepReturn,
]
_AuthorStepCallable: TypeAlias = Callable[..., ComponentStepReturn]
_FieldNames: TypeAlias = Iterable[str]
_AuthorFieldValues: TypeAlias = Mapping[str, object] | None
ComponentInitializeHook = Callable[[Any, SetupContext], None]
ComponentCreatePayloadHook = Callable[[Any], Any | None]
ComponentPrefillHook = Callable[[Any, PrefillContext], PrefillResult | None]
ComponentValidateHook = Callable[[Any, ValidationContext], None]


class ComponentLike(Protocol):
    """Public structural contract for user-provided model components."""

    @property
    def name(self) -> str:
        """Return the component name used in exchanges and run order."""
        ...

    @property
    def grid(self) -> "RectilinearGrid":
        """Return the component grid."""
        ...

    @property
    def spec(self) -> "ComponentSpec":
        """Return the component runtime field contract."""
        ...

    def initial_fields(self) -> Mapping[str, RuntimeArray]:
        """Return setup-time fields used to seed runtime state."""
        ...

    def initialize(self, context: SetupContext) -> None:
        """Run setup-time initialization before runtime state is created."""
        ...

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> ComponentStepReturn:
        """Return runtime field updates for one component step."""
        ...


@dataclass(frozen=True, eq=False)
class ComponentInfo:
    """Public immutable description of a registered component."""

    name: str
    grid: "RectilinearGrid"
    spec: "ComponentSpec"

    def __eq__(self, other: object) -> bool:
        """Compare component metadata without array-valued grid equality."""

        if not isinstance(other, ComponentInfo):
            return NotImplemented
        return (
            self.name == other.name
            and self.grid.name == other.grid.name
            and self.grid.shape == other.grid.shape
            and self.spec == other.spec
        )


@dataclass(frozen=True)
class LifecycleHooks:
    """Optional lifecycle hooks for component setup and runtime customization."""

    initialize: ComponentInitializeHook | None = None
    create_payload: ComponentCreatePayloadHook | None = None
    prefill: ComponentPrefillHook | None = None
    validate: ComponentValidateHook | None = None


@dataclass(frozen=True)
class FieldImportPolicy:
    """Policy for selecting time-dependent fields when a component sends data."""

    time_interpolation: bool = False
    daily_selection: bool = False

    def __post_init__(self) -> None:
        """Validate mutually exclusive import selection policies."""

        if self.time_interpolation and self.daily_selection:
            raise ValueError(
                "time_interpolation and daily_selection cannot both be enabled"
            )


@dataclass(frozen=True, init=False)
class ComponentSpec:
    """Author-facing declaration of a component's runtime data-field contract.

    Attributes:
        inputs: Fields the model expects to read from runtime data.
        outputs: Fields the model may write. These are pre-seeded as grid-shaped
            zeros before traced runtime execution.
        defaults: Field defaults used when runtime state is created.
        execution: Runtime path for this component, either differentiable JAX
            execution or Python host execution.
    """

    inputs: Iterable[str] = ()
    outputs: Iterable[str] = ()
    defaults: Mapping[str, object] = field(default_factory=dict)
    execution: Literal["jax", "host"] = "jax"
    lifecycle: LifecycleHooks = field(default_factory=LifecycleHooks)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __init__(
        self,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        defaults: Mapping[str, object] | None = None,
        *,
        execution: Literal["jax", "host"] = "jax",
        lifecycle: LifecycleHooks | None = None,
        output: OutputConfig | None = None,
    ) -> None:
        """Create a field declaration."""

        if execution not in ("jax", "host"):
            raise ValueError("execution must be 'jax' or 'host'")
        object.__setattr__(self, "inputs", _unique_field_names(inputs))
        object.__setattr__(self, "outputs", _unique_field_names(outputs))
        object.__setattr__(
            self,
            "defaults",
            MappingProxyType(dict(defaults or {})),
        )
        object.__setattr__(self, "execution", execution)
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
    "ComponentLike",
    "ComponentInfo",
    "ComponentCreatePayloadHook",
    "FieldImportPolicy",
    "LifecycleHooks",
    "ComponentInitializeHook",
    "ComponentPrefillHook",
    "ComponentStepReturn",
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
