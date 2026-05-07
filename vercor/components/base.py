from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast, final

import jax.numpy as jnp

from vercor.dtypes import PrecisionPolicy, jax_full, jax_zeros
from vercor.exceptions import ComponentError, CouplerError
from vercor.field_layout import validate_component_data_layout
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import (
        RuntimeComponentContract,
        RuntimeComponentState,
    )


@dataclass(frozen=True)
class ComponentStepResult:
    """Result returned by callable component wrappers.

    Attributes:
        fields: Runtime data fields to update.
        payload: Replacement runtime payload. Use a plain mapping return from a
            callable step when the existing payload should be preserved.
    """

    fields: Mapping[str, RuntimeArray]
    payload: Any | None = None


_ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | ComponentStepResult
_ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], RuntimeStepContext, Any | None],
    _ComponentStepReturn,
]
_FieldNames: TypeAlias = Iterable[str]
_FieldDefaults: TypeAlias = Mapping[str, RuntimeArray] | None
_AuthorFieldValues: TypeAlias = Mapping[str, object] | None


@dataclass(frozen=True)
class ComponentFieldSpec:
    """Author-facing declaration of a component's runtime data-field contract.

    Attributes:
        inputs: Fields the model expects to read from runtime data.
        outputs: Fields the model may write. These are pre-seeded as grid-shaped
            zeros before traced runtime execution.
        required_fields: Extra fields that must already exist but should not be
            pre-seeded by this declaration.
        default_fields: Field defaults used when runtime state is created.
    """

    inputs: _FieldNames = ()
    outputs: _FieldNames = ()
    required_fields: _FieldNames = ()
    default_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize field-name iterables once while preserving declaration order."""

        object.__setattr__(self, "inputs", _unique_field_names(self.inputs))
        object.__setattr__(self, "outputs", _unique_field_names(self.outputs))
        object.__setattr__(
            self,
            "required_fields",
            _unique_field_names(self.required_fields),
        )
        object.__setattr__(self, "default_fields", dict(self.default_fields))


@dataclass
class Component(ABC):
    """Active differentiable component-author contract for VerCOR model adapters.

    Component instances own mutable setup-time metadata: name, grid, seed data,
    and component-specific settings. During coupling, the coupler copies those
    seed fields into immutable runtime state containers so JAX can trace the
    integration. Active differentiable components must implement
    :meth:`step_runtime_state` while preserving its signature. Data-only forcing
    adapters should inherit :class:`DataComponent`; non-differentiable adapters
    should inherit :class:`HostRuntimeComponent`.

    Common exchange-field conventions:
        - fields use SI units
        - surface fluxes are positive downward and negative upward
        - data fields use canonical trailing horizontal dimensions:
          (nLat, nLon), (nTime, nLat, nLon), (nLev, nLat, nLon), or
          (nTime, nLev, nLat, nLon)

    Attributes:
        name: component name
        grid: component grid
        data: internal storage for component data arrays to/from which fields
            seed the runtime state during initialization
        settings: component-specific settings
    """

    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: VercorSettings = field(default_factory=VercorSettings)
    _field_spec: ComponentFieldSpec = field(
        default_factory=ComponentFieldSpec,
        init=False,
        repr=False,
    )

    @classmethod
    def wrap(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _ComponentStepCallable,
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        *,
        required_fields: _FieldNames = (),
        prefill_fields: _FieldNames = (),
        field_defaults: _FieldDefaults = None,
    ) -> "Component":
        """Create a differentiable callable-backed component.

        Example:
            ``Component.wrap("ATM", grid, step=my_step, fields={"temperature": t})``
            creates an active component whose callable receives ``(fields,
            context, payload)`` and returns runtime field updates.
        """

        _ = cls
        return make_differentiable_component(
            name=name,
            grid=grid,
            step=step,
            fields=fields,
            payload=payload,
            settings=settings,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )

    @classmethod
    def from_model(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _ComponentStepCallable,
        initial_fields: _AuthorFieldValues = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        *,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        required_fields: _FieldNames = (),
    ) -> "Component":
        """Create a differentiable component from a user model callable.

        This author-facing constructor mirrors normal Python alternate
        constructors: ``initial_fields`` seed model state, ``inputs`` declare
        fields the model reads, ``outputs`` declare fields the model writes, and
        scalar initial/default values expand to this component's grid shape.
        """

        _ = cls
        component_settings = settings if settings is not None else VercorSettings()
        field_spec = ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            required_fields=required_fields,
            default_fields=_normalize_author_field_values(
                component_name=name,
                grid=grid,
                fields=default_fields,
                policy=component_settings,
            )
            or {},
        )
        return make_differentiable_component(
            name=name,
            grid=grid,
            step=step,
            fields=_normalize_author_field_values(
                component_name=name,
                grid=grid,
                fields=initial_fields,
                policy=component_settings,
            ),
            payload=payload,
            settings=component_settings,
            required_fields=(*field_spec.inputs, *field_spec.required_fields),
            prefill_fields=field_spec.outputs,
            field_defaults=cast(_FieldDefaults, field_spec.default_fields),
        )

    def declare_fields(
        self,
        field_spec: ComponentFieldSpec | None = None,
        *,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        required_fields: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
    ) -> ComponentFieldSpec:
        """Declare runtime data fields for subclasses using author-facing names.

        The base runtime hooks use this declaration to prefill output/default
        fields and validate required fields. Subclasses with special lifecycle
        needs can still override those hooks directly.
        """

        declared = field_spec or ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            required_fields=required_fields,
            default_fields=default_fields or {},
        )
        self._field_spec = ComponentFieldSpec(
            inputs=declared.inputs,
            outputs=declared.outputs,
            required_fields=declared.required_fields,
            default_fields=_normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=declared.default_fields,
                policy=self.settings,
            )
            or {},
        )
        return self._field_spec

    def seed_field(self, name: str, value: RuntimeArray) -> "Component":
        """Seed one setup-time grid field and return this component.

        Seeded fields must follow VerCOR's canonical component-data layout so
        runtime state can be created with a stable PyTree structure.
        """

        return self.seed_fields({name: value})

    def seed_fields(self, fields: Mapping[str, RuntimeArray]) -> "Component":
        """Seed setup-time grid fields and return this component."""

        field_updates = dict(fields)
        validate_component_data_layout(
            component_name=self.name,
            grid_shape=self.grid.shape,
            data=field_updates,
        )
        self.data.update(field_updates)
        return self

    def seed_zero_field(
        self,
        name: str,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed one grid-shaped zero field and return this component."""

        return self.seed_field(
            name,
            jax_zeros(
                self.grid.shape,
                self.settings if policy is None else policy,
            ),
        )

    def seed_zero_fields(
        self,
        names: _FieldNames,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed multiple grid-shaped zero fields and return this component."""

        for name in names:
            self.seed_zero_field(name, policy)
        return self

    def seed_constant_field(
        self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed one grid-shaped constant field and return this component."""

        return self.seed_field(
            name,
            jax_full(
                self.grid.shape,
                value,
                self.settings if policy is None else policy,
            ),
        )

    def runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> dict[str, RuntimeArray]:
        """Return runtime data fields as a plain name-to-array mapping."""

        _ = self
        return dict(
            zip(
                component_state.data.field_names,
                component_state.data.values,
                strict=True,
            )
        )

    def runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> RuntimeArray:
        """Return one runtime data field with a component-oriented error."""

        try:
            return component_state.data.get(name)
        except KeyError as exc:
            raise ComponentError(
                f"Runtime data field '{name}' is missing for component '{self.name}'."
            ) from exc

    def has_runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> bool:
        """Return whether one runtime data field exists."""

        _ = self
        return name in component_state.data.field_names

    def runtime_field_or(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        default: object,
        policy: PrecisionPolicy = None,
    ) -> RuntimeArray:
        """Return one runtime field or a grid-shaped/default array fallback."""

        if self.has_runtime_field(component_state, name):
            return self.runtime_field(component_state, name)
        normalized = _normalize_author_field_values(
            component_name=self.name,
            grid=self.grid,
            fields={name: default},
            policy=self.settings if policy is None else policy,
        )
        if normalized is None:
            raise ComponentError(
                f"Default runtime field '{name}' could not be normalized for "
                f"component '{self.name}'."
            )
        return normalized[name]

    def runtime_field_or_zeros_like(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        like: str | RuntimeArray,
    ) -> RuntimeArray:
        """Return one runtime field or zeros matching another field/array."""

        if self.has_runtime_field(component_state, name):
            return self.runtime_field(component_state, name)
        reference = (
            self.runtime_field(component_state, like) if isinstance(like, str) else like
        )
        return jnp.zeros_like(jnp.asarray(reference))

    def with_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        fields: Mapping[str, RuntimeArray],
    ) -> "RuntimeComponentState":
        """Return ``component_state`` with existing runtime data fields updated."""

        data = component_state.data
        for field_name, field_value in fields.items():
            if field_name not in data.field_names:
                raise ComponentError(
                    f"Component '{self.name}' returned update for runtime data "
                    f"field '{field_name}', but it is missing from runtime data. "
                    "Seed the field with seed_field()/seed_fields(), include it in "
                    "factory fields, declare it as an output/default in "
                    "from_model()/declare_fields(), or declare it through an "
                    "exchange before runtime execution."
                )
            data = data.set(field_name, field_value)
        return component_state.with_data(data)

    def require_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        *names: str,
    ) -> None:
        """Validate that named runtime data fields exist and match this grid."""

        for field_name in names:
            if field_name not in component_state.data.field_names:
                raise CouplerError(
                    "Runtime missing required data field "
                    f"'{field_name}' for component '{self.name}'"
                )
            field_shape = jnp.asarray(component_state.data.get(field_name)).shape
            if field_shape != self.grid.shape:
                raise CouplerError(
                    "Runtime required data field "
                    f"'{field_name}' for component '{self.name}' has shape "
                    f"{field_shape}, expected {self.grid.shape}"
                )

    def prefill_runtime_fields(
        self,
        data: dict[str, RuntimeArray],
        field_spec: ComponentFieldSpec | None = None,
        *,
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> None:
        """Prefill a mutable runtime data mapping with declared fields.

        This helper is intended for ``prefill_runtime_state_fields()`` overrides.
        Default fields are inserted first, then output fields are inserted as
        grid-shaped zeros when they are still missing.
        """

        declared = field_spec or ComponentFieldSpec(
            outputs=outputs,
            default_fields=default_fields or {},
        )
        normalized_defaults = _normalize_author_field_values(
            component_name=self.name,
            grid=self.grid,
            fields=declared.default_fields,
            policy=self.settings if policy is None else policy,
        )
        for field_name, field_value in (normalized_defaults or {}).items():
            data.setdefault(field_name, field_value)

        zeros = jax_zeros(
            self.grid.shape,
            self.settings if policy is None else policy,
        )
        for field_name in _unique_field_names((*declared.outputs, *tuple(outputs))):
            data.setdefault(field_name, zeros)

    def initialize(self, context: ComponentInitContext) -> None:
        """Optionally initialize component-owned runtime data before coupling.

        Override this hook when setup depends on coupler context such as start
        time, coupling timestep, run sequence, settings, or logger.
        """

        _ = context

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state.

        Override this hook for differentiable models that need non-field PyTree
        state, for example model internals or forcing containers.
        """

        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: RuntimeComponentContract,
    ) -> None:
        """Optionally pre-seed fields required by runtime execution.

        Override this hook when a component creates fields during stepping and
        those fields must exist before the first JAX scan iteration.
        """

        self.prefill_runtime_fields(data, _component_field_spec(self))
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract,
    ) -> None:
        """Optionally validate component-specific runtime fields before execution.

        Override this hook to report missing payloads, diagnostic fields, or
        non-standard shapes before traced runtime execution begins.
        """

        _ = contract
        required_fields = _required_field_names(_component_field_spec(self))
        if required_fields:
            self.require_runtime_fields(component_state, *required_fields)

    @abstractmethod
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Return this differentiable component advanced by one runtime step."""

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Runtime fields: Configured by Coupler runtime contract\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)})"


class DataComponent(Component):
    """Base class for data-only components that intentionally do not step.

    Use this for forcing and boundary-condition adapters whose runtime behavior is
    limited to importing/exporting seeded fields through the coupler contract.
    Data components must not own active runtime stepping behavior; compute
    plotting-only diagnostics outside runtime state. Active differentiable models
    should inherit :class:`Component` and implement
    :meth:`Component.step_runtime_state` instead.
    """

    @classmethod
    def from_fields(
        cls,
        name: str,
        grid: RectilinearGrid,
        fields: _AuthorFieldValues = None,
        settings: VercorSettings | None = None,
    ) -> "DataComponent":
        """Create a data-only component from user-provided grid fields.

        Scalar field values expand to grid-shaped constants. Use
        :meth:`wrap` when preserving the older strict array-only behavior is
        required.
        """

        if settings is None:
            component = cls(name=name, grid=grid)
        else:
            component = cls(name=name, grid=grid, settings=settings)
        normalized_fields = _normalize_author_field_values(
            component_name=name,
            grid=grid,
            fields=fields,
            policy=component.settings,
        )
        if normalized_fields is not None:
            component.seed_fields(normalized_fields)
        return component

    @classmethod
    def wrap(  # type: ignore[override]
        cls,
        name: str,
        grid: RectilinearGrid,
        fields: Mapping[str, RuntimeArray] | None = None,
        settings: VercorSettings | None = None,
    ) -> "DataComponent":
        """Create a data-only component from optional seeded grid fields.

        Example:
            ``DataComponent.wrap("OBS", grid, fields={"temperature": data})``
            creates a forcing component with the shared no-op runtime step.
        """

        if settings is None:
            component = cls(name=name, grid=grid)
        else:
            component = cls(name=name, grid=grid, settings=settings)
        if fields is not None:
            component.seed_fields(fields)
        return component

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Return the runtime state unchanged for data-only components."""

        _ = context
        return component_state


class HostRuntimeComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    @classmethod
    def from_model(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _ComponentStepCallable,
        initial_fields: _AuthorFieldValues = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        *,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        required_fields: _FieldNames = (),
    ) -> "HostRuntimeComponent":
        """Create a host-runtime component from a Python model callable."""

        _ = cls
        component_settings = settings if settings is not None else VercorSettings()
        field_spec = ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            required_fields=required_fields,
            default_fields=_normalize_author_field_values(
                component_name=name,
                grid=grid,
                fields=default_fields,
                policy=component_settings,
            )
            or {},
        )
        return make_host_component(
            name=name,
            grid=grid,
            step=step,
            fields=_normalize_author_field_values(
                component_name=name,
                grid=grid,
                fields=initial_fields,
                policy=component_settings,
            ),
            payload=payload,
            settings=component_settings,
            required_fields=(*field_spec.inputs, *field_spec.required_fields),
            prefill_fields=field_spec.outputs,
            field_defaults=cast(_FieldDefaults, field_spec.default_fields),
        )

    @classmethod
    def wrap(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _ComponentStepCallable,
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        *,
        required_fields: _FieldNames = (),
        prefill_fields: _FieldNames = (),
        field_defaults: _FieldDefaults = None,
    ) -> "HostRuntimeComponent":
        """Create a host-runtime callable-backed component.

        Example:
            ``HostRuntimeComponent.wrap("ATM", grid, step=python_model_step)``
            creates a component that runs only through ``Coupler.run()``.
        """

        _ = cls
        return make_host_component(
            name=name,
            grid=grid,
            step=step,
            fields=fields,
            payload=payload,
            settings=settings,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Reject accidental execution on the differentiable scanned runtime."""

        _ = component_state, context
        component_name = getattr(self, "name", self.__class__.__name__)
        raise ComponentError(
            f"Component '{component_name}' is host-backed and cannot run through "
            "the differentiable scanned runtime. Use Coupler.run() so VerCOR can "
            "select the host runtime path, or implement a differentiable Component."
        )

    @abstractmethod
    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""


class _CallableRuntimeMixin:
    """Shared metadata hooks for callable-backed component wrappers."""

    _payload: Any | None
    _required_fields: tuple[str, ...]
    _prefill_fields: tuple[str, ...]
    _field_defaults: dict[str, RuntimeArray]

    def _initialize_callable_runtime(
        self,
        *,
        payload: Any | None,
        required_fields: _FieldNames,
        prefill_fields: _FieldNames,
        field_defaults: _FieldDefaults,
    ) -> None:
        self._payload = payload
        self._prefill_fields = _unique_field_names(prefill_fields)
        self._field_defaults = dict(field_defaults or {})
        component = cast(Component, self)
        validate_component_data_layout(
            component_name=component.name,
            grid_shape=component.grid.shape,
            data=self._field_defaults,
        )
        self._required_fields = _unique_field_names(
            (
                *tuple(required_fields),
                *self._prefill_fields,
                *tuple(self._field_defaults),
            )
        )
        component._field_spec = ComponentFieldSpec(
            outputs=self._prefill_fields,
            required_fields=required_fields,
            default_fields=self._field_defaults,
        )

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        return self._payload

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: RuntimeComponentContract,
    ) -> None:
        """Pre-seed callable runtime fields declared by wrapper metadata."""

        component = cast(Component, self)
        for field_name, field_value in self._field_defaults.items():
            data.setdefault(field_name, field_value)
        zeros = jax_zeros(component.grid.shape, component.settings)
        for field_name in self._prefill_fields:
            data.setdefault(field_name, zeros)
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract,
    ) -> None:
        """Validate callable runtime fields declared as required."""

        _ = contract
        cast(Component, self).require_runtime_fields(
            component_state,
            *self._required_fields,
        )


class _CallableComponent(_CallableRuntimeMixin, Component):
    """Differentiable component backed by a user-provided step callable."""

    _step: _ComponentStepCallable
    _payload: Any | None

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: _ComponentStepCallable,
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        required_fields: _FieldNames = (),
        prefill_fields: _FieldNames = (),
        field_defaults: _FieldDefaults = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = step
        self._initialize_callable_runtime(
            payload=payload,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )
        if fields is not None:
            self.seed_fields(fields)

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed differentiable component one step."""

        return _apply_callable_step_result(
            self,
            component_state,
            self._step(
                self.runtime_fields(component_state),
                context,
                component_state.runtime_payload,
            ),
        )


class _CallableHostRuntimeComponent(_CallableRuntimeMixin, HostRuntimeComponent):
    """Host-runtime component backed by a user-provided step callable."""

    _step: _ComponentStepCallable
    _payload: Any | None

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: _ComponentStepCallable,
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        required_fields: _FieldNames = (),
        prefill_fields: _FieldNames = (),
        field_defaults: _FieldDefaults = None,
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = step
        self._initialize_callable_runtime(
            payload=payload,
            required_fields=required_fields,
            prefill_fields=prefill_fields,
            field_defaults=field_defaults,
        )
        if fields is not None:
            self.seed_fields(fields)

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return _apply_callable_step_result(
            self,
            component_state,
            self._step(
                self.runtime_fields(component_state),
                context,
                component_state.runtime_payload,
            ),
        )


def _apply_callable_step_result(
    component: Component,
    component_state: "RuntimeComponentState",
    result: _ComponentStepReturn,
) -> "RuntimeComponentState":
    """Apply a callable wrapper result to runtime state."""

    if isinstance(result, ComponentStepResult):
        updated_state = component.with_runtime_fields(component_state, result.fields)
        return updated_state.with_runtime_payload(result.payload)

    return component.with_runtime_fields(component_state, result)


def _normalize_author_field_values(
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
        field_array = jnp.asarray(field_value)
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


def _required_field_names(field_spec: ComponentFieldSpec) -> tuple[str, ...]:
    """Return all fields that a declaration requires at runtime."""

    return _unique_field_names(
        (
            *field_spec.inputs,
            *field_spec.outputs,
            *field_spec.required_fields,
            *tuple(field_spec.default_fields),
        )
    )


def _component_field_spec(component: Component) -> ComponentFieldSpec:
    """Return the component's declared field spec, tolerating light test fixtures."""

    return getattr(component, "_field_spec", ComponentFieldSpec())


def _unique_field_names(field_names: _FieldNames) -> tuple[str, ...]:
    """Return field names without duplicates while preserving order."""

    unique: list[str] = []
    for field_name in field_names:
        if field_name not in unique:
            unique.append(field_name)
    return tuple(unique)


def make_data_component(
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, RuntimeArray] | None = None,
    settings: VercorSettings | None = None,
) -> DataComponent:
    """Create a data-only component from optional seeded grid fields."""

    return DataComponent.wrap(name=name, grid=grid, fields=fields, settings=settings)


def make_differentiable_component(
    name: str,
    grid: RectilinearGrid,
    step: _ComponentStepCallable,
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    *,
    required_fields: _FieldNames = (),
    prefill_fields: _FieldNames = (),
    field_defaults: _FieldDefaults = None,
) -> Component:
    """Create a differentiable component from a runtime step callable.

    The callable receives ``(fields, context, payload)``. Return a mapping to
    update data fields while preserving payload, or return
    :class:`ComponentStepResult` to replace both fields and payload.
    """

    return _CallableComponent(
        name=name,
        grid=grid,
        step=step,
        fields=fields,
        payload=payload,
        settings=settings,
        required_fields=required_fields,
        prefill_fields=prefill_fields,
        field_defaults=field_defaults,
    )


def make_host_component(
    name: str,
    grid: RectilinearGrid,
    step: _ComponentStepCallable,
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    *,
    required_fields: _FieldNames = (),
    prefill_fields: _FieldNames = (),
    field_defaults: _FieldDefaults = None,
) -> HostRuntimeComponent:
    """Create a host-runtime component from a Python runtime step callable."""

    return _CallableHostRuntimeComponent(
        name=name,
        grid=grid,
        step=step,
        fields=fields,
        payload=payload,
        settings=settings,
        required_fields=required_fields,
        prefill_fields=prefill_fields,
        field_defaults=field_defaults,
    )


def validate_component_setup(component: Component) -> None:
    """Raise a clear error when a component skipped base initialization."""

    required_attributes = ("name", "grid", "data", "settings")
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(component, attribute)
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ComponentError(
            f"Component '{component.__class__.__name__}' is missing required setup "
            f"attribute(s): {missing_names}. Call super().__init__(name, grid=...) "
            "from the component constructor before runtime initialization, "
            "execution, or finalization."
        )

    if not isinstance(component.grid, RectilinearGrid):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'grid'; "
            "expected RectilinearGrid."
        )
    if not isinstance(component.data, dict):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'data'; "
            "expected dict[str, RuntimeArray]."
        )
    if not isinstance(component.settings, VercorSettings):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'settings'; "
            "expected VercorSettings."
        )
    validate_component_data_layout(
        component_name=component.name,
        grid_shape=component.grid.shape,
        data=component.data,
    )
