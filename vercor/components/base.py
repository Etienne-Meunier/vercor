from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, final

from vercor.exceptions import ComponentError
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
                    "factory fields, or declare it through an exchange before "
                    "runtime execution."
                )
            data = data.set(field_name, field_value)
        return component_state.with_data(data)

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

        _ = data, incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract,
    ) -> None:
        """Optionally validate component-specific runtime fields before execution.

        Override this hook to report missing payloads, diagnostic fields, or
        non-standard shapes before traced runtime execution begins.
        """

        _ = component_state, contract

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


class _CallableComponent(Component):
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
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = step
        self._payload = payload
        if fields is not None:
            self.seed_fields(fields)

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        return self._payload

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


class _CallableHostRuntimeComponent(HostRuntimeComponent):
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
    ) -> None:
        if settings is None:
            super().__init__(name=name, grid=grid)
        else:
            super().__init__(name=name, grid=grid, settings=settings)
        self._step = step
        self._payload = payload
        if fields is not None:
            self.seed_fields(fields)

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable host-component factory."""

        return self._payload

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


def make_data_component(
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, RuntimeArray] | None = None,
    settings: VercorSettings | None = None,
) -> DataComponent:
    """Create a data-only component from optional seeded grid fields."""

    if settings is None:
        component = DataComponent(name=name, grid=grid)
    else:
        component = DataComponent(name=name, grid=grid, settings=settings)
    if fields is not None:
        component.seed_fields(fields)
    return component


def make_differentiable_component(
    name: str,
    grid: RectilinearGrid,
    step: _ComponentStepCallable,
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
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
    )


def make_host_component(
    name: str,
    grid: RectilinearGrid,
    step: _ComponentStepCallable,
    fields: Mapping[str, RuntimeArray] | None = None,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
) -> HostRuntimeComponent:
    """Create a host-runtime component from a Python runtime step callable."""

    return _CallableHostRuntimeComponent(
        name=name,
        grid=grid,
        step=step,
        fields=fields,
        payload=payload,
        settings=settings,
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
