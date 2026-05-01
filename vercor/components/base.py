from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, final

from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.settings import ComponentSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import (
        RuntimeComponentContract,
        RuntimeComponentState,
    )


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
        - default grid dimensions (nTime, nLev, nLat, nLon)

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
    settings: ComponentSettings = field(default_factory=ComponentSettings)

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
    if not isinstance(component.settings, ComponentSettings):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'settings'; "
            "expected ComponentSettings."
        )
