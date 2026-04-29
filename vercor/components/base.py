from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any

from vercor.clock import ModelDateTime
from vercor.grid import RectilinearGrid
from vercor.run_sequence import RunSequence
from vercor.settings import ComponentSettings, VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import (
        RuntimeComponentContract,
        RuntimeComponentState,
    )


@dataclass(frozen=True)
class ComponentInitContext:
    """Minimal component initialization context owned by the coupler."""

    start: datetime | ModelDateTime
    dt_seconds: float
    run_sequence: RunSequence
    settings: VercorSettings
    logger: Logger


@dataclass(frozen=True)
class RuntimeStepContext:
    """Minimal runtime step context passed to component step boundaries."""

    dt_seconds: float
    settings: VercorSettings
    time: datetime | ModelDateTime | None = None
    logger: Logger | None = None


@dataclass
class Component:
    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: ComponentSettings = field(default_factory=ComponentSettings)
    """A component's default grid dimensions are (nTime, nLev, nLon, nLat)

    Some components may have different dimensions, e.g., sea-ice (nTime, nLon, nLat) or
    JCM atmospheric model (nTime, nLev, nLon, nLat).

    One must implement necessary dimensions check and reshaping of fields
    during import/export if needed.

    Common conventions for exchange fields:
        - All fields must have SI units.
        - Surface fluxes are positive downward and negative upward.

    Attributes:
        name: component name
        grid: component grid
        data: internal storage for component data arrays to/from which fields
                        seed the runtime state during initialization
        settings: component-specific settings
    """

    def initialize(self, context: ComponentInitContext) -> None:
        """Initialize component-owned runtime data before coupling."""

        _ = context

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state."""

        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: RuntimeComponentContract,
    ) -> None:
        """Pre-seed component-specific fields required by runtime execution."""

        _ = data, incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract,
    ) -> None:
        """Validate component-specific runtime fields before execution."""

        _ = component_state, contract

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Return this component advanced by one runtime step."""

        _ = context
        return component_state

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


class HostRuntimeComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""

        _ = component_state, context
        raise NotImplementedError
