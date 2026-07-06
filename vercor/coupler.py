from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

from vercor.clock import Clock
from vercor.components.setup_validation import validate_component_setup
from vercor.exceptions import CouplerError
from vercor._exchange import Exchange
from vercor.fields import VectorField
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger as _setup_logger,
)
from vercor._run_order import normalize_run_order
import vercor.runtime.facade as _runtime_facade
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.runtime.state import CouplerState
from vercor.runtime.views import ComponentView
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.base import Component


class Coupler:
    """Public orchestration facade for configured component integrations.

    The coupler owns registration, exchange declarations, run sequence,
    regridder/mask setup, runtime-state creation, execution, and final output.
    The differentiable integration itself operates on immutable runtime state;
    component objects remain setup/configuration adapters rather than the
    traced integration state.

    Attributes:
        clock: Clock instance for managing simulation time
        log_level: logging threshold for coupler logs (e.g., "INFO", "DEBUG", etc.)
        logger: Logger instance for coupler logging
        run_order: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        settings: Settings instance for coupler settings
        lnd_bmask_on_atm_grid: binary land mask regridded onto atmosphere grid
        ocn_fmask_on_atm_grid: fractional ocean mask regridded onto atmosphere grid
        lnd_fmask_on_atm_grid: fractional land mask regridded onto atmosphere grid
        _runtime_resources: runtime-owned holder for topology maps, runtime
            contracts, and interrupt controller.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        components: Iterable["Component"] = (),
        exchanges: Iterable[Exchange] = (),
        run_order: Sequence[str] = (),
        settings: Settings | None = None,
        logger: LoggerLike | None = None,
        log_level: int | str = "INFO",
    ) -> None:
        """Create a coupler from public configuration objects."""

        self.clock = clock
        self.log_level = log_level
        self.logger = logger if logger is not None else _setup_logger()
        self.settings = settings or Settings()
        self._components: dict[str, Component] = {}
        self._components_view: MappingProxyType[str, Component] = MappingProxyType(
            self._components
        )
        self._exchanges: tuple[Exchange, ...] = ()
        self._run_order: tuple[str, ...] = ()
        self._runtime_resources = CouplerRuntimeResources()
        configured_run_order = normalize_run_order(run_order)

        if isinstance(self.logger, logging.Logger):
            self.logger = JaxCallbackLogger(
                configure_python_logger(self.logger, self.log_level)
            )
        elif isinstance(self.logger, JaxCallbackLogger):
            configure_python_logger(self.logger.logger, self.log_level)

        set_level = getattr(self.logger, "setLevel", None)
        if callable(set_level):
            set_level(self.log_level)

        for component in components:
            self.add_component(component)
        for exchange in exchanges:
            self.add_exchange(exchange)
        if configured_run_order:
            self.set_run_order(configured_run_order)

    def _invalidate_runtime_resources(self) -> None:
        """Clear cached runtime topology and contracts after setup changes."""

        self._runtime_resources = CouplerRuntimeResources()

    @property
    def components(self) -> MappingProxyType[str, "Component"]:
        """Return a read-only view of registered components by name."""

        return self._components_view

    @property
    def exchanges(self) -> tuple[Exchange, ...]:
        """Return immutable exchange declarations."""

        return self._exchanges

    @property
    def run_order(self) -> tuple[str, ...]:
        """Return component names in runtime execution order."""

        return self._run_order

    def add_component(
        self,
        component: "Component",
    ) -> Self:
        """Register a component with the coupler."""

        validate_component_setup(component)
        if component.name in self.components:
            raise CouplerError(f"Component {component.name} already registered")

        self._components[component.name] = component
        self._invalidate_runtime_resources()
        self.logger.info(f" Registered component {component.name}")
        return self

    def add_exchange(self, exchange: Exchange) -> Self:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

        self._exchanges = (*self._exchanges, exchange)
        self._invalidate_runtime_resources()
        formatted_field_names = ", ".join(
            ", ".join((item.u, item.v)) if isinstance(item, VectorField) else item
            for item in exchange.fields
        )
        self.logger.info(
            f" Added exchange {exchange.label}: Fields ({formatted_field_names})"
        )
        return self

    def add_exchanges(self, exchanges: Iterable[Exchange]) -> Self:
        """Add multiple exchange definitions to the coupler."""

        for exchange in exchanges:
            self.add_exchange(exchange)
        return self

    def set_run_order(
        self,
        run_order: Sequence[str],
    ) -> Self:
        """Set the run order for coupler components."""

        normalized_run_order = normalize_run_order(run_order)
        for cname in normalized_run_order:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self._run_order = normalized_run_order
        self._invalidate_runtime_resources()
        self.logger.info(
            f" Set coupler components run order: {', '.join(self.run_order)}"
        )
        return self

    def _runtime_inputs(self) -> _runtime_facade.RuntimeFacadeInputs:
        """Return the repeated runtime facade input bundle for this coupler."""

        return _runtime_facade.RuntimeFacadeInputs(
            self.components,
            self.exchanges,
            self._runtime_resources,
            self.run_order,
            self.clock,
            self.settings,
        )

    def initialize(self) -> None:
        """
        Initialize the coupler and all registered components.
        """

        initialized = _runtime_facade.initialize_coupler_runtime(
            inputs=self._runtime_inputs(),
            logger=self.logger,
        )

        topology = initialized.topology
        surface_masks = topology.surface_masks
        self.ocn_fmask_on_atm_grid = surface_masks.ocn_fmask_on_atm_grid
        self.lnd_fmask_on_atm_grid = surface_masks.lnd_fmask_on_atm_grid
        self.lnd_bmask_on_atm_grid = surface_masks.lnd_bmask_on_atm_grid

    def state(self, *, prefill: bool = True) -> CouplerState:
        """Create and validate the coupled runtime state."""

        return _runtime_facade.create_runtime_state(
            inputs=self._runtime_inputs(),
            prefill_missing=prefill,
        )

    def view(
        self,
        state: CouplerState,
        name: str,
    ) -> ComponentView:
        """Return a component view for diagnostics and output."""

        return _runtime_facade.runtime_component_view(
            components=self.components,
            runtime_state=state,
            name=name,
        )

    def views(
        self,
        state: CouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, ComponentView]:
        """Return component views for diagnostics and output."""

        return _runtime_facade.runtime_component_views(
            components=self.components,
            runtime_state=state,
            names=names,
        )

    def write_outputs(
        self,
        state: CouplerState,
        *,
        output_dir: Path = Path("."),
        filename_template: str = "{component}.runtime_fields.nc",
        write_snapshots: bool = True,
    ) -> None:
        """Write final runtime fields and optional native component snapshots."""

        self.logger.info(" ------------ Writing coupler outputs ------------")
        _runtime_facade.finalize(
            final_state=state,
            inputs=self._runtime_inputs(),
            output_dir=output_dir,
            filename_template=filename_template,
            write_snapshots=write_snapshots,
            logger=self.logger,
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Run start: {self.clock.start}\n"
            f"├── Components: "
            + ", ".join(
                f"<{component.__class__.__name__}>({name})"
                for name, component in self.components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.label for exchange in self.exchanges)}\n"
            f"└── Run order: {', '.join(self.run_order)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_order={' -> '.join(self.run_order)})"

    def run(
        self,
        state: CouplerState | None = None,
    ) -> CouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the JIT-scanned runtime.
        Host-backed components run through the Python host bridge.
        """

        inputs = self._runtime_inputs()
        runtime_state = _runtime_facade.prepare_runtime_state(
            state,
            inputs=inputs,
        )
        return _runtime_facade.run(
            runtime_state,
            inputs=inputs,
            logger=self.logger,
        )
