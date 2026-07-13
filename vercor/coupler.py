from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

from vercor.clock import Clock
from vercor.components._adapter import normalize_component
from vercor.components.contracts import ComponentInfo
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.fields import VectorField
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger as _setup_logger,
)
from vercor._run_order import normalize_run_order
import vercor._runtime.facade as _runtime_facade
from vercor._runtime.prepared import PreparedCoupling
from vercor.runtime import RuntimeOptions
from vercor.settings import Settings
from vercor.state import RunState

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contracts import ComponentLike


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
        _prepared: single private prepared runtime boundary, created lazily.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        components: Iterable["ComponentLike"] = (),
        exchanges: Iterable[Exchange] = (),
        run_order: Sequence[str] = (),
        runtime: RuntimeOptions | None = None,
        logger: LoggerLike | None = None,
        log_level: int | str = "INFO",
    ) -> None:
        """Create a coupler from public configuration objects."""

        self.clock = clock
        self.log_level = log_level
        self.logger = logger if logger is not None else _setup_logger()
        self.runtime = RuntimeOptions() if runtime is None else runtime
        self.settings = Settings(
            enable_x64=self.runtime.dtype.enable_x64,
        )
        self._components: dict[str, Component] = {}
        self._components_view: MappingProxyType[str, Component] = MappingProxyType(
            self._components
        )
        self._exchanges: tuple[Exchange, ...] = ()
        self._run_order: tuple[str, ...] = ()
        self._prepared: PreparedCoupling | None = None
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

    def _invalidate_preparation(self) -> None:
        """Clear prepared runtime resources after public setup changes."""

        self._prepared = None

    @property
    def components(self) -> MappingProxyType[str, ComponentInfo]:
        """Return read-only public descriptions of registered components."""

        return MappingProxyType(
            {
                name: ComponentInfo(
                    name=component.name,
                    grid=component.grid,
                    spec=component.spec,
                )
                for name, component in self._components.items()
            }
        )

    @property
    def _runtime_components(self) -> MappingProxyType[str, "Component"]:
        """Return normalized component adapters for private runtime use."""

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
        component: "ComponentLike",
    ) -> Self:
        """Register a component with the coupler."""

        normalized_component = normalize_component(component)
        if normalized_component.name in self._components:
            raise CouplerError(
                f"Component {normalized_component.name} already registered"
            )

        self._components[normalized_component.name] = normalized_component
        self._invalidate_preparation()
        self.logger.info(f"Registered component {normalized_component.name}")
        return self

    def add_exchange(self, exchange: Exchange) -> Self:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

        self._exchanges = (*self._exchanges, exchange)
        self._invalidate_preparation()
        formatted_field_names = ", ".join(
            ", ".join((item.u, item.v)) if isinstance(item, VectorField) else item
            for item in exchange.fields
        )
        self.logger.info(
            f"Added exchange {exchange.label}: Fields ({formatted_field_names})"
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
            if cname not in self._components:
                raise CouplerError(f"Component {cname} not registered in coupler")
        self._run_order = normalized_run_order
        self._invalidate_preparation()
        self.logger.info(
            f"Set coupler components run order: {', '.join(self.run_order)}"
        )
        return self

    def _ensure_prepared(self) -> PreparedCoupling:
        """Return the one prepared runtime boundary for this configuration."""

        if self._prepared is not None:
            self._prepared.validate_configuration(
                self._runtime_components,
                clock=self.clock,
                settings=self.settings,
                runtime=self.runtime,
            )
            return self._prepared
        prepared = _runtime_facade.prepare_coupling(
            components=self._runtime_components,
            exchanges=self.exchanges,
            run_order=self.run_order,
            clock=self.clock,
            settings=self.settings,
            runtime=self.runtime,
            logger=self.logger,
        )
        self._prepared = prepared
        return prepared

    def _initialize_runtime(self) -> None:
        """Prepare components, topology, contracts, and runtime dispatch once."""

        self._ensure_prepared()

    def initial_state(self, *, prefill_missing: bool = True) -> RunState:
        """Create and validate the coupled runtime state."""

        prepared = self._ensure_prepared()
        return _runtime_facade.create_runtime_state(
            prepared=prepared,
            prefill_missing=prefill_missing,
        )

    def write_outputs(
        self,
        state: RunState,
        *,
        output_dir: Path = Path("."),
        filename_template: str = "{component}.runtime_fields.nc",
        write_snapshots: bool = True,
    ) -> None:
        """Write final runtime fields and optional native component snapshots."""

        prepared = self._ensure_prepared()
        _runtime_facade.validate_runtime_state(state, prepared=prepared)
        self.logger.info("------------ Writing coupler outputs ------------")
        _runtime_facade.finalize(
            final_state=state,
            prepared=prepared,
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
                for name, component in self._components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.label for exchange in self.exchanges)}\n"
            f"└── Run order: {', '.join(self.run_order)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_order={' -> '.join(self.run_order)})"

    def run(
        self,
        state: RunState | None = None,
    ) -> RunState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the JIT-scanned runtime.
        Host-backed components run through the Python host bridge.
        """

        prepared = self._ensure_prepared()
        runtime_state = _runtime_facade.prepare_runtime_state(
            state,
            prepared=prepared,
        )
        return _runtime_facade.run(
            runtime_state,
            prepared=prepared,
            logger=self.logger,
        )
