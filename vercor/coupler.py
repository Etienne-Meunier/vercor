import logging
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional, cast

import jax
import jax.numpy as jnp

from vercor.clock import Clock
from vercor.components.base import Component, ComponentInitContext
from vercor.exceptions import (
    CouplerError,
    ComponentError,
)
from vercor.exchange import Exchange
from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
)
from vercor.run_sequence import RunSequence
from vercor.output import write_runtime_component_view_to_netcdf
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeCouplerState,
    RuntimeFieldStore,
    RuntimeStepInfo,
    build_runtime_contracts,
    exchange_key_name,
)
from vercor.runtime_components import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
    create_runtime_component_state,
    validate_component_runtime_contract_fields,
)
from vercor.runtime_driver import (
    RuntimeDispatchContext,
    host_component_names,
    prime_runtime_outgoing,
    step_runtime_component,
)
from vercor.runtime_time import (
    build_runtime_step_info,
    initial_runtime_step_info,
    scalar_runtime_step_info,
)
from vercor.runtime_views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.grid_masks import (
    check_total_lnd_ocn_mask_sum,
    get_component,
    grids_identical,
    check_remap_conservation,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.types import RuntimeArray


def setup_logger() -> Logger:
    """
    Setup and return a logger for the Coupler.
    """
    logger = logging.getLogger("VerCOR")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    return logger


@dataclass
class Coupler:
    clock: Clock
    logger: Logger = field(default_factory=setup_logger)
    run_sequence: RunSequence = field(init=False)
    components: dict[str, Component] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    ocn_bmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_bmask_on_atm_grid: RuntimeArray = field(init=False)
    ocn_fmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_fmask_on_atm_grid: RuntimeArray = field(init=False)
    _regridders: dict[
        tuple[str, str, str],
        BilinearRectilinearRegridder | ConservativeRectilinearRegridder,
    ] = field(default_factory=dict)
    _binary_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _fractional_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _runtime_contracts: dict[str, RuntimeComponentContract] = field(
        default_factory=dict
    )
    _compiled_runtime_cache: dict[
        tuple[Any, ...],
        Callable[[RuntimeCouplerState], RuntimeCouplerState],
    ] = field(default_factory=dict, init=False, repr=False)

    """
    Main coupler class to manage components and exchanges between them.

    Attributes:
        clock: Clock instance for managing simulation time
        logger: Logger instance for coupler logging
        run_sequence: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        settings: VercorSettings instance for coupler settings
        ocn_bmask_on_atm_grid: binary ocean mask regridded onto atmosphere grid
        lnd_bmask_on_atm_grid: binary land mask regridded onto atmosphere grid
        ocn_fmask_on_atm_grid: fractional ocean mask regridded onto atmosphere grid
        lnd_fmask_on_atm_grid: fractional land mask regridded onto atmosphere grid
        _regridders: mapping of (source component name, destination component name)
                to Regridder instance (a pool of all available regridders)
        _binary_masks: mapping of (source component name, destination component name)
                to a binary mask array. This mask is used during regridding of fields
                to ensure that only valid (e.g., ocean or land) points are considered
                during the regridding process.
        _fractional_masks: mapping of (source component name, destination component name)
                to a fractional mask array. This mask is applied during field exchanges
                after regridding to ensure that only the appropriate portion from source
                grid cells of the forcing or boundary conditions is transferred to
                destination grid cells, reflecting the partial coverage of source grid cells
                within destination grid cells.
    """

    def register(
        self,
        component: Component,
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        if component.name in self.components:
            raise CouplerError(f"Component {component.name} already registered")

        self.components[component.name] = component
        self.logger.info(f" Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

        self.exchanges.append(exchange)
        formatted_field_names = ", ".join(
            ", ".join(item) if isinstance(item, tuple) else item
            for item in exchange.field_names
        )
        self.logger.info(
            f" Added exchange {exchange.name}: Fields ({formatted_field_names})"
        )

    def set_components_run_sequence(self, run_sequence: RunSequence) -> None:
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: RunSequence instance defining the order of components execution
        """

        for cname in run_sequence:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self, enable_x64_computations: Optional[bool] = None) -> None:
        """
        Initialize the coupler and all registered components.
        """

        self.logger.info(" Initializing coupler and components")

        self.logger.info(
            f" Setting default precision for JAX computations: {self.settings.enable_x64}"
        )
        if enable_x64_computations is not None:
            self.settings.enable_x64 = enable_x64_computations

        if self.settings.enable_x64:
            import jax

            jax.config.update("jax_enable_x64", True)

        init_context = ComponentInitContext(
            start=self.clock.start,
            dt_seconds=self.clock.dt_seconds,
            run_sequence=getattr(self, "run_sequence", RunSequence(order=[])),
            settings=self.settings,
            logger=self.logger,
        )

        # Initialize each component
        for name, component in self.components.items():
            component.initialize(init_context)

            if name not in ("ATM", "OCN", "LND", "ICE"):
                raise ComponentError(
                    f"Incorrect component name: {name}, must be ATM, OCN, LND, or ICE"
                )

            self.logger.info(f" Initialized {name}")

        self._runtime_contracts = build_runtime_contracts(
            tuple(self.components),
            self.exchanges,
            validate_endpoints=True,
        )

        for name, component in self.components.items():
            contract = self._runtime_contracts[name]
            check_not_empty_import_export_lists(component, contract)
            check_valid_exchange_field_names(component, contract)

        self._create_exchange_masks()
        self._validate_land_mask_consistency()
        self.logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")

        # Build regridders per (source component, destination component) pair
        # initialize binary and fractional masks for each regridding pair
        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination, exchange.interpolation_type)

            if key not in self._regridders:
                self._regridders[key] = exchange.create(
                    self.components[exchange.source].grid,
                    self.components[exchange.destination].grid,
                )
                self._binary_masks[key] = jnp.ones(
                    self.components[exchange.destination].grid.shape,
                    dtype=jnp.float_,
                )
                self._fractional_masks[key] = jnp.ones(
                    self.components[exchange.destination].grid.shape,
                    dtype=jnp.float_,
                )
            else:
                self.logger.warning(
                    f" Regridder for exchange {exchange.name} already exists, skipping creation"
                )

        self._patch_exchange_masks()
        self.logger.info(" Exchange masks patching complete")

    def _patch_exchange_masks(self) -> None:
        keys = self._binary_masks.keys()

        for key in keys:
            source, destination, interp_type = key
            if "bilinear" in interp_type:
                if source == "OCN" and destination == "ATM":
                    self._fractional_masks[key] = self.ocn_fmask_on_atm_grid
                elif source == "LND" and destination == "ATM":
                    self._binary_masks[key] = self.lnd_bmask_on_atm_grid
                    self._fractional_masks[key] = self.lnd_fmask_on_atm_grid

    def _create_exchange_masks(self) -> None:
        """
        Create binary and fractional masks for exchanges between
        land, ocean, and atmosphere components.
        """

        land_component = get_component(self.components, "LND")
        atmosphere_component = get_component(self.components, "ATM")
        ocean_component = get_component(self.components, "OCN")

        if not grids_identical(land_component.grid, atmosphere_component.grid):
            raise CouplerError(
                "Land and atmospheric components must use identical horizontal grids"
            )

        # Remapping the binary mask from the mask origin component
        # to the destination component grid
        regridder = ConservativeRectilinearRegridder(
            ocean_component.grid,
            atmosphere_component.grid,
        )

        ocean_binary_mask = ocean_component.grid.binary_mask
        if ocean_binary_mask is None:
            raise ComponentError(
                f"Ocean component {ocean_component.name} has no binary mask defined"
            )

        (
            self.ocn_fmask_on_atm_grid,
            self.lnd_fmask_on_atm_grid,
            self.lnd_bmask_on_atm_grid,
        ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

        check_remap_conservation(
            regridder,
            ocean_binary_mask,
            self.ocn_fmask_on_atm_grid,
        )

        check_total_lnd_ocn_mask_sum(
            self.lnd_fmask_on_atm_grid, self.ocn_fmask_on_atm_grid
        )

    def _validate_land_mask_consistency(self) -> None:
        land_component = get_component(self.components, "LND")
        lnd_mask_from_component = land_component.grid.binary_mask
        if lnd_mask_from_component is not None:
            component_mask = jnp.asarray(lnd_mask_from_component)
            remapped_mask = jnp.asarray(self.lnd_bmask_on_atm_grid)
            if component_mask.shape != self.lnd_bmask_on_atm_grid.shape:
                raise CouplerError(
                    "Land binary mask read from component does not match atmospheric grid shape"
                )
            if not bool(jnp.all(component_mask == remapped_mask)):
                mismatch = int(jnp.count_nonzero(component_mask != remapped_mask))
                raise CouplerError(
                    "Land binary mask created from remapped ocean mask does not match component-provided mask "
                    f"(mismatched points: {mismatch})"
                )

    def _runtime_state_from_components(
        self, *, prefill_missing: bool = False
    ) -> RuntimeCouplerState:
        self._runtime_contracts = build_runtime_contracts(
            tuple(self.components),
            self.exchanges,
            validate_endpoints=False,
        )
        components = tuple(
            create_runtime_component_state(
                component,
                prefill_missing=prefill_missing,
                contract=self._runtime_contracts[name],
            )
            for name, component in self.components.items()
        )
        fractional_masks = {
            exchange_key_name(*key): value
            for key, value in self._fractional_masks.items()
        }
        binary_masks = {
            exchange_key_name(*key): value for key, value in self._binary_masks.items()
        }
        return RuntimeCouplerState(
            component_names=tuple(self.components.keys()),
            components=components,
            fractional_masks=RuntimeFieldStore.from_mapping(fractional_masks),
            binary_masks=RuntimeFieldStore.from_mapping(binary_masks),
        )

    def _validate_runtime_state(self, runtime_state: RuntimeCouplerState) -> None:
        if set(self._runtime_contracts) != set(self.components):
            self._runtime_contracts = build_runtime_contracts(
                tuple(self.components),
                self.exchanges,
                validate_endpoints=False,
            )

        if not hasattr(self, "run_sequence"):
            raise CouplerError("Runtime requires a configured component run sequence")

        run_order = tuple(self.run_sequence)
        if not run_order:
            raise CouplerError("Runtime requires a non-empty component run sequence")

        runtime_component_names = set(runtime_state.component_names)
        for cname in run_order:
            if cname not in self.components:
                raise CouplerError(
                    f"Run-sequence component '{cname}' is not registered in coupler"
                )
            if cname not in runtime_component_names:
                raise CouplerError(
                    f"Run-sequence component '{cname}' is missing from runtime state"
                )

            component = self.components[cname]
            component_state = runtime_state.get_component_state(cname)
            validate_component_runtime_contract_fields(
                component,
                component_state,
                self._runtime_contracts[cname],
            )
            component.validate_runtime_state(
                component_state,
                self._runtime_contracts[cname],
            )

        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination, exchange.interpolation_type)
            if exchange.source not in runtime_component_names:
                raise CouplerError(
                    f"Exchange source component '{exchange.source}' is missing from runtime state"
                )
            if exchange.destination not in runtime_component_names:
                raise CouplerError(
                    f"Exchange destination component '{exchange.destination}' is missing from runtime state"
                )
            if key not in self._regridders:
                raise CouplerError(
                    "Runtime requires an initialized regridder for exchange "
                    f"{exchange.name}"
                )

            mask_name = exchange_key_name(*key)
            if mask_name not in runtime_state.fractional_masks.field_names:
                raise CouplerError(
                    "Runtime requires an initialized fractional mask for exchange "
                    f"{exchange.name}"
                )
            destination_shape = self.components[exchange.destination].grid.shape
            mask_shape = jnp.asarray(
                runtime_state.fractional_masks.get(mask_name)
            ).shape
            if mask_shape != destination_shape:
                raise CouplerError(
                    "Runtime fractional mask for exchange "
                    f"{exchange.name} has shape {mask_shape}, expected {destination_shape}"
                )

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        runtime_state = self._runtime_state_from_components(
            prefill_missing=prefill_missing
        )
        if prefill_missing and hasattr(self, "run_sequence"):
            runtime_state = prime_runtime_outgoing(
                runtime_state,
                tuple(self.run_sequence),
                dispatch_context=self._runtime_dispatch_context(),
                step_info=initial_runtime_step_info(self.clock, self.settings),
            )
        self._validate_runtime_state(runtime_state)
        return runtime_state

    def _output_masks_for_component(
        self,
        name: str,
    ) -> dict[str, RuntimeArray]:
        """Return runtime output mask fields for one destination component."""

        masks = {}
        for exchange in self.exchanges:
            if name != exchange.destination:
                continue

            key = (exchange.source, name, exchange.interpolation_type)
            source_destination_name = "_".join(key)
            masks["bmask_" + source_destination_name] = self._binary_masks[key]
            masks["fmask_" + source_destination_name] = self._fractional_masks[key]
        return masks

    def _runtime_dispatch_context(self) -> RuntimeDispatchContext:
        """Return static runtime dispatch plumbing for the current coupler state."""

        return RuntimeDispatchContext(
            components=self.components,
            exchanges=self.exchanges,
            regridders=self._regridders,
            contracts=self._runtime_contracts,
            dt_seconds=self.clock.dt_seconds,
            settings=self.settings,
        )

    def runtime_component_view(
        self,
        runtime_state: RuntimeCouplerState,
        name: str,
    ) -> RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        return RuntimeComponentView.from_component_state(
            name,
            self.components[name].grid,
            runtime_state.get_component_state(name),
        )

    def finalize(
        self,
        final_state: RuntimeCouplerState,
        output_file_mask: Optional[Path] = None,
    ) -> None:
        """
        Write final runtime component state to component output files.

        Arguments:
            final_state: runtime state returned by run/create_runtime_state
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for name, component in self.components.items():
            if output_file_mask is None:
                filepath = Path(f"{name.lower()}_component_runtime_fields.nc")
            else:
                filepath = Path(f"{name.lower()}_{output_file_mask}.nc")
            view = self.runtime_component_view(final_state, name)
            write_runtime_component_view_to_netcdf(
                view,
                filepath,
                masks=self._output_masks_for_component(name),
            )
            self.logger.info(f" Finalized {name}")

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
            f"├── Exchanges: {', '.join(exchange.name for exchange in self.exchanges)}\n"
            f"└── Run sequence: {', '.join(self.run_sequence)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_sequence={' -> '.join(self.run_sequence)})"

    def run(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        donate_state: bool = False,
    ) -> RuntimeCouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the cached JIT-scanned runtime.
        Host-backed components run through the Python host bridge. When
        ``donate_state`` is true for pure runs, callers must treat the input
        runtime state as consumed after this method returns.
        """

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        self._validate_runtime_state(runtime_state)

        host_names = host_component_names(self.components)
        if not host_names:
            return self._compiled_scanned_runtime(donate_state=donate_state)(
                runtime_state
            )

        if donate_state:
            names = ", ".join(host_names)
            raise CouplerError(
                "Runtime state donation is only supported for differentiable "
                f"components; host-backed component(s) require non-donating run(): {names}"
            )

        return self._run_host_runtime(runtime_state)

    def _run_host_runtime(
        self,
        runtime_state: RuntimeCouplerState,
    ) -> RuntimeCouplerState:
        """Run the host-enabled runtime path for non-differentiable adapters."""

        dispatch_context = self._runtime_dispatch_context()

        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )
            step_info = scalar_runtime_step_info(time, self.clock, self.settings)

            for cname in self.run_sequence:
                self.logger.info(f" Run component: {cname}")
                runtime_state = step_runtime_component(
                    runtime_state,
                    cname,
                    step_info,
                    dispatch_context=dispatch_context,
                    allow_host_runtime=True,
                    time=time,
                    logger=self.logger,
                )

        return runtime_state

    def _compiled_scanned_runtime(
        self, *, donate_state: bool = True
    ) -> Callable[[RuntimeCouplerState], RuntimeCouplerState]:
        """Return a cached JIT-scanned runtime for the current runtime topology."""

        cache_key = self._compiled_runtime_cache_key(donate_state=donate_state)
        if cache_key in self._compiled_runtime_cache:
            return self._compiled_runtime_cache[cache_key]

        def scanned_runtime(
            state: RuntimeCouplerState,
        ) -> RuntimeCouplerState:
            return self._run_scanned_runtime(state, validate_state=False)

        if donate_state:
            compiled_runtime = cast(
                Callable[[RuntimeCouplerState], RuntimeCouplerState],
                jax.jit(scanned_runtime, donate_argnums=(0,)),
            )
        else:
            compiled_runtime = cast(
                Callable[[RuntimeCouplerState], RuntimeCouplerState],
                jax.jit(scanned_runtime),
            )
        self._compiled_runtime_cache[cache_key] = compiled_runtime
        return compiled_runtime

    def _compiled_runtime_cache_key(self, *, donate_state: bool) -> tuple[Any, ...]:
        """Return a static cache key for the compiled pure-runtime wrapper."""

        return (
            donate_state,
            tuple((name, id(component)) for name, component in self.components.items()),
            tuple(self.run_sequence),
            tuple(
                (
                    id(exchange),
                    exchange.source,
                    exchange.destination,
                    exchange.interpolation_type,
                    tuple(exchange.field_names),
                )
                for exchange in self.exchanges
            ),
            tuple(sorted((key, id(value)) for key, value in self._regridders.items())),
            tuple(
                (name, contract.imports, contract.exports)
                for name, contract in sorted(self._runtime_contracts.items())
            ),
            repr(self.clock.start),
            self.clock.dt_seconds,
            self.clock.steps,
            self.clock.year_type,
            self.settings.year_in_seconds,
        )

    def _run_scanned_runtime(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        validate_state: bool = True,
    ) -> RuntimeCouplerState:
        """Run the unified runtime path under ``jax.lax.scan`` and return state."""

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        if validate_state:
            self._validate_runtime_state(runtime_state)
        step_infos = build_runtime_step_info(self.clock, self.settings)
        dispatch_context = self._runtime_dispatch_context()

        def step_all_components(
            state: RuntimeCouplerState, step_info: RuntimeStepInfo
        ) -> tuple[RuntimeCouplerState, None]:
            for cname in self.run_sequence:
                state = step_runtime_component(
                    state,
                    cname,
                    step_info,
                    dispatch_context=dispatch_context,
                    allow_host_runtime=False,
                )
            return state, None

        final_state, _ = jax.lax.scan(
            step_all_components,
            runtime_state,
            step_infos,
            length=self.clock.steps,
        )
        return final_state
