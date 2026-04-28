import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Callable, Optional, cast

import jax
import jax.numpy as jnp

from vercor.clock import Clock, ModelDateTime
from vercor.components.base import (
    ComponentInitContext,
    HostRuntimeComponent,
    RuntimeStepContext,
)
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
    dispatch_component_exchanges,
    exchange_key_name,
)
from vercor.runtime_components import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
    create_runtime_component_state,
    receive_runtime_fields,
    send_runtime_fields,
    validate_component_runtime_state,
)
from vercor.runtime_views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.tools import (
    check_total_lnd_ocn_mask_sum,
    datetime_to_seconds_in_year,
    get_periodic_interval,
    get_component,
    grids_identical,
    is_leap_year,
    _append_unique,
    _flatten_fields,
    check_remap_conservation,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.types import AllComponentsType, RuntimeArray


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
    components: dict[
        str,
        AllComponentsType,
    ] = field(default_factory=dict)
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
        component: AllComponentsType,
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

        self._runtime_contracts = self._build_runtime_contracts(validate_endpoints=True)

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
        self._runtime_contracts = self._build_runtime_contracts(
            validate_endpoints=False
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

    def _extend_contract_fields(
        self,
        fields: tuple[str, ...],
        new_fields: list[str],
    ) -> tuple[str, ...]:
        """Return ``fields`` extended by new unique field names."""

        updated = list(fields)
        _append_unique(updated, new_fields)
        return tuple(updated)

    def _build_runtime_contracts(
        self,
        *,
        validate_endpoints: bool,
    ) -> dict[str, RuntimeComponentContract]:
        """Build coupler-owned runtime field contracts from exchanges."""

        contracts = {name: RuntimeComponentContract.empty() for name in self.components}
        for exchange in self.exchanges:
            if exchange.source not in self.components:
                if validate_endpoints:
                    raise CouplerError(
                        f"Source component '{exchange.source}' not registered in coupler"
                    )
                continue
            if exchange.destination not in self.components:
                if validate_endpoints:
                    raise CouplerError(
                        f"Destination component '{exchange.destination}' not registered in coupler"
                    )
                continue

            flattened_fields = _flatten_fields(exchange.field_names)
            source_contract = contracts[exchange.source]
            destination_contract = contracts[exchange.destination]
            contracts[exchange.source] = RuntimeComponentContract(
                imports=source_contract.imports,
                exports=self._extend_contract_fields(
                    source_contract.exports,
                    flattened_fields,
                ),
            )
            contracts[exchange.destination] = RuntimeComponentContract(
                imports=self._extend_contract_fields(
                    destination_contract.imports,
                    flattened_fields,
                ),
                exports=destination_contract.exports,
            )
        return contracts

    def _runtime_step_info_from_times(
        self,
        times: list[datetime | ModelDateTime],
    ) -> RuntimeStepInfo:
        monthly_index_left: list[int] = []
        monthly_index_right: list[int] = []
        monthly_weight_left: list[float] = []
        monthly_weight_right: list[float] = []
        daily_index: list[int] = []

        for time in times:
            total_seconds = datetime_to_seconds_in_year(time)
            (n1, f1), (n2, f2) = get_periodic_interval(
                current_time=total_seconds,
                cycle_length=self.settings.year_in_seconds,
                rec_spacing=self.settings.year_in_seconds / 12.0,
                n_rec=12,
            )
            monthly_index_left.append(n1)
            monthly_index_right.append(n2)
            monthly_weight_left.append(f1)
            monthly_weight_right.append(f2)
            daily_index.append(self._runtime_daily_index(time))

        return RuntimeStepInfo.from_sequences(
            monthly_index_left,
            monthly_index_right,
            monthly_weight_left,
            monthly_weight_right,
            daily_index,
        )

    def _runtime_daily_index(self, time: datetime | ModelDateTime) -> int:
        if self.clock.year_type == "360":
            month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
            month_length = month_lengths[time.month - 1]
            mapped_day = ((time.day - 1) * (month_length - 1)) // 29 + 1
            day_of_year = sum(month_lengths[: time.month - 1]) + mapped_day
        elif self.clock.year_type == "noleap":
            model_day_of_year = getattr(time, "day_of_year", None)
            if model_day_of_year is None:
                raise ValueError("ModelDateTime.day_of_year is not initialized")
            day_of_year = model_day_of_year
        else:
            day_of_year = time.timetuple().tm_yday
            if is_leap_year(time.year) and day_of_year > 59:
                day_of_year -= 1

        return day_of_year - 1

    def _build_runtime_step_info(self) -> RuntimeStepInfo:
        times = [time for _, time, _ in self.clock.iter()]
        return self._runtime_step_info_from_times(times)

    def _initial_runtime_step_info(self) -> RuntimeStepInfo:
        clock_iter = self.clock.iter()
        try:
            _, first_time, _ = next(clock_iter)
        except StopIteration:
            first_time = self.clock.start
        batched_step_info = self._runtime_step_info_from_times([first_time])
        return cast(
            RuntimeStepInfo,
            jax.tree_util.tree_map(lambda value: value[0], batched_step_info),
        )

    def _prime_runtime_outgoing(
        self,
        runtime_state: RuntimeCouplerState,
    ) -> RuntimeCouplerState:
        step_info = self._initial_runtime_step_info()
        for cname in self.run_sequence:
            component_state = runtime_state.get_component_state(cname)
            component_state = send_runtime_fields(
                self.components[cname],
                component_state,
                step_info,
                contract=self._runtime_contracts[cname],
            )
            runtime_state = runtime_state.set_component_state(cname, component_state)
        return runtime_state

    def _validate_runtime_state(self, runtime_state: RuntimeCouplerState) -> None:
        if set(self._runtime_contracts) != set(self.components):
            self._runtime_contracts = self._build_runtime_contracts(
                validate_endpoints=False
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
            validate_component_runtime_state(
                component,
                component_state,
                self._runtime_contracts.get(
                    cname,
                    RuntimeComponentContract.empty(),
                ),
            )
            component.validate_runtime_state(
                component_state,
                self._runtime_contracts.get(
                    cname,
                    RuntimeComponentContract.empty(),
                ),
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
            runtime_state = self._prime_runtime_outgoing(runtime_state)
        self._validate_runtime_state(runtime_state)
        return runtime_state

    def _scalar_runtime_step_info(
        self,
        time: datetime | ModelDateTime,
    ) -> RuntimeStepInfo:
        batched_step_info = self._runtime_step_info_from_times([time])
        return cast(
            RuntimeStepInfo,
            jax.tree_util.tree_map(lambda value: value[0], batched_step_info),
        )

    def _step_runtime_component(
        self,
        runtime_state: RuntimeCouplerState,
        component_name: str,
        step_info: RuntimeStepInfo,
        *,
        time: datetime | ModelDateTime | None = None,
    ) -> RuntimeCouplerState:
        runtime_state = dispatch_component_exchanges(
            runtime_state,
            component_name,
            self.exchanges,
            self._regridders,
        )
        component_state = runtime_state.get_component_state(component_name)
        component = self.components[component_name]
        contract = self._runtime_contracts.get(
            component_name,
            RuntimeComponentContract.empty(),
        )
        component_state = receive_runtime_fields(
            component_state,
            contract,
        )
        step_context = RuntimeStepContext(
            dt_seconds=self.clock.dt_seconds,
            settings=self.settings,
            time=time,
            logger=self.logger if time is not None else None,
        )
        if time is not None and isinstance(component, HostRuntimeComponent):
            component_state = component._step_host_runtime_state(
                component_state,
                step_context,
            )
        else:
            component_state = component.step_runtime_state(
                component_state,
                step_context,
            )
        component_state = send_runtime_fields(
            component,
            component_state,
            step_info,
            contract=contract,
        )
        runtime_state = runtime_state.set_component_state(
            component_name,
            component_state,
        )

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
    ) -> RuntimeCouplerState:
        """
        Run the coupler and all registered components according to the run sequence.
        """

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        self._validate_runtime_state(runtime_state)

        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )
            step_info = self._scalar_runtime_step_info(time)

            for cname in self.run_sequence:
                self.logger.info(f" Run component: {cname}")
                runtime_state = self._step_runtime_component(
                    runtime_state,
                    cname,
                    step_info,
                    time=time,
                )

        return runtime_state

    def compile_runtime(
        self, *, donate_state: bool = True
    ) -> Callable[[RuntimeCouplerState], RuntimeCouplerState]:
        """Return a reusable compiled scanned-runtime callable.

        The returned callable runs the pure scanned runtime under ``jax.jit``.
        Reuse the same compiled callable for repeated runs with the same
        runtime-state PyTree structure and array shapes to maximize compile
        cache hits. Host-backed adapters that require Python object mutation
        should use ``run()`` instead of the scanned runtime.

        If ``donate_state`` is true, the input ``RuntimeCouplerState`` is donated
        to XLA at the outer runtime boundary. Callers must treat the donated input
        state as consumed and must not read it after invoking the compiled
        callable.
        """
        host_component_names = [
            name
            for name, component in self.components.items()
            if isinstance(component, HostRuntimeComponent)
        ]
        if host_component_names:
            names = ", ".join(host_component_names)
            raise CouplerError(
                "compile_runtime() only supports differentiable runtime components; "
                f"host-backed component(s) require run(): {names}"
            )

        def scanned_runtime(
            state: RuntimeCouplerState,
        ) -> RuntimeCouplerState:
            return self._run_scanned_runtime(state)

        if donate_state:
            return cast(
                Callable[[RuntimeCouplerState], RuntimeCouplerState],
                jax.jit(scanned_runtime, donate_argnums=(0,)),
            )
        return cast(
            Callable[[RuntimeCouplerState], RuntimeCouplerState],
            jax.jit(scanned_runtime),
        )

    def _run_scanned_runtime(
        self, initial_state: RuntimeCouplerState | None = None
    ) -> RuntimeCouplerState:
        """Run the unified runtime path under ``jax.lax.scan`` and return state."""

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        self._validate_runtime_state(runtime_state)
        step_infos = self._build_runtime_step_info()

        def step_all_components(
            state: RuntimeCouplerState, step_info: RuntimeStepInfo
        ) -> tuple[RuntimeCouplerState, None]:
            for cname in self.run_sequence:
                state = self._step_runtime_component(
                    state,
                    cname,
                    step_info,
                )
            return state, None

        final_state, _ = jax.lax.scan(
            step_all_components,
            runtime_state,
            step_infos,
            length=self.clock.steps,
        )
        return final_state
