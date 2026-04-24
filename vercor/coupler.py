import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp

from vercor.clock import Clock, ModelDateTime
from vercor.components import Shared
from vercor.components import TimedNamedArray as TNA
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
from vercor.runtime import (
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
    RuntimeStepInfo,
    JAXGCMRuntimePayload,
    create_component_runtime_payload,
    dispatch_component_exchanges,
    exchange_key_name,
    is_supported_differentiable_component,
    receive_component_fields,
    send_component_fields,
    step_component_state,
)
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

        # Initialize each component
        for name, component in self.components.items():
            component.initialize(self)

            if name not in ("ATM", "OCN", "LND", "ICE"):
                raise ComponentError(
                    f"Incorrect component name: {name}, must be ATM, OCN, LND, or ICE"
                )

            self.logger.info(f" Initialized {name}")

        # Setup components' import/export field lists based on exchanges
        for exchange in self.exchanges:
            if exchange.source not in self.components:
                raise CouplerError(
                    f"Source component '{exchange.source}' not registered in coupler"
                )
            if exchange.destination not in self.components:
                raise CouplerError(
                    f"Destination component '{exchange.destination}' not registered in coupler"
                )

            source_component = self.components[exchange.source]
            destination_component = self.components[exchange.destination]

            flattened_fields = _flatten_fields(exchange.field_names)
            _append_unique(source_component._fields2export, flattened_fields)
            _append_unique(destination_component._fields2import, flattened_fields)

        for name, component in self.components.items():
            component.check_not_empty_import_export_lists()
            component.check_valid_exchange_field_names()
            # Deposit initial data to be sent from component to coupler
            component.send_fields(self.clock.start, self)

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

    def _component_to_runtime_state(
        self,
        component: AllComponentsType,
        *,
        prefill_missing: bool,
    ) -> RuntimeComponentState:
        data = dict(component.data)
        incoming = component.incoming_fields.fields()
        outgoing = component.outgoing_fields.fields()

        if prefill_missing:
            zeros = jnp.zeros(component.grid.shape, dtype=jnp.float_)
            if component.__class__.__name__ == "ERA5Atmosphere":
                data.setdefault("total_surface_temperature", zeros)
            if self._is_jax_gcm_component(component):
                jax_gcm_component = cast(Any, component)
                data.setdefault("total_surface_temperature", zeros)
                for field_name in (
                    "u_velocity",
                    "v_velocity",
                    "temperature",
                    "specific_humidity",
                    "sensible_heat_flux",
                    "latent_heat_flux",
                    "net_shortwave_radiation_flux",
                    "downward_longwave_radiation_flux",
                    "density",
                    "potential_temperature",
                    "model_level_height",
                ):
                    data.setdefault(field_name, zeros)
                sigma_levels = jnp.asarray(jax_gcm_component.sigma_levels)
                data.setdefault(
                    "pressure",
                    jnp.zeros(
                        (sigma_levels.shape[0], *component.grid.shape),
                        dtype=jnp.float_,
                    ),
                )
            for field_name in component._fields2import:
                incoming.setdefault(field_name, zeros)
                data.setdefault(field_name, zeros)
            for field_name in component._fields2export:
                outgoing.setdefault(field_name, data.get(field_name, zeros))
                data.setdefault(field_name, zeros)

        return RuntimeComponentState(
            name=component.name,
            data=RuntimeFieldStore.from_mapping(data),
            incoming=RuntimeFieldStore.from_mapping(incoming),
            outgoing=RuntimeFieldStore.from_mapping(outgoing),
            fields_to_import=tuple(component._fields2import),
            fields_to_export=tuple(component._fields2export),
            runtime_payload=create_component_runtime_payload(component),
        )

    def _runtime_state_from_components(
        self, *, prefill_missing: bool = False
    ) -> RuntimeCouplerState:
        components = tuple(
            self._component_to_runtime_state(
                component,
                prefill_missing=prefill_missing,
            )
            for component in self.components.values()
        )
        fractional_masks = {
            exchange_key_name(*key): value
            for key, value in self._fractional_masks.items()
        }
        binary_masks = {
            exchange_key_name(*key): value for key, value in self._binary_masks.items()
        }
        return RuntimeCouplerState(
            components=components,
            fractional_masks=RuntimeFieldStore.from_mapping(fractional_masks),
            binary_masks=RuntimeFieldStore.from_mapping(binary_masks),
        )

    def _validate_runtime_store_field(
        self,
        component_name: str,
        store: RuntimeFieldStore,
        field_name: str,
        store_description: str,
        expected_shape: tuple[int, int],
    ) -> None:
        if field_name not in store.field_names:
            raise CouplerError(
                "Differentiable runtime missing "
                f"{store_description} field '{field_name}' for component '{component_name}'"
            )

        field_shape = jnp.asarray(store.get(field_name)).shape
        if field_shape != expected_shape:
            raise CouplerError(
                "Differentiable runtime "
                f"{store_description} field '{field_name}' for component '{component_name}' "
                f"has shape {field_shape}, expected {expected_shape}"
            )

    def _is_jax_gcm_component(self, component: AllComponentsType) -> bool:
        return (
            component.__class__.__module__,
            component.__class__.__name__,
        ) == ("vercor.components.external.jax_gcm", "JAXGCM")

    def _validate_jax_gcm_runtime_payload(
        self,
        component_name: str,
        component_state: RuntimeComponentState,
    ) -> None:
        if not isinstance(component_state.runtime_payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "Differentiable JAXGCM runtime requires an initialized immutable "
                f"runtime payload for component '{component_name}'"
            )

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

    def _build_differentiable_step_info(self) -> RuntimeStepInfo:
        times = [time for _, time, _ in self.clock.iter()]
        return self._runtime_step_info_from_times(times)

    def _initial_differentiable_step_info(self) -> RuntimeStepInfo:
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

    def _prime_differentiable_runtime_outgoing(
        self,
        runtime_state: RuntimeCouplerState,
    ) -> RuntimeCouplerState:
        step_info = self._initial_differentiable_step_info()
        for cname in self.run_sequence:
            component_state = runtime_state.get_component_state(cname)
            component_state = send_component_fields(
                component_state,
                self.components[cname],
                step_info,
            )
            runtime_state = runtime_state.set_component_state(component_state)
        return runtime_state

    def _validate_differentiable_runtime(
        self, runtime_state: RuntimeCouplerState
    ) -> None:
        if not hasattr(self, "run_sequence"):
            raise CouplerError(
                "Differentiable runtime requires a configured component run sequence"
            )

        run_order = tuple(self.run_sequence)
        if not run_order:
            raise CouplerError(
                "Differentiable runtime requires a non-empty component run sequence"
            )

        runtime_component_names = set(runtime_state.component_names)
        for cname in run_order:
            if cname not in self.components:
                raise CouplerError(
                    f"Run-sequence component '{cname}' is not registered in coupler"
                )
            if cname not in runtime_component_names:
                raise CouplerError(
                    f"Run-sequence component '{cname}' is missing from differentiable state"
                )

            component = self.components[cname]
            if not is_supported_differentiable_component(component):
                raise ComponentError(
                    "Differentiable runtime currently supports VerCOR slab components "
                    "pure data-forcing components, and JAXGCM "
                    f"(got {component.__class__.__name__} for component '{cname}')"
                )
            component_state = runtime_state.get_component_state(cname)
            if self._is_jax_gcm_component(component):
                self._validate_jax_gcm_runtime_payload(cname, component_state)
            if (
                component.__class__.__name__ == "ERA5Atmosphere"
                and "total_surface_temperature" not in component_state.data.field_names
            ):
                raise CouplerError(
                    "Differentiable runtime missing data field "
                    f"'total_surface_temperature' for component '{cname}'"
                )
            for field_name in component_state.fields_to_export:
                self._validate_runtime_store_field(
                    cname,
                    component_state.outgoing,
                    field_name,
                    "exported source",
                    component.grid.shape,
                )
            if component.__class__.__module__.startswith("vercor.components.slab."):
                for field_name in component_state.data.field_names:
                    self._validate_runtime_store_field(
                        cname,
                        component_state.data,
                        field_name,
                        "data",
                        component.grid.shape,
                    )
            for field_name in component_state.incoming.field_names:
                self._validate_runtime_store_field(
                    cname,
                    component_state.incoming,
                    field_name,
                    "incoming",
                    component.grid.shape,
                )

        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination, exchange.interpolation_type)
            if exchange.source not in runtime_component_names:
                raise CouplerError(
                    f"Exchange source component '{exchange.source}' is missing from differentiable state"
                )
            if exchange.destination not in runtime_component_names:
                raise CouplerError(
                    f"Exchange destination component '{exchange.destination}' is missing from differentiable state"
                )
            if key not in self._regridders:
                raise CouplerError(
                    "Differentiable runtime requires an initialized regridder for exchange "
                    f"{exchange.name}"
                )

            mask_name = exchange_key_name(*key)
            if mask_name not in runtime_state.fractional_masks.field_names:
                raise CouplerError(
                    "Differentiable runtime requires an initialized fractional mask for exchange "
                    f"{exchange.name}"
                )
            destination_shape = self.components[exchange.destination].grid.shape
            mask_shape = jnp.asarray(
                runtime_state.fractional_masks.get(mask_name)
            ).shape
            if mask_shape != destination_shape:
                raise CouplerError(
                    "Differentiable runtime fractional mask for exchange "
                    f"{exchange.name} has shape {mask_shape}, expected {destination_shape}"
                )

            source_shape = self.components[exchange.source].grid.shape
            source_state = runtime_state.get_component_state(exchange.source)
            for field_name in _flatten_fields(exchange.field_names):
                self._validate_runtime_store_field(
                    exchange.source,
                    source_state.outgoing,
                    field_name,
                    "source",
                    source_shape,
                )

    def create_differentiable_state(
        self, *, prefill_missing: bool = True
    ) -> RuntimeCouplerState:
        """Create and validate the immutable state used by ``run_differentiable``."""

        runtime_state = self._runtime_state_from_components(
            prefill_missing=prefill_missing
        )
        if hasattr(self, "run_sequence"):
            runtime_state = self._prime_differentiable_runtime_outgoing(runtime_state)
        self._validate_differentiable_runtime(runtime_state)
        return runtime_state

    def append_masks_to_output(
        self,
        name: str,
        shared_fields: Shared,
    ) -> None:
        """
        Append binary and fractional masks to the output shared fields of component 'name'.

        Arguments:
            name: component name
            shared_fields: Shared instance containing fields to be written to output
        """

        for exchange in self.exchanges:
            if name != exchange.destination:
                continue

            key = (exchange.source, name, exchange.interpolation_type)
            source_destination_name = "_".join(key)

            shared_fields["bmask_" + source_destination_name] = (
                self._binary_masks[key],
                datetime.now(),
                name,
            )

            shared_fields["fmask_" + source_destination_name] = (
                self._fractional_masks[key],
                datetime.now(),
                name,
            )

    def interpolate_and_dispatch_fields(
        self,
        component: AllComponentsType,
        timestamp: datetime | ModelDateTime,
    ) -> None:
        """
        Interpolate and dispatch fields to the given component at the specified timestamp.

        Arguments:
            timestamp: current simulation (coupler's) time
            component: destination component instance to process exchanges for
        """

        for exchange in self.exchanges:
            # Ensure exchange for currently stepping component only
            if exchange.destination != component.name:
                continue

            self.logger.info(f" Exchange fields: {exchange.name}")

        runtime_state = self._runtime_state_from_components(prefill_missing=False)
        runtime_state = dispatch_component_exchanges(
            runtime_state,
            component.name,
            self.exchanges,
            self._regridders,
        )
        destination_state = runtime_state.get_component_state(component.name)
        destination_fields = Shared()

        for exchange in self.exchanges:
            if exchange.destination != component.name:
                continue
            for field_name in exchange.field_names:
                if isinstance(field_name, tuple):
                    setattr(
                        destination_fields,
                        field_name[0],
                        TNA(
                            destination_state.incoming.get(field_name[0]),
                            timestamp,
                            exchange.source,
                        ),
                    )
                    setattr(
                        destination_fields,
                        field_name[1],
                        TNA(
                            destination_state.incoming.get(field_name[1]),
                            timestamp,
                            exchange.source,
                        ),
                    )
                else:
                    setattr(
                        destination_fields,
                        field_name,
                        TNA(
                            destination_state.incoming.get(field_name),
                            timestamp,
                            exchange.source,
                        ),
                    )

        if not destination_fields.is_empty:
            component.import_fields(destination_fields)
            self.logger.debug(
                f" Exchanged {destination_fields.field_names}" f" to {component.name}"
            )

    def finalize(self, output_file_mask: Optional[Path] = None) -> None:
        """
        Finalize the coupler and all registered components.

        Arguments:
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for name, component in self.components.items():
            component.finalize(self, output_file_mask)
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

    def run(self) -> None:
        """
        Run the coupler and all registered components according to the run sequence.
        """

        # TODO: add setup checks like time step consistency,
        # component's readiness (outgoing fields), etc.
        # Wrap in a class method or function
        for cname in self.run_sequence:
            if self.components[cname].outgoing_fields.is_empty:
                raise ComponentError(
                    f"Component {cname} outgoing fields were not initialized properly."
                )

        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )

            # Step components in declared order
            for cname in self.run_sequence:
                self.interpolate_and_dispatch_fields(self.components[cname], time)

                self.logger.info(f" Run component: {cname}")
                self.components[cname].receive_fields(time)

                self.components[cname].step(dt, time, self)

                self.components[cname].send_fields(time, self)

    def run_differentiable(
        self, initial_state: RuntimeCouplerState | None = None
    ) -> RuntimeCouplerState:
        """Run the pure JAX differentiable runtime path and return the final state.

        Existing public component and coupler APIs remain imperative. This entrypoint
        provides a differentiable state path for VerCOR-owned slab and pure
        data-forcing components while external model adapters stay explicit
        host/runtime boundaries.
        """

        runtime_state = (
            self.create_differentiable_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        self._validate_differentiable_runtime(runtime_state)
        step_infos = self._build_differentiable_step_info()

        def step_all_components(
            state: RuntimeCouplerState, step_info: RuntimeStepInfo
        ) -> tuple[RuntimeCouplerState, None]:
            for cname in self.run_sequence:
                state = dispatch_component_exchanges(
                    state,
                    cname,
                    self.exchanges,
                    self._regridders,
                )
                component_state = state.get_component_state(cname)
                component_state = receive_component_fields(component_state)
                component_state = step_component_state(
                    self.components[cname],
                    component_state,
                    self.clock.dt_seconds,
                    self.settings,
                )
                component_state = send_component_fields(
                    component_state,
                    self.components[cname],
                    step_info,
                )
                state = state.set_component_state(component_state)
            return state, None

        final_state, _ = jax.lax.scan(
            step_all_components,
            runtime_state,
            step_infos,
            length=self.clock.steps,
        )
        return final_state
