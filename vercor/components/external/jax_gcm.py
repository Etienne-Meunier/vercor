from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Literal, cast

import jax
import jax.numpy as jnp
import tree_math
import xarray as xr

from dinosaur import primitive_equations
from dinosaur.coordinate_systems import CoordinateSystem

from jcm.constants import p0
from jcm.forcing import default_forcing
from jcm.model import ForcingData, Model, Predictions
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics_interface import (
    PhysicsState,
    TerrainData,
    dynamics_state_to_physics_state,
)

from vercor.clock import ModelDateTime
from vercor.components.base import Component
from vercor.exceptions import ComponentError, CouplerError
from vercor.components.external.jax_gcm_tools import (
    change_jcm_parameter_values,
    mean_leaf,
    stack_objects,
    unwrap_leading_dims,
    get_altitudes_sigma_levels,
    compute_pressure_levels,
)
from vercor.dtypes import (
    PrecisionPolicy,
    as_jax_real_array,
    jax_full,
    jax_real_dtype,
    jax_zeros,
)
from vercor.grid import RectilinearGrid
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.runtime.components import (
    validate_runtime_data_field_exists,
    validate_runtime_grid_data_field,
)
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentContract, RuntimeComponentState


try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with `pip install jcm`."
    )


def asfloat(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda arr: arr.astype(jax_real_dtype()), tree)


_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
_COLD_SURFACE_TEMPERATURE_THRESHOLD = 250.0
_JAXGCM_OUTPUT_GRID_FIELD_NAMES = (
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
)
_JAXGCM_REQUIRED_GRID_FIELD_NAMES = (
    "land_surface_temperature",
    "sea_surface_temperature",
    "total_surface_temperature",
    *_JAXGCM_OUTPUT_GRID_FIELD_NAMES,
)


def _default_jax_gcm_grid_fields(
    grid_shape: tuple[int, int],
    *,
    include_total_surface_temperature: bool,
    policy: PrecisionPolicy = None,
) -> dict[str, RuntimeArray]:
    """Return default grid-shaped JAXGCM runtime fields."""

    zeros = jax_zeros(grid_shape, policy)
    fields: dict[str, RuntimeArray] = {
        field_name: zeros for field_name in _JAXGCM_OUTPUT_GRID_FIELD_NAMES
    }
    fields["land_surface_temperature"] = zeros
    fields["sea_surface_temperature"] = jax_full(
        grid_shape,
        _REFERENCE_SURFACE_TEMPERATURE,
        policy,
    )
    if include_total_surface_temperature:
        fields["total_surface_temperature"] = zeros
    return fields


@jax.jit
def _cleanup_surface_temperature_fields(
    land_surface_temperature: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    land_surface_temperature_array = jnp.nan_to_num(
        as_jax_real_array(land_surface_temperature)
    )
    sea_surface_temperature_array = jnp.nan_to_num(
        as_jax_real_array(sea_surface_temperature)
    )
    total_surface_temperature = (
        land_surface_temperature_array + sea_surface_temperature_array
    )
    cold_surface_cells = total_surface_temperature < _COLD_SURFACE_TEMPERATURE_THRESHOLD
    return (
        land_surface_temperature_array,
        sea_surface_temperature_array,
        total_surface_temperature,
        cold_surface_cells,
    )


@jax.jit
def _prepare_surface_temperature_forcing(
    total_surface_temperature: object,
    land_fraction_mask: object,
) -> tuple[jax.Array, jax.Array]:
    total_surface_temperature_array = as_jax_real_array(total_surface_temperature)
    land_fraction_mask_array = as_jax_real_array(land_fraction_mask)

    land_surface_temperature = (
        total_surface_temperature_array * land_fraction_mask_array
    )
    sea_surface_temperature = total_surface_temperature_array * (
        1.0 - land_fraction_mask_array
    )

    land_surface_temperature = jnp.where(
        land_surface_temperature == 0.0,
        _REFERENCE_SURFACE_TEMPERATURE,
        land_surface_temperature,
    )
    sea_surface_temperature = jnp.where(
        sea_surface_temperature == 0.0,
        _REFERENCE_SURFACE_TEMPERATURE,
        sea_surface_temperature,
    )

    return land_surface_temperature, sea_surface_temperature


@jax.jit
def _map_jcm_output_fields(
    latvap: float,
    reference_pressure: float,
    sigma_levels: object,
    mwdair: float,
    rgas: float,
    potential_temperature_reference_pressure: float,
    cappa: float,
    surface_sensible_heat_flux: object,
    surface_evaporation: object,
    downward_longwave_radiation_flux: object,
    net_shortwave_radiation_flux: object,
    normalized_surface_pressure: object,
    u_wind: object,
    v_wind: object,
    temperature: object,
    specific_humidity: object,
) -> dict[str, jax.Array]:
    u_velocity = as_jax_real_array(u_wind)[-1, :, :].T
    v_velocity = as_jax_real_array(v_wind)[-1, :, :].T
    temperature_2m = as_jax_real_array(temperature)[-1, :, :].T
    specific_humidity_2m = as_jax_real_array(specific_humidity)[-1, :, :].T / 1000.0

    sensible_heat_flux = -jnp.sum(
        as_jax_real_array(surface_sensible_heat_flux), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        as_jax_real_array(surface_evaporation) / 1e3 * latvap,
        axis=2,
    ).T
    net_shortwave_radiation_flux_2m = as_jax_real_array(net_shortwave_radiation_flux).T
    downward_longwave_radiation_flux_2m = as_jax_real_array(
        downward_longwave_radiation_flux
    ).T

    pressure = compute_pressure_levels(
        as_jax_real_array(reference_pressure),
        as_jax_real_array(0.0),
        as_jax_real_array(sigma_levels),
        as_jax_real_array(normalized_surface_pressure).T,
    )

    density = (
        as_jax_real_array(mwdair)
        / as_jax_real_array(rgas)
        * pressure[-1, ...]
        / temperature_2m
    )
    potential_temperature = temperature_2m * (
        as_jax_real_array(potential_temperature_reference_pressure) / pressure[-1, ...]
    ) ** as_jax_real_array(cappa)

    model_level_height = get_altitudes_sigma_levels(
        as_jax_real_array(temperature).transpose((0, 2, 1))[::-1, :, :],
        pressure[::-1, :, :],
        as_jax_real_array(specific_humidity).transpose((0, 2, 1))[::-1, :, :] / 1000.0,
    )[1, :, :]

    return {
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "temperature": temperature_2m,
        "specific_humidity": specific_humidity_2m,
        "sensible_heat_flux": sensible_heat_flux,
        "latent_heat_flux": latent_heat_flux,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux_2m,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux_2m,
        "pressure": pressure,
        "density": density,
        "potential_temperature": potential_temperature,
        "model_level_height": model_level_height,
    }


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JAXGCMRuntimePayload:
    """Immutable JAXGCM model state carried by runtime component state."""

    jcm_state: Any
    forcing: Any

    def tree_flatten(self) -> tuple[tuple[Any, Any], None]:
        return (self.jcm_state, self.forcing), None

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Any, Any]
    ) -> "JAXGCMRuntimePayload":
        _ = aux_data
        jcm_state, forcing = children
        return cls(jcm_state=jcm_state, forcing=forcing)


class JAXGCM(Component):
    """JCM Wrapper"""

    _predictions_list: list[Predictions]
    _step_function: Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]
    _state: JCMState
    forcing: ForcingData

    def __init__(
        self,
        coords: CoordinateSystem,
        terrain: TerrainData,
        name: str = "ATM",
        custom_parameters: Optional[dict[str, float]] = None,
        model_timestep: timedelta = timedelta(minutes=30),
        save_interval: timedelta = timedelta(days=1),
        spinup_time: timedelta = timedelta(days=2),
        forcing_data: Optional[ForcingData] = None,
        # Output frequency in days for saving JCM predictions.
        output_frequency: Optional[str] = None,
        do_spinup: bool = False,
        jitted: bool = True,
    ) -> None:

        self.forcing_data = forcing_data
        self.output_frequency = output_frequency
        self.model_timestep = model_timestep
        self.save_interval = save_interval
        self.spinup_time = spinup_time
        self.do_spinup = do_spinup
        self.jitted = jitted

        jcm_parameters = Parameters.default()

        if custom_parameters is not None:
            change_jcm_parameter_values(
                parameters=custom_parameters,
                default_parameters=jcm_parameters,
            )

        physics = SpeedyPhysics(parameters=jcm_parameters)

        self.model = Model(
            coords,
            time_step=model_timestep.total_seconds() / 60.0,
            terrain=terrain,
            physics=physics,
        )

        hgrid = self.model.coords.horizontal
        grid = RectilinearGrid(
            name=name,
            longitude=jnp.rad2deg(jnp.asarray(hgrid.longitudes)),
            latitude=jnp.rad2deg(jnp.asarray(hgrid.latitudes)),
            binary_mask=jnp.ones_like(
                jnp.asarray(self.model.terrain.fmask)
            ).transpose(),  # This is used for interpolation, which all points are valid
        )

        self.sigma_levels: RuntimeArray = self.model.coords.vertical.centers

        super().__init__(name, grid)
        self.declare_fields(
            inputs=("land_surface_temperature", "sea_surface_temperature"),
            outputs=(
                "land_surface_temperature",
                "sea_surface_temperature",
                "total_surface_temperature",
                *_JAXGCM_OUTPUT_GRID_FIELD_NAMES,
                "pressure",
            ),
            default_fields=_default_jax_gcm_grid_fields(
                self.grid.shape,
                include_total_surface_temperature=True,
                policy=self.settings,
            ),
        )

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]:
        def step_function(
            state: JCMState, forcing: ForcingData
        ) -> tuple[JCMState, Predictions]:
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                forcing=forcing,
            )

            # phydata is a stacked object, so I take the mean here.
            # However, this action will be done by jcm in the new jcm PR.
            return (
                JCMState(
                    prog=asfloat(mean_leaf(predictions.dynamics, axis=0)),
                    phydata=asfloat(mean_leaf(predictions.physics, axis=0)),
                    metadata=new_atm_modal_state,
                ),
                predictions,
            )

        return jax.jit(step_function) if jitted else step_function

    def initialize(self, context: ComponentInitContext) -> None:
        self.coupling_timestep = timedelta(seconds=context.dt_seconds)
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        if self.coupling_timestep % self.model_timestep != timedelta(days=0):
            raise ValueError(
                f"model_timestep ({self.model_timestep}) must be a "
                f"multiple of coupling_timestep ({self.coupling_timestep})"
            )

        _modal_state = self.model._prepare_initial_modal_state()
        self._state = JCMState(
            metadata=_modal_state,
            phydata=PhysicsData.zeros(
                self.model.coords.horizontal.nodal_shape,
                self.model.coords.vertical.layers,
            ),
            prog=dynamics_state_to_physics_state(_modal_state, self.model.primitive),
        )

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal).copy(
                lfluxland=True
            )

        self._step_function = self._generate_step_function(jitted=self.jitted)

        self.seed_fields(
            _default_jax_gcm_grid_fields(
                self.grid.shape,
                include_total_surface_temperature=False,
                policy=context.settings,
            )
        )

        self._predictions_list = []

        if self.do_spinup and "OCN" in context.run_sequence.order:
            context.logger.info(
                f" Performing JCM spinup for {self.spinup_time} day(s)..."
            )
            # Spin-up from the default JCM forcing
            for i in range(self.spinup_steps):
                context.logger.info(f" JCM spinup step {i+1} / {self.spinup_steps}")
                _new_state, _predictions = self._step_function(
                    self._state,
                    self.forcing,
                )
                self._state = _new_state
                self._predictions_list.append(_predictions)

    def create_runtime_payload(self) -> JAXGCMRuntimePayload:
        """Return immutable JCM state and forcing for runtime execution."""

        missing = [
            name
            for name in ("_state", "forcing", "_step_function")
            if not hasattr(self, name)
        ]
        if missing:
            missing_names = ", ".join(missing)
            raise ComponentError(
                "JAXGCM runtime requires component initialization before "
                f"state creation; missing {missing_names}"
            )

        return JAXGCMRuntimePayload(
            jcm_state=self._state,
            forcing=self.forcing,
        )

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: "RuntimeComponentContract",
    ) -> None:
        """Pre-seed JAXGCM output fields so scan carry structure is stable."""

        self.prefill_runtime_fields(
            data,
            default_fields=_default_jax_gcm_grid_fields(
                self.grid.shape,
                include_total_surface_temperature=True,
                policy=self.settings,
            ),
        )
        sigma_levels = jnp.asarray(self.sigma_levels)
        data.setdefault(
            "pressure",
            jax_zeros((sigma_levels.shape[0], *self.grid.shape)),
        )
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Validate JAXGCM runtime payload and pre-seeded output fields."""

        _ = contract
        if not isinstance(component_state.runtime_payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "JAXGCM runtime requires an initialized immutable runtime payload "
                f"for component '{self.name}'"
            )

        for field_name in _JAXGCM_REQUIRED_GRID_FIELD_NAMES:
            validate_runtime_grid_data_field(
                self,
                component_state,
                field_name,
            )

        validate_runtime_data_field_exists(self, component_state, "pressure")
        pressure_shape = jnp.asarray(component_state.data.get("pressure")).shape
        sigma_levels = jnp.asarray(self.sigma_levels)
        expected_pressure_shape = (sigma_levels.shape[0], *self.grid.shape)
        if pressure_shape != expected_pressure_shape:
            raise CouplerError(
                "Runtime required data field 'pressure' "
                f"for component '{self.name}' has shape {pressure_shape}, "
                f"expected {expected_pressure_shape}"
            )

    def _step_jax_gcm_component_state(
        self,
        component_state: "RuntimeComponentState",
        settings: Any,
    ) -> tuple["RuntimeComponentState", Predictions, Any]:
        """Advance JAXGCM runtime state and return the raw prediction."""

        payload = component_state.runtime_payload
        if not isinstance(payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "JAXGCM runtime requires an initialized immutable runtime payload "
                f"for component '{self.name}'"
            )

        data = component_state.data
        (
            land_surface_temperature,
            sea_surface_temperature,
            total_surface_temperature,
            _,
        ) = _cleanup_surface_temperature_fields(
            data.get("land_surface_temperature"),
            data.get("sea_surface_temperature"),
        )

        land_surface_temperature_forcing, sea_surface_temperature_forcing = (
            _prepare_surface_temperature_forcing(
                total_surface_temperature,
                as_jax_real_array(self.model.terrain.fmask, settings).T,
            )
        )
        applied_forcing = payload.forcing.copy(
            stl_am=land_surface_temperature_forcing.T,
            sea_surface_temperature=sea_surface_temperature_forcing.T,
        )
        jcm_state, prediction = self._step_function(
            payload.jcm_state,
            applied_forcing,
        )
        averaged_prediction = mean_leaf(
            unwrap_leading_dims(stack_objects([prediction])), axis=0
        )

        mapped_fields = _map_jcm_output_fields(
            settings.latvap,
            p0,
            self.sigma_levels,
            settings.mwdair,
            settings.rgas,
            settings.p0,
            settings.cappa,
            averaged_prediction.physics.surface_flux.shf,
            averaged_prediction.physics.surface_flux.evap,
            averaged_prediction.physics.surface_flux.rlds,
            averaged_prediction.physics.shortwave_rad.rsns,
            averaged_prediction.dynamics.normalized_surface_pressure,
            averaged_prediction.dynamics.u_wind,
            averaged_prediction.dynamics.v_wind,
            averaged_prediction.dynamics.temperature,
            averaged_prediction.dynamics.specific_humidity,
        )

        updated_state = self.with_runtime_fields(
            component_state,
            {
                "land_surface_temperature": land_surface_temperature,
                "sea_surface_temperature": sea_surface_temperature,
                "total_surface_temperature": total_surface_temperature,
                **mapped_fields,
            },
        )

        return (
            updated_state.with_runtime_payload(
                JAXGCMRuntimePayload(jcm_state=jcm_state, forcing=payload.forcing)
            ),
            prediction,
            applied_forcing,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance JAXGCM on immutable runtime state."""

        time = context.time
        logger = context.logger
        if logger is not None:
            logger.info(
                " Mean of SST: {}",
                jnp.nanmean(
                    jnp.asarray(component_state.data.get("sea_surface_temperature"))
                ),
            )

        (
            stepped_state,
            prediction,
            applied_forcing,
        ) = self._step_jax_gcm_component_state(
            component_state,
            context.settings,
        )

        if time is None:
            return stepped_state

        payload = stepped_state.runtime_payload
        if isinstance(payload, JAXGCMRuntimePayload):
            self._state = payload.jcm_state
            self.forcing = applied_forcing
        self._predictions_list.append(prediction)

        _, _, _, cold_surface_cells = _cleanup_surface_temperature_fields(
            stepped_state.data.get("land_surface_temperature"),
            stepped_state.data.get("sea_surface_temperature"),
        )
        if logger is not None:
            logger.info(
                " Number of cells with (SST + SKT) less than 250.0 K: {}",
                jnp.sum(cold_surface_cells),
            )

        if self._should_write_output(
            time=time,
            dt=timedelta(seconds=context.dt_seconds),
        ):
            date_time = time.strftime("%Y-%m-%d")
            self._write_output(
                output=f"jcm.averages.{date_time}.nc",
                logger=logger,
            )

        return stepped_state

    def _is_period_end(
        self,
        time: datetime | ModelDateTime,
        dt: timedelta,
        frequency: Literal["day", "month", "year"],
    ) -> bool:
        next_time = time + dt

        if frequency == "day":
            return (
                next_time.year != time.year
                or next_time.month != time.month
                or next_time.day != time.day
            )
        if frequency == "month":
            return next_time.year != time.year or next_time.month != time.month

        return next_time.year != time.year

    def _should_write_output(
        self,
        time: datetime | ModelDateTime,
        dt: timedelta,
    ) -> bool:
        if self.output_frequency is None:
            return True

        if not isinstance(self.output_frequency, str):
            return False

        frequency = self.output_frequency.lower()
        if frequency not in ("day", "month", "year"):
            return False

        return self._is_period_end(
            time=time,
            dt=dt,
            frequency=cast(Literal["day", "month", "year"], frequency),
        )

    def _write_output(
        self,
        output: str,
        logger: LoggerLike | None = None,
    ) -> None:
        ds = cast(
            xr.Dataset,
            xr.merge(
                [_prediction.to_xarray() for _prediction in self._predictions_list]
            ),
        )

        log = logger if logger is not None else get_default_logger()
        log.info(f"Output file: {output:s}")

        t_end = ds.time.isel(time=-1)
        ds.mean(dim="time", keep_attrs=True, keepdims=True).assign_coords(
            time=[t_end.values]
        ).transpose("time", "wvi_id", "hsg_level", "level", "lat", "lon").to_netcdf(
            output, engine="h5netcdf"
        )

        # Clear the predictions list after saving to disk to free up memory
        self._predictions_list = []
