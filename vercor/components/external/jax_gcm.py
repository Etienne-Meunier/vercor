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
from vercor.grid import RectilinearGrid
from vercor.tools import _runtime_array_to_host
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import RuntimeComponentState


try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with `pip install jcm`."
    )


def asfloat(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float_), tree)


_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
_COLD_SURFACE_TEMPERATURE_THRESHOLD = 250.0


@jax.jit
def _cleanup_surface_temperature_fields(
    land_surface_temperature: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    land_surface_temperature_array = jnp.nan_to_num(
        jnp.asarray(land_surface_temperature, dtype=jnp.float_)
    )
    sea_surface_temperature_array = jnp.nan_to_num(
        jnp.asarray(sea_surface_temperature, dtype=jnp.float_)
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
    total_surface_temperature_array = jnp.asarray(
        total_surface_temperature, dtype=jnp.float_
    )
    land_fraction_mask_array = jnp.asarray(land_fraction_mask, dtype=jnp.float_)

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
    u_velocity = jnp.asarray(u_wind, dtype=jnp.float_)[-1, :, :].T
    v_velocity = jnp.asarray(v_wind, dtype=jnp.float_)[-1, :, :].T
    temperature_2m = jnp.asarray(temperature, dtype=jnp.float_)[-1, :, :].T
    specific_humidity_2m = (
        jnp.asarray(specific_humidity, dtype=jnp.float_)[-1, :, :].T / 1000.0
    )

    sensible_heat_flux = -jnp.sum(
        jnp.asarray(surface_sensible_heat_flux, dtype=jnp.float_), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        jnp.asarray(surface_evaporation, dtype=jnp.float_) / 1e3 * latvap,
        axis=2,
    ).T
    net_shortwave_radiation_flux_2m = jnp.asarray(
        net_shortwave_radiation_flux, dtype=jnp.float_
    ).T
    downward_longwave_radiation_flux_2m = jnp.asarray(
        downward_longwave_radiation_flux, dtype=jnp.float_
    ).T

    pressure = compute_pressure_levels(
        jnp.asarray(reference_pressure, dtype=jnp.float_),
        jnp.asarray(0.0, dtype=jnp.float_),
        jnp.asarray(sigma_levels, dtype=jnp.float_),
        jnp.asarray(normalized_surface_pressure, dtype=jnp.float_).T,
    )

    density = (
        jnp.asarray(mwdair, dtype=jnp.float_)
        / jnp.asarray(rgas, dtype=jnp.float_)
        * pressure[-1, ...]
        / temperature_2m
    )
    potential_temperature = temperature_2m * (
        jnp.asarray(potential_temperature_reference_pressure, dtype=jnp.float_)
        / pressure[-1, ...]
    ) ** jnp.asarray(cappa, dtype=jnp.float_)

    model_level_height = get_altitudes_sigma_levels(
        jnp.asarray(temperature, dtype=jnp.float_).transpose((0, 2, 1))[::-1, :, :],
        pressure[::-1, :, :],
        jnp.asarray(specific_humidity, dtype=jnp.float_).transpose((0, 2, 1))[
            ::-1, :, :
        ]
        / 1000.0,
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

    def do_jcm_steps(self) -> tuple[Any, Any]:
        _avg_predictions = []
        _predictions: Predictions

        _new_state, _predictions = self._step_function(
            self._state,
            self.forcing,
        )

        self._state = _new_state

        _avg_predictions.append(_predictions)

        self._predictions_list.append(_predictions)

        _avg_predictions = mean_leaf(
            unwrap_leading_dims(stack_objects(_avg_predictions)), axis=0
        )

        return _avg_predictions.physics, _avg_predictions.dynamics

    def initialize(self, coupler: "Coupler") -> None:
        self.coupling_timestep = timedelta(seconds=coupler.clock.dt_seconds)
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

        grid_shape = self.grid.shape

        zeros = jnp.zeros(grid_shape, dtype=jnp.float_)
        self.data["specific_humidity"] = zeros
        self.data["net_shortwave_radiation_flux"] = zeros
        self.data["downward_longwave_radiation_flux"] = zeros
        self.data["sea_surface_temperature"] = jnp.full(
            grid_shape, _REFERENCE_SURFACE_TEMPERATURE, dtype=jnp.float_
        )
        self.data["land_surface_temperature"] = zeros
        self.data["u_velocity"] = zeros
        self.data["v_velocity"] = zeros
        self.data["temperature"] = zeros
        self.data["potential_temperature"] = zeros
        self.data["density"] = zeros
        self.data["latent_heat_flux"] = zeros
        self.data["sensible_heat_flux"] = zeros
        self.data["model_level_height"] = zeros

        self._predictions_list = []

        if self.do_spinup and "OCN" in coupler.run_sequence.order:
            coupler.logger.info(
                f" Performing JCM spinup for {self.spinup_time} day(s)..."
            )
            # Spin-up from the default JCM forcing
            for i in range(self.spinup_steps):
                coupler.logger.info(f" JCM spinup step {i+1} / {self.spinup_steps}")
                _, _ = self.do_jcm_steps()

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
    ) -> None:
        """Pre-seed JAXGCM output fields so scan carry structure is stable."""

        zeros = jnp.zeros(self.grid.shape, dtype=jnp.float_)
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
        sigma_levels = jnp.asarray(self.sigma_levels)
        data.setdefault(
            "pressure",
            jnp.zeros((sigma_levels.shape[0], *self.grid.shape), dtype=jnp.float_),
        )
        super().prefill_runtime_state_fields(data, incoming, outgoing)

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate JAXGCM runtime payload and pre-seeded output fields."""

        if not isinstance(component_state.runtime_payload, JAXGCMRuntimePayload):
            raise ComponentError(
                "JAXGCM runtime requires an initialized immutable runtime payload "
                f"for component '{self.name}'"
            )

        grid_fields = (
            "land_surface_temperature",
            "sea_surface_temperature",
            "total_surface_temperature",
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
        for field_name in grid_fields:
            self._validate_runtime_grid_data_field(
                component_state,
                field_name,
                expected_shape,
            )

        self._validate_runtime_data_field_exists(component_state, "pressure")
        pressure_shape = jnp.asarray(component_state.data.get("pressure")).shape
        sigma_levels = jnp.asarray(self.sigma_levels)
        expected_pressure_shape = (sigma_levels.shape[0], *expected_shape)
        if pressure_shape != expected_pressure_shape:
            raise CouplerError(
                "Runtime required data field 'pressure' "
                f"for component '{self.name}' has shape {pressure_shape}, "
                f"expected {expected_pressure_shape}"
            )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: ModelDateTime | datetime | None = None,
        coupler: "Coupler | None" = None,
    ) -> "RuntimeComponentState":
        """Advance JAXGCM on immutable runtime state."""

        _ = dt_seconds, time, coupler
        if runtime_settings is None:
            raise NotImplementedError("JAXGCM runtime settings are not initialized")

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
                jnp.asarray(self.model.terrain.fmask, dtype=jnp.float_).T,
            )
        )
        forcing = payload.forcing.copy(
            stl_am=land_surface_temperature_forcing.T,
            sea_surface_temperature=sea_surface_temperature_forcing.T,
        )
        jcm_state, prediction = self._step_function(payload.jcm_state, forcing)
        averaged_prediction = mean_leaf(
            unwrap_leading_dims(stack_objects([prediction])), axis=0
        )

        mapped_fields = _map_jcm_output_fields(
            runtime_settings.latvap,
            p0,
            self.sigma_levels,
            runtime_settings.mwdair,
            runtime_settings.rgas,
            runtime_settings.p0,
            runtime_settings.cappa,
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

        data = data.set("land_surface_temperature", land_surface_temperature)
        data = data.set("sea_surface_temperature", sea_surface_temperature)
        data = data.set("total_surface_temperature", total_surface_temperature)
        for field_name, field_value in mapped_fields.items():
            data = data.set(field_name, field_value)

        return component_state.with_data(data).with_runtime_payload(
            JAXGCMRuntimePayload(jcm_state=jcm_state, forcing=forcing)
        )

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: "Coupler",
    ) -> None:
        settings = coupler.settings

        logger = coupler.logger

        logger.info(
            " Mean of SST: {}".format(
                float(jnp.nanmean(jnp.asarray(self.data["sea_surface_temperature"])))
            )
        )

        (
            land_surface_temperature,
            sea_surface_temperature,
            total_surface_temperature,
            cold_surface_cells,
        ) = _cleanup_surface_temperature_fields(
            self.data["land_surface_temperature"],
            self.data["sea_surface_temperature"],
        )

        self.data["land_surface_temperature"] = land_surface_temperature
        self.data["sea_surface_temperature"] = sea_surface_temperature
        self.data["total_surface_temperature"] = total_surface_temperature

        logger.info(
            " Number of cells with (SST + SKT) less than 250.0 K: {}".format(
                int(jnp.sum(cold_surface_cells))
            ),
        )

        land_surface_temperature_forcing, sea_surface_temperature_forcing = (
            _prepare_surface_temperature_forcing(
                total_surface_temperature,
                jnp.asarray(self.model.terrain.fmask, dtype=jnp.float_).T,
            )
        )

        self.forcing = self.forcing.copy(
            stl_am=_runtime_array_to_host(land_surface_temperature_forcing).T,
            sea_surface_temperature=_runtime_array_to_host(
                sea_surface_temperature_forcing
            ).T,
        )

        p, d = self.do_jcm_steps()

        mapped_fields = _map_jcm_output_fields(
            settings.latvap,
            p0,
            self.sigma_levels,
            settings.mwdair,
            settings.rgas,
            settings.p0,
            settings.cappa,
            p.surface_flux.shf,
            p.surface_flux.evap,
            p.surface_flux.rlds,
            p.shortwave_rad.rsns,
            d.normalized_surface_pressure,
            d.u_wind,
            d.v_wind,
            d.temperature,
            d.specific_humidity,
        )

        self.data.update(mapped_fields)

        if self._should_write_output(time=time, dt=dt):
            date_time = time.strftime("%Y-%m-%d")
            self._write_output(output=f"jcm.averages.{date_time}.nc")

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

    def _write_output(self, output: str) -> None:
        ds = cast(
            xr.Dataset,
            xr.merge(
                [_prediction.to_xarray() for _prediction in self._predictions_list]
            ),
        )

        print(f"Output file: {output:s}")

        t_end = ds.time.isel(time=-1)
        ds.mean(dim="time", keep_attrs=True, keepdims=True).assign_coords(
            time=[t_end.values]
        ).transpose("time", "wvi_id", "hsg_level", "level", "lat", "lon").to_netcdf(
            output, engine="h5netcdf"
        )

        # Clear the predictions list after saving to disk to free up memory
        self._predictions_list = []
