import numpy as np

from vercor.components.base import Component
from vercor.grid import RectilinearGrid


# === jcm imports ===
import jax
import jax.numpy as jnp

import jax_datetime as jdt

from dinosaur import primitive_equations, primitive_equations_states

import tree_math
from jcm.model import Model
from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState
from vercor.components.JCM_tools import mean_leaf, stack_objects, unwrap_leading_dims

from datetime import timedelta

from dataclasses import dataclass
from typing import Any, Optional, List, cast

import xarray as xr

import vercor


from vercor.components.base import TimedNamedArray as TNA

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vercor.coupler import Coupler
# ===================

latent_heat_of_vaporization = vercor.settings.VercorSettings.latvap

def asfloat64(tree):
    return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float64), tree)


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations_states


class JCM(Component):
    """JCM Wrapper"""

    _predictions_list: List

    def __init__(
        self,
        name: str,
        model: Model,
        coupling_timestep: timedelta = timedelta(days=1),
        save_interval: timedelta = timedelta(hours=12),
        jitted: bool = True,
    ) -> None:
        self.model = model
        self.coupling_timestep = coupling_timestep
        self.save_interval = save_interval
        self.jitted = jitted

        hgrid = model.coords.horizontal
        grid = RectilinearGrid(
            name=name,
            longitude=np.array(hgrid.longitudes) * 180.0 / np.pi,
            latitude=np.array(hgrid.latitudes) * 180.0 / np.pi,
            binary_mask=np.where(model.geometry.fmask > 0.0, 1.0, 0.0).transpose() == 0.0,  # true = valid points.
            fraction_mask=model.geometry.fmask.transpose(),
        )

        super().__init__(name, grid)

    def _initialize(
        self,
        initial_state: Optional[PhysicsState | primitive_equations.State] = None,
        forcing: ForcingData = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
    ) -> JCMState:
        model = self.model

        # Copy from jax-gcm jcm/model.py
        if isinstance(initial_state, primitive_equations.State):
            model.initial_state = dynamics_state_to_physics_state(
                initial_state, model.primitive
            )
            model._final_modal_state = initial_state
        else:
            model.initial_state = initial_state
            model._final_modal_state = model._prepare_initial_modal_state(initial_state)

            if initial_state is None:
                model.initial_state = dynamics_state_to_physics_state(
                    model._final_modal_state, model.primitive
                )

        model.start_date = start_date
        model.forcing = forcing or default_forcing(self.model.coords.horizontal)

        # The following code is a solution to have an initial value for phydata by stepping the model one time.
        # The returned phydata is then used for the initial value.
        _, init_phydata = self.model.physics.compute_tendencies(
            state=model.initial_state,
            forcing=model.forcing,
            geometry=model.geometry,
            date=model._date_from_sim_time(
                jnp.array(model._final_modal_state.sim_time)
            ),
        )

        return JCMState(
            prog=asfloat64(model.initial_state),
            phydata=asfloat64(init_phydata),
            metadata=model._final_modal_state,
        )

    def _generate_step_function(self, jitted: bool = True):
        def step_function(state, forcing, t):
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                forcing=forcing,
            )

            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            return JCMState(
                prog=asfloat64(mean_leaf(predictions.dynamics, axis=0)),
                phydata=asfloat64(mean_leaf(predictions.physics, axis=0)),
                metadata=new_atm_modal_state,
            ), predictions

        return jax.jit(step_function) if jitted else step_function

    def initialize(self, coupler: "Coupler") -> None:
        self._state = self._initialize()
        self._step_function = self._generate_step_function(jitted=self.jitted)

        clock_start = coupler.clock.start
        grid_shape = self.grid.shape

        zeros = np.zeros(grid_shape)

        self.incoming_fields.SST = TNA(zeros + 273.15 + 15.0, clock_start, self.name)
        self.outgoing_fields.bottom_level_density = TNA(zeros + 1.22, clock_start, self.name)
        self.outgoing_fields.bottom_level_zonal_velocity = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_meridional_velocity = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_potential_temperature = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_temperature = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_specific_humidity = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_height = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.net_surface_shortwave_radiation_flux = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.net_surface_longwave_radiation_flux = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.bottom_level_ = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.precipitation = TNA(zeros, clock_start, self.name)
        self.outgoing_fields.evaporation = TNA(zeros, clock_start, self.name)

        self._predictions_list = []

    def step(self, dt, time, coupler) -> None:
        N = dt / self.coupling_timestep
        if N % 1 != 0:
            raise ValueError(
                f"dt={str(dt)} must be an integer multiple of coupling_timestep={str(self.coupling_timestep)}."
            )

        _forcing = self.model.forcing.copy(
            sea_surface_temperature=jnp.asarray(self.incoming_fields.SST).transpose(),
        )

        _avg_predictions = []
        N = int(N)
        for _ in range(N):
            _new_state, _predictions = self._step_function(
                self._state,
                _forcing,
                jdt.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            self._state = _new_state
            _avg_predictions.append(_predictions)
            self._predictions_list.append(_predictions)

        _avg_predictions = mean_leaf(
            unwrap_leading_dims(stack_objects(_avg_predictions)), axis=0
        )
        p = _avg_predictions.physics
        d = _avg_predictions.dynamics

        # All the heat and freshwater fluxes are positive upward
        self.outgoing_fields.net_surface_shortwave_radiation_flux = (- np.array(p.shortwave_rad.rsns).sum(axis=2).transpose(), time, self.name)
        self.outgoing_fields.net_surface_longwave_radiation_flux = (np.array(p.surface_flux.rlns).sum(axis=2).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_zonal_velocity = (np.array(d.u_wind[:, :, -1]).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_meridional_velocity = (np.array(d.v_wind[:, :, -1]).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_potential_temperature = (np.array(d.temperature[:, :, -1]).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_temperature = (np.array(d.temperature[:, :, -1]).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_specific_humidity = (np.array(d.specific_humidity[:, :, -1]).transpose(), time, self.name)
        self.outgoing_fields.bottom_level_height = (np.array(d.geopotential[:, :, -1] / jcm.constants.grav).transpose(), time, self.name)
        self.outgoing_fields.precipitation = ((
            -np.array(p.condensation.precls + p.convection.precnv).transpose() / 1e3
        ), time, self.name)
        self.outgoing_fields.evaporation = ((
            np.array(p.surface_flux.evap / 1e3).sum(axis=2).transpose()
        ), time, self.name)

    def _finalize(self, output: Optional[str] = None) -> xr.Dataset:
        # Current JCM returns an Any but is actually an xr.Dataset
        ds = cast(
            xr.Dataset,
            self.model.predictions_to_xarray(
                unwrap_leading_dims(stack_objects(self._predictions_list))
            ),
        )
        if output is not None:
            print(f"Output file: {output:s}")
            ds.to_netcdf(output, engine="netcdf4")

        return ds
