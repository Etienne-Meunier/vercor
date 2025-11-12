import numpy as np

from vercor.components.base import Component
from vercor.grid import RectilinearGrid


# === jcm imports ===
import jax
import jax.numpy as jnp
from jax import tree_util

import jax_datetime as jdt

import dinosaur
from dinosaur import primitive_equations, primitive_equations_states

import tree_math
import jcm
from jcm.forcing import ForcingData, default_forcing
from jcm.physics_interface import dynamics_state_to_physics_state, physics_state_to_dynamics_state
from jcm.physics_interface import PhysicsState
from jcm.model import Predictions
from vercor.components.JCM_tools import mean_leaf, stack_objects, unwrap_leading_dims

from datetime import timedelta

from dataclasses import dataclass
from typing import Any, Optional

import xarray as xr
# ===================

evaporation_enthalpy_of_water = 2.257e6 # J / kg


def asfloat64(tree):
    return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float64), tree)

@tree_math.struct
@dataclass
class JCMState:
    prog    : PhysicsState
    phydata : Any
    metadata    : primitive_equations_states

class JCM(Component):
    """JCM Wrapper
    """

    def __init__(
        self,
        name: str,
        model: jcm.model.Model,
        coupling_timestep: timedelta = timedelta(days=1),
        save_interval: float = timedelta(hours=12),
        jitted: bool = True,
    ) -> None:

        self.model = model
        self.coupling_timestep = coupling_timestep
        self.save_interval     = save_interval
        self.jitted = jitted
        
        hgrid = model.coords.horizontal
        grid = RectilinearGrid(
            name = name,
            longitude = np.array(hgrid.longitudes) * 180.0/np.pi,
            latitude = np.array(hgrid.latitudes) * 180.0/np.pi,
            mask = np.where(model.geometry.fmask > 0.0, 1.0, 0.0).transpose() == 0.0,  # true = valid points.
        )

        super().__init__(name, grid)

    def _initialize(
        self,
        initial_state: PhysicsState | primitive_equations.State = None,
        forcing: ForcingData = None,
        start_date: jdt.Datetime = jdt.to_datetime("2000-01-01"),
    ) -> JCMState:

        model = self.model
        
        # Copy from jax-gcm jcm/model.py
        if isinstance(initial_state, primitive_equations.State):
            model.initial_state = dynamics_state_to_physics_state(initial_state, model.primitive)
            model._final_modal_state = initial_state
        else:
            model.initial_state = initial_state
            model._final_modal_state = model._prepare_initial_modal_state(initial_state)

            if initial_state is None:
                model.initial_state = dynamics_state_to_physics_state(model._final_modal_state, model.primitive)
            
        model.start_date = start_date
        model.forcing = forcing or default_forcing(self.model.coords.horizontal)

        # The following code is a solution to have an initial value for phydata by stepping the model one time.
        # The returned phydata is then used for the initial value.
        _, init_phydata = self.model.physics.compute_tendencies(
            state       = model.initial_state,
            forcing     = model.forcing,
            geometry    = model.geometry,
            date        = model._date_from_sim_time(jnp.array(model._final_modal_state.sim_time)),
        )

        return JCMState(
            prog     = asfloat64(model.initial_state),
            phydata  = asfloat64(init_phydata),
            metadata = model._final_modal_state,
        )

    def _generate_step_function(self, jitted: bool = True):
       
        
        def step_function(state, forcing, t):
          
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state = state.metadata,
                save_interval = self.save_interval   / timedelta(days=1),
                total_time = self.coupling_timestep  / timedelta(days=1),
                forcing = forcing,
            )
            
            # phydata is a stacked object, so I take the mean here.
            # Howwever, this action will be done by jcm in the new jcm PR.
            return JCMState(
                prog    = asfloat64(mean_leaf(predictions.dynamics, axis=0)),
                phydata = asfloat64(mean_leaf(predictions.physics, axis=0)),
                metadata = new_atm_modal_state,
            ), predictions
            
        return jax.jit(step_function) if jitted else step_function

    def initialize(self, coupler) -> None:

        self._state = self._initialize()
        self._step_function = self._generate_step_function(jitted=self.jitted)
        
        ny, nx = self.grid.shape
        self.state = {}
        self.state["SST"] = np.zeros((ny, nx)) + 273.15 + 15.0
        self.state["SHF"] = np.zeros((ny, nx))
        self.state["LHF"] = np.zeros((ny, nx))
        self.state["precipitation"] = np.zeros((ny, nx))
        self.state["evaporation"] = np.zeros((ny, nx))
        self.state["u10m"] = np.zeros((ny, nx))
        self.state["v10m"] = np.zeros((ny, nx))
        
        self._predictions_list = []
        

    def step(self, dt, time, coupler) -> None:
        
        N = dt / self.coupling_timestep
        if N % 1 != 0:
            raise ValueError(f"dt={str(dt)} must be an integer multiple of coupling_timestep={str(self.coupling_timestep)}.")

        _forcing = self.model.forcing.copy(
            sea_surface_temperature = jnp.asarray(self.state["SST"].transpose()),
        )

        _avg_predictions = []
        N = int(N)
        for _ in range(N):
            _new_state, _predictions = self._step_function(self._state, _forcing, jdt.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S")))
            self._state = _new_state
            _avg_predictions.append(_predictions)
            self._predictions_list.append(_predictions)
    
        _avg_predictions = mean_leaf(unwrap_leading_dims(stack_objects(_avg_predictions)), axis=0)
        p = _avg_predictions.physics

        # All the heat and freshwater fluxes are positive upward
        self.state["SHF"] = np.array(p.surface_flux.shf).sum(axis=2).transpose()
        self.state["LHF"] = np.array(p.surface_flux.evap / 1e3 * evaporation_enthalpy_of_water).sum(axis=2).transpose()
        self.state["precipitation"] = - np.array(p.condensation.precls + p.convection.precnv).transpose() / 1e3
        self.state["evaporation"] = np.array(p.surface_flux.evap / 1e3).sum(axis=2).transpose()
        self.state["u10m"] = np.array(p.surface_flux.u0).transpose()
        self.state["v10m"] = np.array(p.surface_flux.v0).transpose()


    def finalize(self, output: Optional[str] = None) -> xr.Dataset:

        print("Converting JCM data to xarray... ")
        ds = self.model.predictions_to_xarray(unwrap_leading_dims(stack_objects(self._predictions_list)))

        if output is not None:
            print(f"Output file: {output:s}")
            ds.to_netcdf(output, engine="netcdf4")

        return ds
