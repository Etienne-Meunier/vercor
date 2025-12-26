import numpy as np

from vercor.components.base import Component
from vercor.grid import RectilinearGrid


# === jcm imports ===
import jax
import jax.numpy as jnp

import jax_datetime as jdt

from dinosaur import primitive_equations_states

import tree_math
from jcm.model import Model
from jcm.forcing import default_forcing
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.physics_interface import PhysicsState
from vercor.components.external.JCM_tools import mean_leaf, stack_objects, unwrap_leading_dims
from vercor.settings import VercorSettings

from datetime import timedelta

from dataclasses import dataclass
from typing import Any, Optional, List
import typing

import xarray as xr




from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vercor.coupler import Coupler
# ===================

latent_heat_of_vaporization = VercorSettings.latvap

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
        save_interval: timedelta = timedelta(hours=24),
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
            binary_mask=np.ones_like(model.geometry.fmask).transpose(),  # This is used for interpolation, which all points are valid
        )
        super().__init__(name, grid)
        
        # has to be defined after super() is called
        self._fields2import = [
            "sst",
            "land_surface_temperature",
        ]
 
        self._fields2export = [
            "u10m",
            "v10m",
            "SHF",
            "LHF",
            "TA2M",
        ]

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
        _modal_state = self.model._prepare_initial_modal_state()
        self._state = JCMState(
            metadata = _modal_state,
            phydata = PhysicsData.zeros(self.model.coords.horizontal.nodal_shape, self.model.coords.vertical.layers),
            prog = dynamics_state_to_physics_state(_modal_state, self.model.primitive),
        )
        
        self.forcing = default_forcing(self.model.coords.horizontal).copy(lfluxland=True)
        self._step_function = self._generate_step_function(jitted=self.jitted)
        
        grid_shape = self.grid.shape

        zeros = np.zeros(grid_shape)
        self.cdata["sst"] = zeros + 273.15 + 15.0
        self.cdata["land_surface_temperature"] = zeros
        self.cdata["u10m"] = zeros.copy()
        self.cdata["v10m"] = zeros.copy()
        self.cdata["LHF"] = zeros.copy()
        self.cdata["SHF"] = zeros.copy()
        self.cdata["TA2M"] = zeros.copy()
        self._predictions_list = []

    def step(self, dt, time, coupler) -> None:
        N = dt / self.coupling_timestep
        if N % 1 != 0:
            raise ValueError(
                f"dt={str(dt)} must be an integer multiple of coupling_timestep={str(self.coupling_timestep)}."
            )

        print("Mean of sst: ", jnp.asarray(self.incoming_fields.sst.data).mean())
        print("number of sst that is less than 250: ", np.sum(self.incoming_fields.sst.data < 250.0))

        self.incoming_fields.sst.data[self.incoming_fields.sst.data < 250.0] = 288.15
        self.incoming_fields.land_surface_temperature.data[self.incoming_fields.land_surface_temperature.data < 250.0] = 288.15

        forcing = self.forcing.copy(
            stl_am=jnp.asarray(self.incoming_fields.land_surface_temperature).transpose(),
            sea_surface_temperature=jnp.asarray(self.incoming_fields.sst).transpose(),
        )

        _avg_predictions = []
        N = int(N)
        for _ in range(N):
            _new_state, _predictions = self._step_function(
                self._state,
                forcing,
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
        self.cdata["u10m"] = np.array(p.surface_flux.u0).transpose()
        self.cdata["v10m"] = np.array(p.surface_flux.v0).transpose()
        self.cdata["TA2M"] = np.array(p.surface_flux.t0).transpose()
        self.cdata["SHF"] = np.array(p.surface_flux.shf).sum(axis=2).transpose()
        self.cdata["LHF"] = (
            np.array(p.surface_flux.evap / 1e3 * latent_heat_of_vaporization)
            .sum(axis=2)
            .transpose()
        )

    def _finalize(self, output: Optional[str] = None) -> xr.Dataset:
        # Current JCM returns an Any but is actually an xr.Dataset
        ds = typing.cast(
            xr.Dataset,
            xr.merge(
                [
                    _prediction.to_xarray()
                    for _prediction in self._predictions_list
                ]
            )
        )
        if output is not None:
            print(f"Output file: {output:s}")
            ds.to_netcdf(output, engine="netcdf4")

        return ds
