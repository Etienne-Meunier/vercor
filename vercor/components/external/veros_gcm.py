from copy import deepcopy

from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta

from vercor.components.external.veros_runtime_settings import *

from veros.setups.global_4deg import GlobalFourDegreeSetup
from veros.core.operators import update, at
from veros.core.operators import numpy as npx, update, at
from veros.routines import veros_kernel
from veros.state import KernelOutput, VerosState
from veros.tools import get_periodic_interval

from vercor.components.base import Component
from vercor.grid import RectilinearGrid
from vercor.fluxes.bulk_formula_cesm import shr_flux_atmOcn


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def copy_state(tree: VerosState, jitted: bool = True) -> VerosState:
    if jitted:
        dimensions = deepcopy(tree._dimensions)
        settings_meta = deepcopy(tree.settings.__metadata__)
        plugin_interfaces = deepcopy(tree._plugin_interfaces)
        var_meta = deepcopy(tree._var_meta)

        state_copy = VerosState(
            var_meta, settings_meta, dimensions, plugin_interfaces=plugin_interfaces
        )

        with state_copy.settings.unlock():
            for k, v in tree.settings.items():
                state_copy.settings.__setattr__(k, v)

        state_copy._variables = deepcopy(tree._variables)
        state_copy.timers = deepcopy(tree.timers)
        state_copy.profile_timers = deepcopy(tree.profile_timers)

        # Replace the above with the following line when Etienne put his fixes in Veros
        # return tree_map(lambda x : x.copy(), tree)
    else:
        state_copy = tree

    return state_copy


def pure(state: VerosState, jitted: bool, step: Callable) -> VerosState:
    """
    Convert the state function into a "pure step" copying the input state
    """
    n_state = copy_state(state, jitted=jitted)
    # This is a function that modifies state object inplace
    step(n_state)

    return n_state


def set_variable(
    variable_name: str, state: VerosState, variable_value: NDArray, jitted: bool = True
) -> VerosState:

    n_state = copy_state(state, jitted=jitted)
    vs = n_state.variables

    with n_state.variables.unlock():
        var = getattr(vs, variable_name)
        var = update(var, at[2:-2, 2:-2, ...], variable_value)
        setattr(vs, variable_name, var)

    return n_state


class CustomGlobalFourDegree(GlobalFourDegreeSetup):
    @veros_kernel
    def set_forcing_kernel(state): # type: ignore
        vs = state.variables
        settings = state.settings

        year_in_seconds = 360 * 86400.0
        (n1, f1), (n2, f2) = get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12.0, 12
        )

        # wind stress
        vs.surface_taux = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        # tke flux
        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[1:-1, 1:-1],
                npx.sqrt(
                    (
                        0.5
                        * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])
                        / settings.rho_0
                    )
                    ** 2
                    + (
                        0.5
                        * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])
                        / settings.rho_0
                    )
                    ** 2
                )
                ** 1.5,
            )

        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
        qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
        qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
        vs.forc_temp_surface = (
            (qnet + qnec * (sst - vs.temp[:, :, -1, vs.tau]))
            * vs.maskT[:, :, -1]
            / cp_0
            / settings.rho_0
        )

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface = (
            1.0
            / t_rest
            * (sss - vs.salt[:, :, -1, vs.tau])
            * vs.maskT[:, :, -1]
            * vs.dzt[-1]
        )

        # apply simple ice mask
        mask = npx.logical_and(
            vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
            vs.forc_temp_surface < 0.0,
        )
        vs.forc_temp_surface = npx.where(mask, 0.0, vs.forc_temp_surface)
        vs.forc_salt_surface = npx.where(mask, 0.0, vs.forc_salt_surface)

        return KernelOutput(
            surface_taux=vs.surface_taux,
            surface_tauy=vs.surface_tauy,
            forc_tke_surface=vs.forc_tke_surface,
            forc_temp_surface=vs.forc_temp_surface,
            forc_salt_surface=vs.forc_salt_surface,
        )


class VerosGCM(Component):
    def __init__(
        self,
        name: str = "OCN",
        do_spinup: bool = True,
        spinup_days: int = 2,
        jitted: bool = False,
    ) -> None:
        """
        Veros GCM component based on the Global 4-degree setup from Veros.

        Arguments:
            name (str): component name
        """

        self.model = CustomGlobalFourDegree()
        self.model.setup()
        self._state = copy_state(self.model.state, jitted=jitted)
        self._step_function = lambda state: pure(
            state, jitted=jitted, step=self.model.step
        )

        # TODO: pass the below as settings
        self.do_spinup = do_spinup
        self.spinup_days = spinup_days
        self.jitted = jitted

        self.dt_tracer = getattr(self._state.settings, "dt_tracer")
        self.dt_mom = getattr(self._state.settings, "dt_mom")
        self.spinup_steps = int(self.dt_tracer * self.spinup_days // self.dt_mom)

        mask = np.where(self._state.variables.maskT[:, :, -1] > 0.0, 1.0, 0.0)

        self.grid = RectilinearGrid(
            name=name,
            longitude=self._state.variables.xt[2:-2],
            latitude=self._state.variables.yt[2:-2],
            binary_mask=mask[2:-2, 2:-2].T,
        )

        super().__init__(name, grid=self.grid)

    def initialize(self, coupler: "Coupler") -> None:
        dt_seconds = coupler.clock.dt_seconds
        self.model_substeps = int(dt_seconds // self.dt_mom)

        if dt_seconds % self.dt_mom != 0:
            raise ValueError(
                f"dt_mom ({self.dt_mom}) must be a multiple of dt ({dt_seconds})"
            )

        if self.do_spinup and "ATM" in coupler.run_sequence.order:
            # Do it similar to CESM spinup when coupling with atmosphere is on
            print(
                " " * 36 + f"Performing Veros spinup for {self.spinup_days} day(s)..."
            )
            for i in range(self.spinup_steps):
                print(" " * 40 + f"Step {i+1} / {self.spinup_steps}", end="\r")
                self._state = self._step_function(self._state)

        self.data["sst"] = self._state.variables.temp[
            2:-2, 2:-2, -1, self._state.variables.tau
        ].T

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:

        vs = self._state.variables

        u_tgrid = 0.5 * (vs.u[1:, 2:-2, -1, vs.tau] + vs.u[:-1, 2:-2, -1, vs.tau])

        v_tgrid = 0.5 * (vs.v[2:-2, 1:, -1, vs.tau] + vs.v[2:-2, :-1, -1, vs.tau])

        temp = vs.temp[2:-2, 2:-2, -1, vs.tau].T + 273.15
        dummy_mask = np.ones_like(self.data["ubot"])

        """senf, latf, lwup, evap, taux, tauy, tref, qref, duu10n, ustar, tstar, qstar = flux_atmOcn(
            coupler.settings,
            dummy_mask,# 1 - vs.maskT[2:-2, 2:-2, -1].T,
            self.data["rbot"],
            self.data["zbot"],
            self.data["ubot"],
            self.data["vbot"],
            self.data["qbot"],
            self.data["tbot"],
            self.data["thbot"],
            u_tgrid[1:-2, :].T,
            v_tgrid[:, 1:-2].T,
            temp,
        )"""

        senf, latf, lwup, evap, taux, tauy, tref, qref, duu10n, ustar, tstar, qstar = (
            shr_flux_atmOcn(
                coupler.settings,
                dummy_mask,  # 1 - vs.maskT[2:-2, 2:-2, -1].T,
                self.data["zbot"],
                self.data["ubot"],
                self.data["vbot"],
                self.data["thbot"],
                self.data["qbot"],
                self.data["rbot"],
                self.data["tbot"],
                # u & v have Arakawa-C grid staggering in Veros
                # (need additional interpolation)
                u_tgrid[1:-2, :].T,
                v_tgrid[:, 1:-2].T,
                temp,
            )
        )

        # Signs & directions convention in Veros
        # Negative out:           LW_up ↑  SENf ↑  LATf ↑
        # Positive in:  SW_net ↓  LW_dw ↓  SENf ↓  LATf ↓

        qnet = self.data["swr_net"] + self.data["lwr_dw"] + lwup + senf + latf

        for var_name, var_value in {
            "taux": taux.T[..., np.newaxis],
            "tauy": tauy.T[..., np.newaxis],
            "qnet": qnet.T[..., np.newaxis],
        }.items():
            self._state = set_variable(
                var_name, self._state, var_value, jitted=self.jitted
            )

        for i in range(self.model_substeps):
            print(" " * 40 + f"Veros sub-step {i+1} / {self.model_substeps}", end="\r")
            self._state = self._step_function(self._state)

        self.data["sst"] = self._state.variables.temp[
            2:-2, 2:-2, -1, self._state.variables.tau
        ].T
