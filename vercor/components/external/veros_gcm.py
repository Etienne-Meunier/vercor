from copy import deepcopy

from typing import TYPE_CHECKING, Any, Callable
import jax
import jax.numpy as jnp
from datetime import datetime, timedelta

from vercor.clock import ModelDateTime
from vercor.components.external.veros_runtime_settings import *  # noqa: F403,F401

from veros.setups.global_4deg import GlobalFourDegreeSetup
from veros.core.operators import numpy as npx, update, at
from veros.routines import veros_kernel, veros_routine
from veros.state import KernelOutput, VerosState
from veros.tools import get_periodic_interval

from vercor.components.base import Component
from vercor.grid import RectilinearGrid
from vercor.fluxes.bulk_formula_cesm import new_flux_atmOcn
from vercor.settings import VercorSettings
from vercor.tools import _runtime_array_to_host
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.runtime import RuntimeComponentState


try:
    import veros  # noqa: F401
except ImportError:
    raise ImportError(
        "The VerosGCM component requires the Veros package. Please install it with `pip install veros`."
    )


class CustomGlobalFourDegree(GlobalFourDegreeSetup):
    @veros_kernel
    def set_forcing_kernel(state):  # type: ignore
        vs = state.variables
        settings = state.settings

        year_in_seconds = 365 * 86400.0
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

    @veros_routine
    def set_diagnostics(self, state):  # type: ignore
        settings = state.settings
        state.diagnostics["snapshot"].output_frequency = 365 * 86400.0
        state.diagnostics["overturning"].output_frequency = 365 * 86400.0
        state.diagnostics["overturning"].sampling_frequency = settings.dt_tracer
        state.diagnostics["energy"].output_frequency = 365 * 86400.0
        state.diagnostics["energy"].sampling_frequency = 86400
        average_vars = [
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "surface_taux",
            "surface_tauy",
            "psi",
            "qnet",
            "qnec",
        ]
        state.diagnostics["averages"].output_variables = average_vars
        state.diagnostics["averages"].output_frequency = 365 * 86400.0
        state.diagnostics["averages"].sampling_frequency = 86400


@jax.jit
def _update_veros_interior(
    array: object,
    interior_value: object,
) -> jax.Array:
    array_jax = jnp.asarray(array, dtype=jnp.float64)
    interior_value_jax = jnp.asarray(interior_value, dtype=jnp.float64)
    return array_jax.at[2:-2, 2:-2, ...].set(interior_value_jax)


@jax.jit
def _prepare_surface_forcing_fields(
    taux: object,
    tauy: object,
    qnet: object,
    qnec: object,
    restore_to_climatology: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    restore_to_climatology_jax = jnp.asarray(restore_to_climatology, dtype=bool)

    def _prepare(field: object) -> jax.Array:
        field_jax = jnp.asarray(field, dtype=jnp.float64)
        return jnp.nan_to_num(field_jax.T[..., jnp.newaxis])

    taux_prepared = _prepare(taux)
    tauy_prepared = _prepare(tauy)
    qnet_prepared = _prepare(qnet)
    qnec_prepared = _prepare(qnec)
    qnec_prepared = jnp.where(
        restore_to_climatology_jax, qnec_prepared, jnp.zeros_like(qnec_prepared)
    )

    return taux_prepared, tauy_prepared, qnet_prepared, qnec_prepared


@jax.jit
def _extract_surface_temperature(
    temperature: object,
    tau: object,
) -> jax.Array:
    temperature_array = jnp.asarray(temperature, dtype=jnp.float64)
    tau_index = jnp.asarray(tau, dtype=jnp.int32)
    return temperature_array[2:-2, 2:-2, -1, tau_index].T + 273.15


def compute_fluxes(
    component_state: "VerosGCM", settings: VercorSettings
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

    cs = component_state
    vs = cs._veros_state.variables

    # u & v have Arakawa-C grid staggering in Veros
    # require additional interpolation
    u_tgrid = 0.5 * (
        jnp.asarray(vs.u[1:, 2:-2, -1, vs.tau], dtype=jnp.float64)
        + jnp.asarray(vs.u[:-1, 2:-2, -1, vs.tau], dtype=jnp.float64)
    )
    v_tgrid = 0.5 * (
        jnp.asarray(vs.v[2:-2, 1:, -1, vs.tau], dtype=jnp.float64)
        + jnp.asarray(vs.v[2:-2, :-1, -1, vs.tau], dtype=jnp.float64)
    )

    temp = jnp.asarray(vs.temp[2:-2, 2:-2, -1, vs.tau], dtype=jnp.float64).T + 273.15

    (
        senf,
        latf,
        lwup,
        evap,
        taux,
        tauy,
        tref,
        qref,
        duu10n,
        ustar,
        tstar,
        qstar,
        dqfldt,
    ) = new_flux_atmOcn(
        settings,
        jnp.asarray(vs.maskT[2:-2, 2:-2, -1], dtype=jnp.float64).T,
        jnp.asarray(cs.data["model_level_height"], dtype=jnp.float64),
        jnp.asarray(cs.data["u_velocity"], dtype=jnp.float64),
        jnp.asarray(cs.data["v_velocity"], dtype=jnp.float64),
        jnp.asarray(cs.data["potential_temperature"], dtype=jnp.float64),
        jnp.asarray(cs.data["specific_humidity"], dtype=jnp.float64),
        jnp.asarray(cs.data["density"], dtype=jnp.float64),
        jnp.asarray(cs.data["temperature"], dtype=jnp.float64),
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )

    # Signs & directions convention in Veros
    # Negative out:        ↑  LW_up ↑  SENf ↑  LATf ↑
    # Positive in:  SW_net ↓  LW_dw ↓  SENf ↓  LATf ↓

    qnet = (
        jnp.asarray(cs.data["net_shortwave_radiation_flux"], dtype=jnp.float64)
        + jnp.asarray(cs.data["downward_longwave_radiation_flux"], dtype=jnp.float64)
        + lwup
        + senf
        + latf
    )
    qnec = -jnp.where(dqfldt <= -1e10, 0.0, dqfldt)

    return (
        jnp.asarray(taux, dtype=jnp.float64),
        jnp.asarray(tauy, dtype=jnp.float64),
        jnp.asarray(qnet, dtype=jnp.float64),
        jnp.asarray(qnec, dtype=jnp.float64),
    )


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
    Convert an in-place Veros step into a copy-before-mutate boundary helper.
    """
    n_state = copy_state(state, jitted=jitted)
    # This is a function that modifies state object inplace
    step(n_state)

    return n_state


def set_variable(
    state: VerosState,
    variable_name: str,
    variable_value: RuntimeArray,
    jitted: bool = True,
) -> VerosState:
    n_state = copy_state(state, jitted=jitted)
    vs = n_state.variables

    with n_state.variables.unlock():
        var = getattr(vs, variable_name)
        updated_var = _update_veros_interior(var, variable_value)
        setattr(vs, variable_name, _runtime_array_to_host(updated_var))

    return n_state


class VerosGCM(Component):
    def __init__(
        self,
        name: str = "OCN",
        spinup_time: timedelta = timedelta(days=2),
        custom_parameters: dict[str, Any] | None = None,
        restore_to_climatology: bool = False,
        do_spinup: bool = False,
        jitted: bool = False,
    ) -> None:
        """
        Veros GCM component based on the Global 4-degree setup from Veros.

        Arguments:
            name (str): component name
            spinup_time (timedelta): duration of the initial Veros spinup
            custom_parameters (dict[str, Any]): dictionary of custom parameter values to override
                                                the default settings in the GlobalFourDegreeSetup
            restore_to_climatology (bool): whether to apply restoring to climatology in
                                           the surface temperature (add salinity later) tendency
            do_spinup (bool): whether to perform the initial spinup with ERA-Interim forcing
            jitted (bool): whether to use JIT compilation for the Veros model step function
        """

        override = custom_parameters or {}

        self.model = CustomGlobalFourDegree(override=override)
        self.model.setup()
        self._veros_state = copy_state(self.model.state, jitted=jitted)
        self._step_function = lambda state: pure(
            state, jitted=jitted, step=self.model.step
        )

        self.do_spinup = do_spinup
        self.spinup_time = spinup_time
        self.restore_to_climatology = restore_to_climatology
        self.jitted = jitted

        self.dt_tracer = getattr(self._veros_state.settings, "dt_tracer")
        self.spinup_steps = int(self.spinup_time.total_seconds() // self.dt_tracer)

        mask = jnp.where(
            jnp.asarray(self._veros_state.variables.maskT[:, :, -1]) > 0.0,
            1.0,
            0.0,
        )

        grid = RectilinearGrid(
            name=name,
            longitude=self._veros_state.variables.xt[2:-2],
            latitude=self._veros_state.variables.yt[2:-2],
            binary_mask=mask[2:-2, 2:-2].T,
        )

        super().__init__(name, grid=grid)

    def initialize(self, coupler: "Coupler") -> None:
        dt_seconds = coupler.clock.dt_seconds
        self.model_substeps = int(dt_seconds // self.dt_tracer)

        if dt_seconds % self.dt_tracer != 0:
            raise ValueError(
                f"dt_tracer ({self.dt_tracer}) must be a multiple of dt ({dt_seconds})"
            )

        # Initial spinup is performed with ERA-Interim (default) atmospheric forcing
        if self.do_spinup and "ATM" in coupler.run_sequence.order:
            # Do it similar to CESM spinup when coupling with atmosphere is on
            coupler.logger.info(
                f" Performing Veros spinup for {self.spinup_time} day(s)..."
            )
            for i in range(self.spinup_steps):
                coupler.logger.info(f" Step {i+1} / {self.spinup_steps}")
                self._veros_state = self._step_function(self._veros_state)

        self.data["sea_surface_temperature"] = _extract_surface_temperature(
            self._veros_state.variables.temp,
            self._veros_state.variables.tau,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        dt_seconds: float,
        runtime_settings: Any | None = None,
        *,
        time: datetime | ModelDateTime | None = None,
        coupler: "Coupler | None" = None,
    ) -> "RuntimeComponentState":
        """Advance the host-backed Veros boundary."""

        _ = dt_seconds, runtime_settings
        if time is None or coupler is None:
            return component_state

        self._sync_data_from_runtime_state(component_state)

        taux, tauy, qnet, qnec = compute_fluxes(self, coupler.settings)
        forcing_fields = _prepare_surface_forcing_fields(
            taux, tauy, qnet, qnec, self.restore_to_climatology
        )

        for variable_name, variable_value in zip(
            ("taux", "tauy", "qnet", "qnec"),
            forcing_fields,
        ):
            self._veros_state = set_variable(
                self._veros_state,
                variable_name,
                variable_value,
                jitted=self.jitted,
            )

        for i in range(self.model_substeps):
            coupler.logger.info(f" Veros sub-step {i+1} / {self.model_substeps}")
            self._veros_state = self._step_function(self._veros_state)

        self.data["sea_surface_temperature"] = _extract_surface_temperature(
            self._veros_state.variables.temp,
            self._veros_state.variables.tau,
        )
        from vercor.runtime import RuntimeFieldStore

        return component_state.with_data(
            RuntimeFieldStore.from_mapping(self.data)
        ).with_runtime_payload(component_state.runtime_payload)
