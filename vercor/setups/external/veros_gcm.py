"""Veros ocean component factory."""

from __future__ import annotations

from functools import partial
from vercor.components import LifecycleHooks, ComponentSpec, HostComponent
from vercor.output.adapters import OutputConfig
from vercor.setup_config import PeriodOutput, VerosConfig
import vercor.setups.external.veros_gcm_state as _veros_gcm_state
import vercor.setups.external.veros_output as _veros_output
import vercor.setups.external.veros_runtime as _veros_runtime
from vercor.setups.external.veros_gcm_state import VerosGCMSetupState

try:
    import veros  # noqa: F401
except ImportError:
    raise ImportError(
        "The VerosGCM component requires the Veros package. Please install it with `pip install veros`."
    )


def make_veros_gcm(
    *,
    config: VerosConfig | None = None,
) -> HostComponent:
    """Return a host-backed Veros GCM component."""

    config = VerosConfig() if config is None else config
    period_output = (
        PeriodOutput() if config.output.period is None else config.output.period
    )
    state = VerosGCMSetupState(
        name=config.name,
        spinup_time=config.spinup.duration,
        custom_parameters=config.custom_parameters,
        restore_to_climatology=config.restore_to_climatology,
        do_spinup=config.spinup.enabled,
        output_frequency=period_output.frequency,
        output_variables=period_output.variables,
        jitted=config.jitted,
    )
    component = HostComponent.from_step(
        name=config.name,
        grid=state.grid,
        step=partial(_veros_runtime.step_veros_runtime, state),
        spec=ComponentSpec(
            inputs=_veros_gcm_state.VEROS_INPUT_FIELD_NAMES,
            outputs=("sea_surface_temperature",),
            defaults=_veros_gcm_state.veros_default_fields(),
            hooks=LifecycleHooks(initialize=state.initialize),
            output=OutputConfig(
                snapshot_writer=config.output.snapshot_writer
                or partial(_veros_output.write_veros_snapshot_output, state),
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_veros_gcm",
]
