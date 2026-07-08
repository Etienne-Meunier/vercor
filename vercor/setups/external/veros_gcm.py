"""Veros ocean component factory."""

from __future__ import annotations

from functools import partial
from typing import Any

from vercor.components import ComponentHooks, FieldSpec, HostComponent
from vercor.output.adapters import OutputSpec
from vercor.setup_config import PeriodOutputConfig, SpinupConfig
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
    name: str = "OCN",
    custom_parameters: dict[str, Any] | None = None,
    restore_to_climatology: bool = False,
    spinup: SpinupConfig | None = None,
    output: PeriodOutputConfig | None = None,
    jitted: bool = False,
) -> HostComponent:
    """Return a host-backed Veros GCM component."""

    spinup_config = SpinupConfig() if spinup is None else spinup
    output_config = PeriodOutputConfig() if output is None else output
    state = VerosGCMSetupState(
        name=name,
        spinup_time=spinup_config.duration,
        custom_parameters=custom_parameters,
        restore_to_climatology=restore_to_climatology,
        do_spinup=spinup_config.enabled,
        output_frequency=output_config.frequency,
        output_variables=output_config.variables,
        jitted=jitted,
    )
    component = HostComponent.from_step(
        name=name,
        grid=state.grid,
        step=partial(_veros_runtime.step_veros_runtime, state),
        spec=FieldSpec(
            inputs=_veros_gcm_state.VEROS_INPUT_FIELD_NAMES,
            outputs=("sea_surface_temperature",),
            defaults=_veros_gcm_state.veros_default_fields(),
        ),
        hooks=ComponentHooks(initialize=state.initialize),
        output=OutputSpec(
            snapshot_writer=partial(_veros_output.write_veros_snapshot_output, state)
        ),
    )
    return component


__all__ = [
    "make_veros_gcm",
]
