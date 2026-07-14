"""Veros ocean component factory."""

from __future__ import annotations

from functools import partial
from typing import Any

from vercor.components import (
    CallableComponent,
    Component,
    LifecycleHooks,
    ComponentSpec,
)
from vercor.output import OutputConfig
from vercor.setups.config import VerosConfig


def _load_veros_implementation() -> tuple[Any, Any, Any, type[Any]]:
    """Load Veros implementation owners after runtime configuration."""

    import vercor.setups._external.veros_gcm_state as veros_gcm_state
    import vercor.setups._external.veros_output as veros_output
    import vercor.setups._external.veros_runtime as veros_runtime
    from vercor.setups._external.veros_gcm_state import VerosGCMSetupState

    return veros_gcm_state, veros_output, veros_runtime, VerosGCMSetupState


def make_veros_gcm(
    *,
    config: VerosConfig | None = None,
) -> Component:
    """Return a host-backed Veros GCM component."""

    try:
        import veros  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The VerosGCM component requires the Veros package. Please install it "
            "with `pip install veros`."
        ) from error

    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
    )

    configure_veros_runtime()

    (
        _veros_gcm_state,
        _veros_output,
        _veros_runtime,
        VerosGCMSetupState,
    ) = _load_veros_implementation()

    config = VerosConfig() if config is None else config
    period_output = config.output.period
    state = VerosGCMSetupState(
        name=config.name,
        spinup_time=config.spinup.duration,
        custom_parameters=config.custom_parameters,
        restore_to_climatology=config.restore_to_climatology,
        do_spinup=config.spinup.enabled,
        output_frequency=None if period_output is None else period_output.frequency,
        output_variables=() if period_output is None else period_output.variables,
        jitted=config.jitted,
    )
    component = CallableComponent(
        config.name,
        state.grid,
        partial(_veros_runtime.step_veros_runtime, state),
        spec=ComponentSpec(
            inputs=_veros_gcm_state.VEROS_INPUT_FIELD_NAMES,
            outputs=("sea_surface_temperature",),
            initial_fields=_veros_gcm_state.veros_default_fields(),
            execution="host",
            lifecycle=LifecycleHooks(setup=state.setup),
            output=OutputConfig(
                snapshot_writer=config.output.snapshot_writer
                or partial(_veros_output.write_veros_snapshot_output, state),
                period=config.output.period,
            ),
        ),
    )
    if period_output is not None:
        setattr(component, "_period_output_handled_by_step", True)
    return component


__all__ = [
    "make_veros_gcm",
]
