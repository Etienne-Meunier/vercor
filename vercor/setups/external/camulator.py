"""CAMulator host-runtime atmosphere component factory."""

from __future__ import annotations

from functools import partial

from vercor.components import LifecycleHooks, ComponentSpec, HostComponent
from vercor.output.adapters import OutputConfig
from vercor.setup_config import CAMulatorConfig, PeriodOutput
import vercor.setups.external.camulator_contracts as _camulator_contracts
import vercor.setups.external.camulator_output as _camulator_output
import vercor.setups.external.camulator_runtime as _camulator_runtime
from vercor.setups.external.camulator_gcm_state import CAMulatorGCMSetupState


def make_camulator_gcm(
    *,
    config: CAMulatorConfig,
) -> HostComponent:
    """Return a host-backed CAMulator atmosphere component."""

    period_output = (
        PeriodOutput() if config.output.period is None else config.output.period
    )
    state = CAMulatorGCMSetupState(
        config_path=config.config_path,
        name=config.name,
        model_weights_path=config.model_weights_path,
        output_subfolder_name=config.output_subfolder_name,
        init_noise=config.init_noise,
        spinup_time=config.spinup.duration,
        do_spinup=config.spinup.enabled,
        device=config.device,
        output_cpus_number=config.output_cpus_number,
        output_frequency=period_output.frequency,
        logger=config.logger,
    )
    component = HostComponent.from_step(
        name=config.name,
        grid=state.grid,
        step=partial(_camulator_runtime.step_camulator_runtime, state),
        spec=ComponentSpec(
            inputs=("sea_surface_temperature", "land_surface_temperature"),
            outputs=_camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
            defaults=_camulator_contracts.camulator_runtime_field_defaults(),
            hooks=LifecycleHooks(initialize=state.initialize),
            output=OutputConfig(
                snapshot_writer=config.output.snapshot_writer
                or partial(_camulator_output.write_camulator_snapshot_output, state),
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_camulator_gcm",
]
