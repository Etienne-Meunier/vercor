"""CAMulator host-runtime atmosphere component factory."""

from __future__ import annotations

from functools import partial
from typing import Optional

from vercor.components import ComponentHooks, FieldSpec, HostComponent
from vercor.jax_logging import LoggerLike
from vercor.output.adapters import OutputSpec
from vercor.setup_config import PeriodOutputConfig, SpinupConfig
import vercor.setups.external.camulator_contracts as _camulator_contracts
import vercor.setups.external.camulator_output as _camulator_output
import vercor.setups.external.camulator_runtime as _camulator_runtime
from vercor.setups.external.camulator_gcm_state import CAMulatorGCMSetupState


def make_camulator_gcm(
    config_path: str,
    name: str = "ATM",
    model_weights_path: str = "checkpoint.pt00091.pt",
    output_subfolder_name: Optional[str] = None,
    init_noise: Optional[float] = None,
    spinup: SpinupConfig | None = None,
    device: str = "cuda",
    output_cpus_number: int = 8,
    output: PeriodOutputConfig | None = None,
    logger: LoggerLike | None = None,
) -> HostComponent:
    """Return a host-backed CAMulator atmosphere component."""

    spinup_config = SpinupConfig() if spinup is None else spinup
    output_config = PeriodOutputConfig() if output is None else output
    state = CAMulatorGCMSetupState(
        config_path=config_path,
        name=name,
        model_weights_path=model_weights_path,
        output_subfolder_name=output_subfolder_name,
        init_noise=init_noise,
        spinup_time=spinup_config.duration,
        do_spinup=spinup_config.enabled,
        device=device,
        output_cpus_number=output_cpus_number,
        output_frequency=output_config.frequency,
        logger=logger,
    )
    component = HostComponent.from_step(
        name=name,
        grid=state.grid,
        step=partial(_camulator_runtime.step_camulator_runtime, state),
        spec=FieldSpec(
            inputs=("sea_surface_temperature", "land_surface_temperature"),
            outputs=_camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
            defaults=_camulator_contracts.camulator_runtime_field_defaults(),
        ),
        hooks=ComponentHooks(initialize=state.initialize),
        output=OutputSpec(
            snapshot_writer=partial(
                _camulator_output.write_camulator_snapshot_output, state
            )
        ),
    )
    return component


__all__ = [
    "make_camulator_gcm",
]
