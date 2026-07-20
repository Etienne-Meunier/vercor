"""CAMulator host-runtime atmosphere component factory."""

from __future__ import annotations

from functools import partial

from vercor.components import (
    CallableComponent,
    Component,
    LifecycleHooks,
    ComponentSpec,
)
from vercor.output import OutputSpec
from vercor.setups.config import CAMulatorConfig


def make_camulator_gcm(
    *,
    config: CAMulatorConfig,
) -> Component:
    """Return a host-backed CAMulator atmosphere component."""

    if config.spinup.enabled:
        raise ValueError(
            "CAMulator spinup is not implemented; set Spinup(enabled=False)."
        )

    from vercor.setups._external import camulator_imports

    camulator_imports.load_credit_modules()

    from vercor.setups._external.camulator_runtime_settings import (
        configure_camulator_runtime,
    )

    configure_camulator_runtime()

    import vercor.setups._external.camulator_contracts as _camulator_contracts
    import vercor.setups._external.camulator_output as _camulator_output
    import vercor.setups._external.camulator_runtime as _camulator_runtime
    from vercor.setups._external.camulator_gcm_state import CAMulatorGCMSetupState

    state = CAMulatorGCMSetupState(
        config_path=config.config_path,
        name=config.name,
        model_weights_path=config.model_weights_path,
        init_noise=config.init_noise,
        device=config.device,
        logger=config.logger,
    )
    output_provider = (
        _camulator_output.camulator_output_provider(state)
        if config.output.provider is None
        else config.output.provider
    )
    snapshot_writer = config.output.snapshot_writer
    if snapshot_writer is None:
        snapshot_provider = _camulator_output.camulator_output_provider(
            state,
            latest_only=True,
        )
        snapshot_writer = partial(
            _camulator_output.write_camulator_snapshot_output,
            state,
            snapshot_provider,
        )
    component = CallableComponent(
        config.name,
        state.grid,
        partial(_camulator_runtime.step_camulator_runtime, state),
        spec=ComponentSpec(
            inputs=("sea_surface_temperature", "land_surface_temperature"),
            outputs=_camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
            initial_fields=_camulator_contracts.camulator_runtime_field_defaults(),
            execution="host",
            lifecycle=LifecycleHooks(setup=state.setup),
            output=OutputSpec(
                provider=output_provider,
                snapshot_writer=snapshot_writer,
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_camulator_gcm",
]
