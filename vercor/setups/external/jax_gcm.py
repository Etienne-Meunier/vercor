"""JAXGCM/JCM atmosphere component factory."""

from __future__ import annotations

from functools import partial

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.physics_interface import TerrainData

from vercor.components import Component, LifecycleHooks, ComponentSpec
from vercor.output import OutputConfig
from vercor.setup_config import JAXGCMConfig
import vercor.setups.external.jax_gcm_fields as _jax_gcm_fields
import vercor.setups.external.jax_gcm_output as _jax_gcm_output
import vercor.setups.external.jax_gcm_runtime as _jax_gcm_runtime
from vercor.setups.external.jax_gcm_state import JAXGCMSetupState

try:
    import jcm  # noqa: F401
except ImportError:
    raise ImportError(
        "The JAXGCM component requires the jcm package. Please install it with "
        "`pip install jcm`."
    )


def make_jax_gcm(
    coords: CoordinateSystem,
    terrain: TerrainData,
    *,
    config: JAXGCMConfig | None = None,
) -> Component:
    """Return a differentiable JAXGCM/JCM atmosphere component."""

    config = JAXGCMConfig() if config is None else config
    period_output = config.output.period
    state = JAXGCMSetupState(
        coords=coords,
        terrain=terrain,
        name=config.name,
        custom_parameters=config.custom_parameters,
        model_timestep=config.model_timestep,
        save_interval=config.save_interval,
        spinup_time=config.spinup.duration,
        forcing_data=config.forcing_data,
        output_frequency=None if period_output is None else period_output.frequency,
        do_spinup=config.spinup.enabled,
        jitted=config.jitted,
    )
    component = Component.from_step(
        name=config.name,
        grid=state.grid,
        step=partial(_jax_gcm_runtime.step_jax_gcm_component, state),
        spec=ComponentSpec(
            inputs=("land_surface_temperature", "sea_surface_temperature"),
            outputs=(
                "land_surface_temperature",
                "sea_surface_temperature",
                "total_surface_temperature",
                *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
                "pressure",
            ),
            defaults=_jax_gcm_runtime.jax_gcm_default_fields(),
            lifecycle=LifecycleHooks(
                initialize=state.initialize,
                create_payload=partial(
                    _jax_gcm_runtime.create_jax_gcm_runtime_payload,
                    state,
                ),
                prefill=partial(
                    _jax_gcm_runtime.prefill_jax_gcm_runtime_fields,
                    state,
                ),
                validate=partial(
                    _jax_gcm_runtime.validate_jax_gcm_runtime_state,
                    state,
                ),
            ),
            output=OutputConfig(
                snapshot_writer=config.output.snapshot_writer
                or partial(_jax_gcm_output.write_jax_gcm_snapshot_output, state),
                period=config.output.period,
            ),
        ),
    )
    return component


__all__ = [
    "make_jax_gcm",
]
