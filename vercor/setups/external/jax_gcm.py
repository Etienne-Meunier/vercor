"""JAXGCM/JCM atmosphere component factory."""

from __future__ import annotations

from datetime import timedelta
from functools import partial

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.model import ForcingData
from jcm.physics_interface import TerrainData

from vercor.components import Component, ComponentHooks, FieldSpec
from vercor.output.adapters import OutputSpec
from vercor.setup_config import PeriodOutputConfig, SpinupConfig
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
    name: str = "ATM",
    custom_parameters: dict[str, float] | None = None,
    model_timestep: timedelta = timedelta(minutes=30),
    save_interval: timedelta = timedelta(days=1),
    forcing_data: ForcingData | None = None,
    spinup: SpinupConfig | None = None,
    output: PeriodOutputConfig | None = None,
    jitted: bool = True,
) -> Component:
    """Return a differentiable JAXGCM/JCM atmosphere component."""

    spinup_config = SpinupConfig() if spinup is None else spinup
    output_config = PeriodOutputConfig() if output is None else output
    state = JAXGCMSetupState(
        coords=coords,
        terrain=terrain,
        name=name,
        custom_parameters=custom_parameters,
        model_timestep=model_timestep,
        save_interval=save_interval,
        spinup_time=spinup_config.duration,
        forcing_data=forcing_data,
        output_frequency=output_config.frequency,
        do_spinup=spinup_config.enabled,
        jitted=jitted,
    )
    component = Component.from_step(
        name=name,
        grid=state.grid,
        step=partial(_jax_gcm_runtime.step_jax_gcm_component, state),
        spec=FieldSpec(
            inputs=("land_surface_temperature", "sea_surface_temperature"),
            outputs=(
                "land_surface_temperature",
                "sea_surface_temperature",
                "total_surface_temperature",
                *_jax_gcm_fields.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
                "pressure",
            ),
            defaults=_jax_gcm_runtime.jax_gcm_default_fields(),
        ),
        hooks=ComponentHooks(
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
        output=OutputSpec(
            snapshot_writer=partial(
                _jax_gcm_output.write_jax_gcm_snapshot_output, state
            )
        ),
    )
    return component


__all__ = [
    "make_jax_gcm",
]
