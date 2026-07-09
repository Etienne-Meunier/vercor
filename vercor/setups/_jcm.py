from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vercor.host_arrays import transposed_host_array
from vercor.components import Component, DataComponent
from vercor.grids import RectilinearGrid
from vercor.output import OutputConfig, PeriodOutput
from vercor.setup_config import JAXGCMConfig, Spinup


@dataclass(frozen=True)
class JCMLandAtmosphereSetup:
    """JCM setup components plus generated setup objects used by runnable scripts."""

    land: DataComponent
    atmosphere: Component
    coords: Any
    terrain: Any
    forcing: Any


@dataclass(frozen=True)
class JCMInputs:
    """Generated JCM coordinate, terrain, and forcing inputs."""

    coords: Any
    terrain: Any
    forcing: Any


def load_jcm_inputs(
    *,
    resolution: int = 31,
    input_data_directory: Path | None = None,
) -> JCMInputs:
    """Load JCM coordinate, terrain, and forcing inputs."""

    from vercor.setups._external.jax_gcm_tools import load_jcm_coords_terrain_forcing

    coords, terrain, forcing = load_jcm_coords_terrain_forcing(
        resolution=resolution,
        input_data_directory=input_data_directory,
    )
    return JCMInputs(coords=coords, terrain=terrain, forcing=forcing)


def _load_jcm_factories() -> tuple[
    Callable[..., DataComponent],
    Callable[..., Component],
]:
    """Return optional JCM setup factories from their owning modules."""

    from vercor.setups._data.jcm_land import make_jcm_land
    from vercor.setups._external.jax_gcm import make_jax_gcm

    return make_jcm_land, make_jax_gcm


def make_jcm_land_atmosphere(
    ocn_grid: RectilinearGrid,
    *,
    inputs: JCMInputs | None = None,
    config: JAXGCMConfig | None = None,
    custom_parameters: Mapping[str, float] | None = None,
    spinup: Spinup | None = None,
    jitted: bool | None = None,
    output: OutputConfig | None = None,
) -> JCMLandAtmosphereSetup:
    """Create paired JCM land and atmosphere setup components for an ocean grid."""

    if config is not None and (
        custom_parameters is not None
        or spinup is not None
        or jitted is not None
        or output is not None
    ):
        raise TypeError(
            "Use either config=JAXGCMConfig(...) or legacy JCM setup keyword "
            "arguments, not both."
        )

    make_jcm_land, make_jax_gcm = _load_jcm_factories()
    if config is None:
        spinup_config = Spinup(enabled=True) if spinup is None else spinup
        output_config = (
            OutputConfig(period=PeriodOutput(frequency="month"))
            if output is None
            else output
        )
        config = JAXGCMConfig(
            custom_parameters=(
                None if custom_parameters is None else dict(custom_parameters)
            ),
            spinup=spinup_config,
            output=output_config,
            jitted=True if jitted is None else jitted,
        )
    jcm_inputs = load_jcm_inputs() if inputs is None else inputs
    coords = jcm_inputs.coords
    terrain = jcm_inputs.terrain
    forcing = jcm_inputs.forcing
    land = make_jcm_land(coords, forcing, ocn_grid)

    # JAXGCM expects the terrain mask in host/transposed JCM layout.
    if land.grid.binary_mask is None:
        raise ValueError("JCM land grid requires a binary mask for terrain patching")
    terrain.fmask = transposed_host_array(land.grid.binary_mask)

    atmosphere = make_jax_gcm(
        coords,
        terrain,
        config=JAXGCMConfig(
            name=config.name,
            custom_parameters=config.custom_parameters,
            model_timestep=config.model_timestep,
            save_interval=config.save_interval,
            forcing_data=(
                forcing if config.forcing_data is None else config.forcing_data
            ),
            spinup=config.spinup,
            output=config.output,
            jitted=config.jitted,
        ),
    )
    return JCMLandAtmosphereSetup(
        land=land,
        atmosphere=atmosphere,
        coords=coords,
        terrain=terrain,
        forcing=forcing,
    )
