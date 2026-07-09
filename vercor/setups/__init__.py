"""Public setup adapter factories for VerCOR examples and applications."""

from __future__ import annotations

from typing import Any

from vercor.setups.config import (
    CAMulatorConfig,
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
)
from vercor.setups._lazy_imports import (
    LazyExport,
    lazy_export_names,
    resolve_lazy_export,
)
from vercor.setups._slab import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.setups._jcm import (
    JCMLandAtmosphereSetup,
    JCMInputs,
    load_jcm_inputs,
    make_jcm_land_atmosphere,
)

_LAZY_EXPORTS = {
    "make_camulator_gcm": LazyExport("_external.camulator", "make_camulator_gcm"),
    "make_camulator_land": LazyExport(
        "_external.camulator_land",
        "make_camulator_land",
    ),
    "make_era5_atmosphere": LazyExport("_data.era5_atmosphere", "make_era5_atmosphere"),
    "make_era5_land": LazyExport("_data.era5_land", "make_era5_land"),
    "make_era5_ocean": LazyExport("_data.era5_ocean", "make_era5_ocean"),
    "make_erainterim_ocean": LazyExport(
        "_data.erainterim_ocean",
        "make_erainterim_ocean",
    ),
    "make_jax_gcm": LazyExport("_external.jax_gcm", "make_jax_gcm"),
    "make_jcm_land": LazyExport("_data.jcm_land", "make_jcm_land"),
    "make_veros_gcm": LazyExport("_external.veros_gcm", "make_veros_gcm"),
}

__all__ = [
    "CAMulatorConfig",
    "JAXGCMConfig",
    "JCMLandAtmosphereConfig",
    "JCMLandAtmosphereSetup",
    "JCMInputs",
    "Spinup",
    "VerosConfig",
    "load_jcm_inputs",
    "make_slab_atmosphere",
    "make_slab_land",
    "make_slab_ocean",
    "make_slab_seaice",
    "make_jcm_land_atmosphere",
    *lazy_export_names(_LAZY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Load optional setup factories only when requested."""

    return resolve_lazy_export(__name__, _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    """Return package exports without importing optional adapters."""

    return __all__
