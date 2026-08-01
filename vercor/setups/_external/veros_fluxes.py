from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray

from veros.state import VerosState

# Reference surface temperature (K), matching era5_atmosphere.py's own
# _REFERENCE_SURFACE_TEMPERATURE convention. Used only as a fallback for NaN
# entries in exchanged fields (e.g. OCN land cells the ATM->OCN regridder
# legitimately has no atmosphere data for) -- never a physically meaningful
# value, since those cells get re-masked to zero in this function's output
# anyway.
_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0

# runtime_fields that need a non-zero fallback because they're a log argument
# or division denominator inside compute_ocean_surface_fluxes:
#   alz = log(zbot / zref)                              -- zbot = model_level_height
#   hol = ... * zbot * (tstar/thref + ...) / ustar**2    -- thref = potential_temperature
#   ssq = 0.98 * qsat(ts) / rbot                          -- rbot = density
# A NaN (or 0) there is masked to a clean *value* downstream by nan_to_num,
# but the log(0)/division-by-zero it hits along the way corrupts the
# *gradient* of everything that mixes with it in the same expression --
# including OCN's own SST-driven terms (tstar depends on OCN's `ts`), which
# do need a real gradient. Fields that only ever appear additively in the
# bulk formula are safe with a plain 0.0 fallback.
_RUNTIME_FIELD_FALLBACKS: dict[str, float] = {
    "model_level_height": 10.0,
    "potential_temperature": _REFERENCE_SURFACE_TEMPERATURE,
    "temperature": _REFERENCE_SURFACE_TEMPERATURE,
    "density": 1.3,
}


def _sanitize_runtime_field(
    name: str,
    value: RuntimeArray,
    dtype: DTypePolicy,
) -> jax.Array:
    """Replace NaN entries in one exchanged field with a fallback that's safe
    for compute_ocean_surface_fluxes's internal log/division, applied before
    that computation runs rather than cleaned up only after the fact."""

    array = as_jax_real_array(value, dtype)
    fallback = _RUNTIME_FIELD_FALLBACKS.get(name, 0.0)
    return jnp.where(jnp.isnan(array), fallback, array)


def compute_fluxes(
    veros_state: VerosState,
    runtime_fields: Mapping[str, RuntimeArray],
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute atmosphere-ocean fluxes from explicit Veros and runtime fields."""

    vs = veros_state.variables

    u_tgrid = 0.5 * (
        as_jax_real_array(vs.u[1:, 2:-2, -1, vs.tau], dtype)
        + as_jax_real_array(vs.u[:-1, 2:-2, -1, vs.tau], dtype)
    )
    v_tgrid = 0.5 * (
        as_jax_real_array(vs.v[2:-2, 1:, -1, vs.tau], dtype)
        + as_jax_real_array(vs.v[2:-2, :-1, -1, vs.tau], dtype)
    )

    temp = as_jax_real_array(vs.temp[2:-2, 2:-2, -1, vs.tau], dtype).T + 273.15

    (
        senf,
        latf,
        lwup,
        evap,
        taux,
        tauy,
        tref,
        qref,
        duu10n,
        ustar,
        tstar,
        qstar,
        dqfldt,
    ) = compute_ocean_surface_fluxes(
        constants,
        as_jax_real_array(vs.maskT[2:-2, 2:-2, -1], dtype).T,
        _sanitize_runtime_field("model_level_height", runtime_fields["model_level_height"], dtype),
        _sanitize_runtime_field("u_velocity", runtime_fields["u_velocity"], dtype),
        _sanitize_runtime_field("v_velocity", runtime_fields["v_velocity"], dtype),
        _sanitize_runtime_field("potential_temperature", runtime_fields["potential_temperature"], dtype),
        _sanitize_runtime_field("specific_humidity", runtime_fields["specific_humidity"], dtype),
        _sanitize_runtime_field("density", runtime_fields["density"], dtype),
        _sanitize_runtime_field("temperature", runtime_fields["temperature"], dtype),
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )
    _ = evap, tref, qref, duu10n, ustar, tstar, qstar

    qnet = (
        _sanitize_runtime_field(
            "net_shortwave_radiation_flux",
            runtime_fields["net_shortwave_radiation_flux"],
            dtype,
        )
        + _sanitize_runtime_field(
            "downward_longwave_radiation_flux",
            runtime_fields["downward_longwave_radiation_flux"],
            dtype,
        )
        + lwup
        + senf
        + latf
    )
    qnec = -jnp.where(dqfldt <= -1e10, 0.0, dqfldt)

    return (
        as_jax_real_array(taux, dtype),
        as_jax_real_array(tauy, dtype),
        as_jax_real_array(qnet, dtype),
        as_jax_real_array(qnec, dtype),
    )


__all__ = ["compute_fluxes"]
