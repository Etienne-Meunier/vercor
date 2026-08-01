from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray

from veros.state import VerosState


def _sanitize_runtime_field(value: RuntimeArray, dtype: DTypePolicy) -> jax.Array:
    array = as_jax_real_array(value, dtype)
    return jnp.where(jnp.isnan(array), 1.0, array)  # masked out downstream; just needs to avoid log(0)/div-by-0


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
        _sanitize_runtime_field(runtime_fields["model_level_height"], dtype),
        _sanitize_runtime_field(runtime_fields["u_velocity"], dtype),
        _sanitize_runtime_field(runtime_fields["v_velocity"], dtype),
        _sanitize_runtime_field(runtime_fields["potential_temperature"], dtype),
        _sanitize_runtime_field(runtime_fields["specific_humidity"], dtype),
        _sanitize_runtime_field(runtime_fields["density"], dtype),
        _sanitize_runtime_field(runtime_fields["temperature"], dtype),
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )
    _ = evap, tref, qref, duu10n, ustar, tstar, qstar

    qnet = (
        _sanitize_runtime_field(runtime_fields["net_shortwave_radiation_flux"], dtype)
        + _sanitize_runtime_field(runtime_fields["downward_longwave_radiation_flux"], dtype)
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
