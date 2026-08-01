import os
import sys

sys.path.insert(0, "/Users/emeunier/Desktop/Projets/Veros-Autodiff/veros")  # use veros-ad instead of pip veros
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `import vercor`

from datetime import datetime

import jax
import jax.numpy as jnp

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.setups import make_era5_atmosphere, make_era5_land, VerosConfig, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

if __name__ == "__main__":
    atm = make_era5_atmosphere()
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=True,
            restore_to_climatology=True,
        ),
    )
    lnd = make_era5_land()

    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=5,
        calendar="noleap",
    )
    run_order = ["OCN", "LND", "ATM"]

    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
            # Plain bilinear: OCN land cells legitimately get NaN here (no
            # ATM forcing applies over land) -- that's meaningful "no data"
            # information, not a bug. compute_fluxes now sanitizes the fields
            # that need a non-singular value before using them.
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_BASIC_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )

    initial_state = cpl.initial_state()

    def final_temp_sq_sum(temp_value: jax.Array) -> jax.Array:
        """Return sum(temp**2) after rollout, for a replacement surface temp."""
        
        component_state = initial_state._component_state("OCN")
        baseline_temp = component_state.payload.variables.temp
        new_temp = baseline_temp.at[:, :, -1, :].set(temp_value)
        new_payload = set_veros_variable(component_state.payload, "temp", new_temp)
        state = initial_state._with_component_state("OCN", component_state.with_payload(new_payload))

        result = cpl.run(state, output=None)
        result_payload = result._component_state("OCN").payload
        return jnp.sum(result_payload.variables.temp ** 2)

    grad_fn = jax.value_and_grad(final_temp_sq_sum)
    value, gradient = grad_fn(jnp.asarray(7.0))

    print("gradient (autodiff):   ", gradient)
    print("value:                 ", value)

    eps = 1e-2
    value_plus = final_temp_sq_sum(jnp.asarray(7.0 + eps))
    value_minus = final_temp_sq_sum(jnp.asarray(7.0 - eps))
    fd_gradient = (value_plus - value_minus) / (2 * eps)

    rel_error = abs(gradient - fd_gradient) / abs(fd_gradient)
    print("gradient (finite diff):", fd_gradient)
    print("relative error:        ", rel_error)
