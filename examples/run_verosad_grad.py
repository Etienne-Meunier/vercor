"""Run the differentiable-Veros fork's ACC setup alone in a vercor Coupler.

Uses the local ``Veros-Autodiff`` checkout instead of pip's ``veros`` (not
installed; add its ``veros/`` package directory to ``sys.path`` instead), and
selects the ACC channel setup instead of the bundled global 4-degree one.

No atmosphere/land coupling. Differentiates w.r.t. the Veros-internal
``temp`` variable directly (mirroring
``Veros-Autodiff/notebooks/demonstration/gradient-computation.ipynb``), not
via ``RunState.replace_fields`` -- the ``sea_surface_temperature`` field
vercor exposes is a write-only diagnostic extracted from ``temp`` after each
step; it is never read back into the physics, so perturbing it produces no
gradient. Setting the component's runtime ``payload`` directly requires the
private ``RunState._component_state``/``ComponentRuntimeState.with_payload``
API -- there is currently no public way to seed a component's opaque runtime
payload.
"""

import os
import sys
import jax
import jax.numpy as jnp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))  # veros-ad submodule instead of pip veros
sys.path.insert(0, _REPO_ROOT)  # repo root, for `import vercor`

from datetime import datetime

from vercor import Clock, Coupler, RuntimeOptions
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.setups._external.veros_state import set_veros_variable

if __name__ == "__main__":
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="acc",
            uses_atmosphere_forcing=False,
            jitted=True,
            execution="jax"))

    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=50,
        calendar="noleap",
    )

    cpl = Coupler(
        clock=clock,
        components=[ocn],
        run_order=["OCN"],
        runtime=RuntimeOptions(),
    )

    initial_state = cpl.initial_state()
    
    def final_temp_sq_sum(temp_value: jax.Array) -> jax.Array:
        """Return sum(temp**2) after rollout, for a replacement surface temp."""

        # Making new initial variable: start from the spun-up temp field and
        # only overwrite the surface level (last z index) with temp_value.
        # variables.temp is stored in Celsius internally, so temp_value here
        # is Celsius too (not the Kelvin convention used for SST elsewhere).
        component_state = initial_state._component_state("OCN")
        baseline_temp = component_state.payload.variables.temp
        new_temp = baseline_temp.at[:, :, -1, :].set(temp_value)
        new_payload = set_veros_variable(component_state.payload, "temp", new_temp)
        state = initial_state._with_component_state("OCN", component_state.with_payload(new_payload))

        result = cpl.run(state, output=None)
        result_payload = result._component_state("OCN").payload
        return jnp.sum(result_payload.variables.temp ** 2)

    #final_temp_sq_sum(jnp.asarray(7.0))
    grad_fn = jax.jit(jax.value_and_grad(final_temp_sq_sum))
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


    value, gradient = grad_fn(jnp.asarray(8.0))

    print("gradient (autodiff):   ", gradient)
    print("value:                 ", value)