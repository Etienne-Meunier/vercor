"""Run the differentiable-Veros fork's ACC setup alone in a vercor Coupler.

Uses the local ``Veros-Autodiff`` checkout instead of pip's ``veros`` (not
installed; add its ``veros/`` package directory to ``sys.path`` instead), and
selects the ACC channel setup instead of the bundled global 4-degree one.

No atmosphere/land coupling, no differentiability -- this only exercises the
ordinary host/numpy runtime, same as running Veros on its own, just wired
through vercor's Coupler/CallableComponent machinery instead of Veros' own
``veros run`` entry point.
"""

import os
import sys
import jax

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))  # veros-ad submodule instead of pip veros
sys.path.insert(0, _REPO_ROOT)  # repo root, for `import vercor`

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RuntimeOptions
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import VerosConfig, make_veros_gcm

if __name__ == "__main__":
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="acc",
            uses_atmosphere_forcing=False))

    
    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365,
        calendar="noleap",
    )

    cpl = Coupler(
        clock=clock,
        components=[ocn],
        run_order=["OCN"],
        runtime=RuntimeOptions(),
    )

    initial_state = cpl.initial_state()

    temp_before = jnp.asarray(initial_state._component_state("OCN").payload.variables.temp)
    sst_before = jnp.asarray(initial_state.component("OCN").field("sea_surface_temperature"))

    #jrun = jax.jit(cpl.run)
    final_state = cpl.run(initial_state, output=None)

    temp_after = jnp.asarray(final_state._component_state("OCN").payload.variables.temp)
    sst_after = jnp.asarray(final_state.component("OCN").field("sea_surface_temperature"))

    print()
    print("temp mean before:", float(temp_before.mean()))
    print("temp mean after: ", float(temp_after.mean()))
    print("temp identical (array_equal):", bool(jnp.array_equal(temp_before, temp_after)))
    print("max abs diff temp:", float(jnp.abs(temp_after - temp_before).max()))
    print()
    print("sst mean before:", float(sst_before.mean()))
    print("sst mean after: ", float(sst_after.mean()))
    print("sst identical (array_equal):", bool(jnp.array_equal(sst_before, sst_after)))
