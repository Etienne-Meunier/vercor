"""Run the differentiable-Veros fork's global 4-degree learning setup, coupled to
ERA5 atmosphere/land, in a vercor Coupler.

Uses the local ``Veros-Autodiff`` checkout instead of pip's ``veros`` (not
installed; add its ``veros/`` package directory to ``sys.path`` instead), and
selects the global 4-degree learning setup (ported into vercor as
``GlobalFourDegreeLearningSetup``, see
``vercor/setups/_external/veros_setup_global4deg_learning.py``) -- a
differentiable-friendly variant of ``setups/global_4deg/global_4deg_learning.py``
from the Veros-Autodiff checkout (``enable_streamfunction=False``,
``eq_of_state_type=3``, diagnostics disabled).

Structured the same way as ``run_veros_with_era5data.py``: ERA5 atmosphere and
land components exchange fluxes with the ocean each step
(``uses_atmosphere_forcing=True``), so no ``jax.jit`` wrapping here -- the ERA5
components read forcing data from disk and aren't jit-safe.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "veros"))  # veros-ad submodule instead of pip veros
sys.path.insert(0, _REPO_ROOT)  # repo root, for `import vercor`

from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import make_era5_atmosphere
from vercor.setups import make_era5_land
from vercor.setups import VerosConfig, make_veros_gcm
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

if __name__ == "__main__":
    atm = make_era5_atmosphere(
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=(
                    "surface_pressure",
                    "temperature",
                    "net_shortwave_radiation_flux",
                    "downward_longwave_radiation_flux",
                ),
            ),
        )
    )
    ocn = make_veros_gcm(
        config=VerosConfig(
            setup="global_4deg_learning",
            uses_atmosphere_forcing=True,
            restore_to_climatology=True,
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=(
                        "temp",
                        "salt",
                        "u",
                        "v",
                        "w",
                        "surface_taux",
                        "surface_tauy",
                    ),
                ),
            ),
        ),
    )
    lnd = make_era5_land(
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=("land_surface_temperature",),
            ),
        )
    )

    # Clock and sequence
    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365,
        calendar="noleap",
    )
    run_order = ["OCN", "LND", "ATM"]

    # Exchanges
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
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

    cpl.run(output=OutputTarget("."))
