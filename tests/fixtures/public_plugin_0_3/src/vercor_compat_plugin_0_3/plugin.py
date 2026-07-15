"""Exercise only public component contracts available in VerCOR 0.3."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import jax.numpy as jnp

from vercor.clock import Clock
from vercor.components import Component, ComponentSpec
from vercor.coupling import Coupler
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


def _step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
    """Advance the frozen compatibility field once."""

    return {"temperature": fields["temperature"] + 2.0}


def run_smoke() -> dict[str, object]:
    """Run the frozen 0.3 workflow and return compact evidence."""

    grid = RectilinearGrid.uniform(
        "compat-0-3-grid",
        nlon=2,
        nlat=2,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
    )
    component = Component.from_step(
        "MODEL",
        grid,
        _step,
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=(component.name,),
    )
    temperature = float(
        jnp.asarray(coupler.run().component("MODEL").field("temperature"))[0, 0]
    )
    if temperature != 282.0:
        raise AssertionError("frozen VerCOR 0.3 workflow produced an invalid field")
    return {"temperature": temperature}


__all__ = ["run_smoke"]
