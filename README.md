# VerCOR

Versatile Earth system COupleR (VerCOR) is a JAX-first coupler for composing
Earth-system model components, forcing data, exchanges, regridding, diagnostics,
and output.

## Public API

The stable user-facing modules are:

- `vercor`: convenience exports for core workflows.
- `vercor.components`: component author contracts such as `Component`,
  `DataComponent`, `HostComponent`, `ComponentLike`, `ComponentInfo`, and
  `ComponentSpec`.
- `vercor.runtime`: runtime policy and extension contracts such as
  `RuntimeOptions`, `SurfaceMaskPolicy`, `ExecutionBackend`,
  `ExecutionContext`, `RuntimeDriver`, `RunState`, and `ComponentState`.
- `vercor.coupling`: `Coupler`, `CouplerSpec`, and `Exchange`.
- `vercor.fields`, `vercor.regridding`, `vercor.output`, `vercor.setups`,
  `vercor.recipes`, and `vercor.diagnostics`.

Custom models can either use `Component.from_step(...)`,
`HostComponent.from_step(...)`, `DataComponent.from_fields(...)`, or implement
the structural `ComponentLike` contract. Runtime execution is selected with
`ComponentSpec(execution="jax" | "host")`. Custom execution backends implement
`ExecutionBackend.run(state, *, context, driver)`.

Bundled ATM/OCN/LND setup examples opt in to
`RuntimeOptions(surface_masks=SurfaceMaskPolicy())`. For setup-agnostic custom
graphs, leave `RuntimeOptions.surface_masks` as `None`.

## Minimal Example

```python
from datetime import datetime

from vercor import Clock, Component, ComponentSpec, Coupler, DataComponent, Exchange
from vercor.grids import RectilinearGrid

grid = RectilinearGrid.uniform(
    "grid",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)

forcing = DataComponent.from_fields("FORCING", grid, {"heat_flux": 1.0})


def step(fields, context):
    return {"temperature": fields["temperature"] + fields["heat_flux"]}


model = Component.from_step(
    "MODEL",
    grid,
    step,
    spec=ComponentSpec(
        inputs=("heat_flux", "temperature"),
        outputs=("temperature",),
        defaults={"temperature": 280.0},
    ),
)

coupler = Coupler(
    Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=24),
    components=(forcing, model),
    exchanges=(Exchange("FORCING", "MODEL", ("heat_flux",)),),
    run_order=("FORCING", "MODEL"),
)
final_state = coupler.run()
```
