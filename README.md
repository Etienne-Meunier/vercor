# VerCOR

Versatile Earth system COupleR (VerCOR) is a JAX-first coupler for composing
Earth-system model components, forcing data, exchanges, regridding, diagnostics,
and output. VerCOR 3.1 keeps the valid 3.0 API while making structural
`ComponentLike` objects the canonical extension boundary.

## Installation

Install the core package normally. Development tools are optional:

```bash
python -m pip install vercor
python -m pip install "vercor[test]"  # tests only
python -m pip install "vercor[dev]"   # formatting, lint, typing, build, tests
```

Install the `jcm` or `veros` extra before using those external-model factories.
CAMulator additionally requires NCAR's
[MILES-CREDIT](https://github.com/NCAR/miles-credit). CREDIT is intentionally
not pinned until an exact compatible release has been verified.

## Public API

The stable core owner modules are `vercor.components`, `vercor.runtime`,
`vercor.topology`, `vercor.coupling`, `vercor.exchanges`,
`vercor.regridding`, `vercor.grids`, `vercor.fields`, `vercor.state`,
`vercor.output`, and `vercor.setups`. `vercor.types`, `vercor.dtypes`, and
`vercor.jax_logging` provide supporting public typing, precision, and logging
contracts. The root `vercor` package keeps convenience exports for common core
workflows without duplicating setup- or topology-specific aliases.

Configuration has four owners:

- `RuntimeOptions` owns static policy for execution, topology, dtype, and the
  runtime.
- `Settings` is mutable setup-time metadata for physics and component/model
  constants. Its values may be JAX-traced when the container is constructed
  inside a differentiated workflow.
- `ComponentSpec`: fields, lifecycle, execution capability, and output.
- Setup config dataclasses: construction policy for one bundled model.

Prefer three assembly paths:

- the `Coupler` constructor for complete one-off setups;
- `CouplerSpec` for reusable recipes;
- mutators for incremental assembly.

Public mutators safely invalidate preparation; direct configuration mutation
after preparation is an error.

The following snippets share this small grid and clock:

```python
from datetime import datetime

from vercor import Clock, Coupler
from vercor.grids import RectilinearGrid

grid = RectilinearGrid.uniform(
    "demo",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
clock = Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
```

### Default built-in setup

Bundled factories return ordinary public components. This dependency-free slab
ocean is a complete default workflow:

```python
from vercor.setups import make_slab_ocean

ocean = make_slab_ocean(grid)
coupler = Coupler(clock, components=(ocean,), run_order=("OCN",))
final_state = coupler.run()
sea_surface_temperature = final_state.component("OCN").field(
    "sea_surface_temperature"
)
```

### Structural custom JAX component

No VerCOR inheritance is required. A structural component supplies `name`,
`grid`, `spec`, `initial_fields`, `initialize`, and `step`:

```python
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from vercor.components import ComponentSpec, SetupContext, StepContext
from vercor.types import RuntimeArray


class WarmingModel:
    name = "MODEL"

    def __init__(self, model_grid: RectilinearGrid) -> None:
        self.grid = model_grid
        self.spec = ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 280.0},
        )

    def initial_fields(self) -> Mapping[str, RuntimeArray]:
        return {"temperature": jnp.full(self.grid.shape, 280.0)}

    def initialize(self, context: SetupContext) -> None:
        _ = context

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"] + 0.25}


model = WarmingModel(grid)
custom_coupler = Coupler(clock, components=(model,), run_order=(model.name,))
custom_state = custom_coupler.run()
```

Structural lifecycle hooks receive the original `WarmingModel`, not a private
adapter. `Component.from_step(...)` provides the same JAX contract with less
boilerplate.

### Host component

Use `HostComponent.from_step` for a Python or foreign-runtime model. It forces
host capability, and `RuntimeOptions(execution="auto")` selects the host loop:

```python
from collections.abc import Mapping

from vercor.components import ComponentSpec, HostComponent
from vercor.types import RuntimeArray


def host_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"counter": fields["counter"] + 1.0}


host = HostComponent.from_step(
    "HOST",
    grid,
    host_step,
    spec=ComponentSpec(outputs=("counter",), defaults={"counter": 0.0}),
)
host_state = Coupler(clock, components=(host,), run_order=("HOST",)).run()
```

Forced JAX execution rejects host components. Forced host execution runs every
component through the Python loop.

### Custom execution backend

A backend controls ordering through the validated public driver. It must return
`RunState`; `step` must be a concrete scalar integer within the clock range:

```python
from vercor import RuntimeOptions
from vercor.runtime import ExecutionContext, RuntimeDriver
from vercor.state import RunState


class SequentialBackend:
    def run(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        driver: RuntimeDriver,
    ) -> RunState:
        for step in range(context.clock.steps):
            for component_name in context.run_order:
                state = driver.step_component(
                    state, component_name, step=step
                )
        return state


backend_model = WarmingModel(grid)
backend_coupler = Coupler(
    clock,
    components=(backend_model,),
    run_order=(backend_model.name,),
    runtime=RuntimeOptions(execution=SequentialBackend()),
)
backend_state = backend_coupler.run()
```

Custom backends currently cannot be combined with period output because the
public backend contract has no period-session hook.

### Custom topology policy

Topology is also structural. Policies inspect a public read-only context and
return patches keyed by `(source, target, regrid_key)`:

```python
from vercor import RuntimeOptions
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
)


class NoMaskTopology:
    def applies(self, context: TopologyContext) -> bool:
        return bool(context.components)

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        _ = context
        return ExchangeTopologyPatch()


topology_model = WarmingModel(grid)
topology_coupler = Coupler(
    clock,
    components=(topology_model,),
    run_order=(topology_model.name,),
    runtime=RuntimeOptions(topology=NoMaskTopology()),
)
topology_state = topology_coupler.run()
```

Use `vercor.topology.SurfaceMaskPolicy()` for the bundled ATM/OCN/LND policy;
leave topology as `None` for ordinary setup-agnostic graphs.

### Lifecycle hooks and output

Lifecycle and output policy belong on one `ComponentSpec`. Snapshot writers see
only public metadata/state. Generic period output samples declared runtime
fields; an empty variable list defaults to declared outputs:

```python
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vercor.components import (
    Component,
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
)
from vercor.output import (
    OutputConfig,
    PeriodOutput,
    SnapshotContext,
)
from vercor.types import RuntimeArray


def initialize_hook(owner: Any, context: SetupContext) -> None:
    assert owner.name == "OUTPUT"
    context.logger.info("Initialized {}", owner.name)


def snapshot_writer(context: SnapshotContext) -> None:
    value = context.state.field("temperature")
    context.output_path.write_text(str(value), encoding="utf-8")


def output_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"temperature": fields["temperature"] + 1.0}


output_model = Component.from_step(
    "OUTPUT",
    grid,
    output_step,
    spec=ComponentSpec(
        outputs=("temperature",),
        defaults={"temperature": 280.0},
        lifecycle=LifecycleHooks(initialize=initialize_hook),
        output=OutputConfig(
            snapshot_writer=snapshot_writer,
            period=PeriodOutput(frequency="step", variables=("temperature",)),
        ),
    ),
)
output_coupler = Coupler(
    clock, components=(output_model,), run_order=(output_model.name,)
)
output_state = output_coupler.run()  # period files are written in the current cwd
output_directory = Path("output")
output_directory.mkdir(exist_ok=True)
output_coupler.write_outputs(output_state, output_dir=output_directory)
```

Period output is an I/O workflow and is rejected when `Coupler.run()` receives
traced state leaves. Disable period output for differentiated or outer-jitted
runs. `write_outputs(output_dir=...)` controls final runtime-view and snapshot
paths; it does not redirect the period files emitted during `run()`.

## Further reading

See the [VerCOR 3.1.1 API architecture review](docs/api-architecture-review.md)
for the complete public/private inventory, execution precedence, and migration
table. The independently packaged
[`tests/fixtures/public_plugin`](tests/fixtures/public_plugin) fixture exercises
the current 3.1 API, while
[`tests/fixtures/public_plugin_3_0`](tests/fixtures/public_plugin_3_0) freezes a
valid 3.0-only workflow. Both prove installed-wheel isolation and strict mypy
using public imports only.
