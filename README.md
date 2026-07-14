# VerCOR

Versatile Earth system COupleR (VerCOR) is a JAX-first coupler for composing
Earth-system model components, forcing data, exchanges, regridding, diagnostics,
and output. VerCOR 4.0.0a1 uses a protocol-first
component contract with one immutable declaration for fields, lifecycle hooks,
and runtime capabilities. Assembly is constructor-only, static workflows feed
chunk-oriented execution backends, and explicit `OutputProvider`/
`OutputTarget` contracts keep all output cadence and writes core-owned.

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

The package root intentionally exports exactly `Clock`, `Coupler`, `Exchange`,
`RectilinearGrid`, `RunState`, and `RuntimeOptions`. Advanced contracts live in
their canonical owner modules: `vercor.components`, `vercor.runtime`,
`vercor.physics`, `vercor.topology`, `vercor.coupler`, `vercor.exchanges`,
`vercor.regridding`, `vercor.grids`, `vercor.fields`, `vercor.state`,
`vercor.output`, and `vercor.setups`. `vercor.types`, `vercor.dtypes`, and
`vercor.jax_logging` provide supporting public typing, precision, and logging
contracts. Component, output, topology, and setup-specific names are not
duplicated at the root.

Configuration currently has four owners:

- `RuntimeOptions` owns static policy for dtype, backend, workflow, topology,
  and model-year length.
- `PhysicalConstants` is the frozen traced PyTree owner for physical constants.
- `ComponentSpec`: inputs, outputs, initial fields, lifecycle, transfer,
  execution capability, and output.
- Setup config dataclasses: construction policy for one bundled model.

`Coupler(...)` is the sole primary assembly path. Components, exchanges, and
run order are supplied together and exposed through read-only views; changed
configuration requires a new coupler. `run_order=()` is valid setup-only
semantics: setup, validation, state creation, and output preparation still run,
but no component is advanced by the runtime loop.

There is no primary `Settings`, `vercor.physical_constants`, or
`vercor.coupling` module. Physical values come from
`vercor.physics.PhysicalConstants`, runtime precision from
`vercor.runtime.RuntimeOptions.dtype`, and setup-specific configuration from
frozen setup/plugin dataclasses. Host transfer, shared PyTree mechanics, and
interpolation implementations are private under `vercor._host_arrays`,
`vercor._pytree`, and `vercor._interpolators`.

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
`grid`, `spec`, and `step`:

```python
from collections.abc import Mapping
from typing import Any

from vercor.components import ComponentSpec, StepContext
from vercor.types import RuntimeArray


class WarmingModel:
    name = "MODEL"

    def __init__(self, model_grid: RectilinearGrid) -> None:
        self.grid = model_grid
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        )

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
adapter. `CallableComponent(...)` provides the same JAX contract with less
boilerplate. `LifecycleHooks(setup=...)` may return
`SetupResult(fields=..., payload=...)`; `TransferPolicy` selects current,
linear, or daily source data. A `StepResult` with omitted payload
preserves it, while an explicit replacement updates it. Compiled scanned JAX
execution requires every replacement to keep the setup payload's PyTree
structure; host execution may clear or restructure payload state.

### Host component

Use `CallableComponent` with `ComponentSpec(execution="host")` for a Python or
foreign-runtime model. `RuntimeOptions(backend="auto")` selects the host loop:

```python
from collections.abc import Mapping

from vercor.components import CallableComponent, ComponentSpec
from vercor.types import RuntimeArray


def host_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"counter": fields["counter"] + 1.0}


host = CallableComponent(
    "HOST",
    grid,
    host_step,
    spec=ComponentSpec(
        outputs=("counter",),
        initial_fields={"counter": 0.0},
        execution="host",
    ),
)
host_state = Coupler(clock, components=(host,), run_order=("HOST",)).run()
```

Forced JAX execution rejects host components. Forced host execution runs every
component through the Python loop.

### Custom execution backend

A backend consumes core-authored plans through the validated public driver and
must return `RunState`:

```python
from vercor import RuntimeOptions
from vercor.runtime import ExecutionChunk, ExecutionContext, RuntimeDriver
from vercor.state import RunState


class SequentialBackend:
    def execute(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        chunk: ExecutionChunk,
        driver: RuntimeDriver,
    ) -> RunState:
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return state


backend_model = WarmingModel(grid)
backend_coupler = Coupler(
    clock,
    components=(backend_model,),
    run_order=(backend_model.name,),
    runtime=RuntimeOptions(backend=SequentialBackend()),
)
backend_state = backend_coupler.run()
```

VerCOR owns workflow validation, chunk boundaries, cancellation, and period
output around custom backend calls. A backend must consume every supplied plan
exactly once through `RuntimeDriver.run_step(...)`.

### Custom topology policy

Topology is also structural. Policies inspect a public read-only context and
return patches keyed by stable exchange `route_id` values:

```python
from vercor import RuntimeOptions
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
)


class NoMaskTopology:
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

Lifecycle and output policy belong on one `ComponentSpec`. Snapshot writers
receive the original public component, its final public state view, the payload,
and a coordinator-allocated path. Generic period output samples declared runtime
fields; an empty variable list defaults to declared outputs:

```python
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
)
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
    SnapshotContext,
)
from vercor.types import RuntimeArray


def setup_hook(owner: Any, context: SetupContext) -> None:
    assert owner.name == "OUTPUT"
    context.logger.info("Initialized {}", owner.name)


def snapshot_writer(context: SnapshotContext) -> None:
    value = context.state.field("temperature")
    context.output_path.write_text(str(value), encoding="utf-8")


class KelvinProvider:
    def sample(self, context: OutputContext) -> OutputFrame:
        return OutputFrame(
            {
                "surface_temperature": OutputVariable(
                    ("latitude", "longitude"),
                    context.state.field("temperature"),
                    {"units": "K"},
                )
            },
            metadata={"source": "custom-provider"},
        )


def output_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"temperature": fields["temperature"] + 1.0}


output_model = CallableComponent(
    "OUTPUT",
    grid,
    output_step,
    spec=ComponentSpec(
        outputs=("temperature",),
        initial_fields={"temperature": 280.0},
        lifecycle=LifecycleHooks(setup=setup_hook),
        output=OutputSpec(
            provider=KelvinProvider(),
            snapshot_writer=snapshot_writer,
            period=PeriodOutput(
                frequency="step",
                variables=("surface_temperature",),
            ),
        ),
    ),
)
output_coupler = Coupler(
    clock, components=(output_model,), run_order=(output_model.name,)
)
output_directory = Path("output")
output_state = output_coupler.run(output=OutputTarget(output_directory))
```

Output is opt-in. `Coupler.run(output=None)` performs no I/O, while one
`OutputTarget` controls period, final runtime-view, and snapshot output paths.
`OutputTarget(directory)` enables all three kinds by default; set
`write_period`, `write_final_fields`, or `write_snapshots` to `False` to disable
one. For every provider, an empty `PeriodOutput.variables` selects the complete
frame, a non-empty tuple selects that ordered subset, and an unknown name is an
error. Providers see the post-step state, payload, and end-of-step model time.
Enabled output is rejected when `Coupler.run()` receives traced state leaves;
use the default `output=None` for differentiated or outer-jitted runs.

## Further reading

The [VerCOR 4.0.0a1 API architecture review](docs/api-architecture-review.md)
contains the complete public/private inventory and release decisions. Existing
applications should follow the runnable [3-to-4 migration guide](docs/migration-3-to-4.md);
maintainers use the [release checklist](docs/releasing.md) and
[changelog](CHANGELOG.md). The independently packaged
[`tests/fixtures/public_plugin`](tests/fixtures/public_plugin) fixture exercises
plugin-owned frozen configuration and dependency-injected assembly together
with a structural component and regridder, a stable route ID, a non-empty
topology patch, a custom workflow/backend, immutable state replacement, and
period/snapshot output. It uses only canonical public VerCOR modules, while
[`tests/fixtures/public_plugin_3_0`](tests/fixtures/public_plugin_3_0) freezes a
valid 3.0-only workflow that is intentionally rejected by the removed authoring
surface. The current fixture proves installed-wheel isolation and strict mypy
using public imports only; the frozen fixture remains historical artifact
evidence and is not executed against v4. This alpha intentionally ships no v3
adapter namespace.
