# Migrating from VerCOR 0.3 to VerCOR 0.4

VerCOR 0.4.0a1 is intentionally source-breaking. This release does not include
a legacy adapter namespace: migrate imports, component declarations, assembly,
execution, state access, and output explicitly.

## Import and ownership changes

The package root now contains only `Clock`, `Coupler`, `Exchange`,
`RectilinearGrid`, `RunState`, and `RuntimeOptions`. Import advanced contracts
from their canonical modules:

| VerCOR 0.3 concept | VerCOR 0.4 replacement |
| --- | --- |
| `Settings` and a physical-constants facade | `vercor.physics.PhysicalConstants` for traced SI values; `RuntimeOptions.dtype` for precision |
| `Component`/`HostComponent` inheritance and `ComponentLike` | structural `vercor.components.Component`, `CallableComponent`, or `DataComponent` |
| `initial_fields()` and `initialize()` | `ComponentSpec.initial_fields` and `LifecycleHooks.setup` returning `SetupResult` |
| constructor payload / payload factory | `SetupResult.payload` |
| `CouplerSpec`, incremental mutators, `vercor.coupling` | one complete `vercor.coupler.Coupler(...)` constructor |
| exchange callable identity | `Exchange.route_id` and `regridder_factory` |
| backend `run(...)` / driver component calls | backend `execute(...)` and `RuntimeDriver.run_step(...)` |
| backend- or component-owned output | `OutputSpec` providers plus a run-level `OutputTarget` |
| direct runtime-store access | `RunState.component(s)` and immutable `replace_fields` |

## Component and assembly migration

A 0.3 class typically mixed declaration, initialization, mutable payload, and
step behavior. In 0.4, declare fields and lifecycle once and supply the whole
graph to the constructor. This complete replacement is executable:

```python
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from vercor import Clock, Coupler
from vercor.components import ComponentSpec, StepContext
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


class MigratedAtmosphere:
    """Minimal structural 0.4 component with no VerCOR base class."""

    name = "ATM"

    def __init__(self, grid: RectilinearGrid) -> None:
        self.grid = grid
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
        return {"temperature": fields["temperature"] + 1.0}


migration_grid = RectilinearGrid.uniform(
    "migration",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
migration_component = MigratedAtmosphere(migration_grid)
migration_coupler = Coupler(
    Clock(datetime(2000, 1, 1), dt_seconds=60.0, steps=2),
    components=(migration_component,),
    run_order=(migration_component.name,),
)
migration_state = migration_coupler.run(output=None)
migrated_temperature = migration_state.component("ATM").field("temperature")
```

`output=None` is explicit above because differentiated applications should keep
I/O outside JAX transforms. It is also the default.

## Exchanges, topology, and workflow

Give repeated source/target routes distinct `route_id` values. A topology
policy returns one `ExchangeTopologyPatch` keyed by those route IDs. Scalar
routes require a `Regridder`; vector routes require `VectorRegridder`. Multiple
routes may not produce the same target field.

A custom workflow returns exactly one ascending `StepPlan` per clock step. A
plan may reorder or omit registered components, but it may not invent or repeat
one. A custom backend receives core-authored `ExecutionChunk` objects and must
consume each supplied plan exactly once through `RuntimeDriver.run_step`.

## Output migration

Attach sampling policy to `ComponentSpec.output` with `OutputSpec`,
`OutputProvider`, and `PeriodOutput`. Pass `OutputTarget(directory)` to
`Coupler.run` to enable period, final-field, and snapshot writes. The core owns
selection, accumulation, cadence, paths, host transfer, and NetCDF writes for
runtime, JAXGCM, Veros, CAMulator, and third-party providers.

Remove post-run `write_outputs` and component-native period-file calls. Snapshot
writers receive a public `SnapshotContext` and a collision-safe output path.

## Historical evidence

`tests/contracts/vercor-0.3.2-public-api.json` records the 0.3.2 surface used to
define this migration. `tests/fixtures/public_plugin_0_3` is retained only as a
historical artifact and is expected not to run against 0.4. It is not a supported
plugin lane or an adapter implementation.
