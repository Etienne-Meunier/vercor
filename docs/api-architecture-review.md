# VerCOR 3.1 API architecture review

This review compares the pre-3.1 implementation with the API shipped in
VerCOR 3.1. It treats public `__all__` declarations, annotations, tests, examples,
and the independently packaged fixture under `tests/fixtures/public_plugin` as
the source of truth. “Plugin” means a separately packaged Python object supplied
through dependency injection; VerCOR does not discover plugins globally.

## 1. Executive summary

Before 3.1, VerCOR already had useful public component, backend, topology, and
output contracts, but the implementation did not consistently honor those
boundaries. Structural components and VerCOR subclasses took different lifecycle
paths; public annotations exposed internal return vocabulary; runtime contracts,
topology, and dispatch data were rebuilt at several entry points; host and JAX
loops had split ownership; period output depended on the selected backend; and
merely importing bundled setup modules could load or configure optional models.

The essential 3.1 strategy is consolidation, not a second facade:

- `vercor.components.ComponentLike` is the canonical structural extension
  contract. `Component.from_step`, `HostComponent.from_step`, and
  `DataComponent.from_fields` are convenience adapters to the same runtime
  bridge.
- A `Coupler` prepares one private immutable `PreparedCoupling`, then reuses its
  normalized components, contracts, topology maps, dispatch context, run order,
  and configuration fingerprints for state creation, execution, and output.
- `RuntimeOptions.execution` selects one implementation owned by
  `vercor._runtime.backends`; `vercor._runtime.runner` only selects, checks, and
  delegates.
- Topology customization uniformly calls `TopologyPolicy.applies(context)` and
  then `build(context)`. Generic and JAXGCM period output use private schemas,
  boundary plans, and sum/count sessions with host writes between compiled
  chunks; existing native Veros/CAMulator host adapters retain their private
  step-managed accumulator path.
- Bundled setup factories remain the only supported entrance to their private
  adapters. Optional imports and environment configuration occur when a factory
  is invoked, not when `vercor` or `vercor.setups` is imported.

Essential changes are the contract corrections, original-object lifecycle
semantics, one prepared runtime, topology/backend/output unification, strict
validation, import isolation, PEP 561 packaging, installed-plugin tests, and the
3.1.0 release artifacts. Optional ideas are deliberately deferred: global
registries or entry-point discovery, Pydantic configuration, a generic forcing
index protocol, periodic-grid endpoint default changes, raw interpolator
removal, setup-package extraction, and wholesale private-directory
reorganization. Those additions have no demonstrated 3.1 requirement and would
increase compatibility cost.

No valid 3.0 root import is removed. The root convenience surface remains the
same 48 symbols; `ComponentStepReturn` is intentionally public only from its
owner package. Error tightening applies to invalid or previously silent cases,
not valid 3.0 workflows.

## 2. Duplication map

| Pre-3.1 duplication or boundary problem | 3.1 disposition | Reason |
| --- | --- | --- |
| Root convenience exports versus canonical owner modules | **kept separate** | The unchanged root supports concise workflows; owner modules provide stable, discoverable contracts. A reexport is not a second owner. |
| `RunState`/`ComponentState` visible through `vercor.runtime` and owned by `vercor.state` | **kept separate** | The runtime reexports are valid 3.0 imports; `vercor.state` remains canonical. |
| `Exchange` visible through `vercor.coupling` and owned by `vercor.exchanges` | **kept separate** | The coupling facade is convenient while `vercor.exchanges` owns the declaration. |
| Structural models, subclasses, callable wrappers, data adapters, and host adapters as apparently different component APIs | **merged** | They normalize through `vercor.components._adapter.normalize_component`; `ComponentLike` is canonical and the three class constructors are conveniences. |
| Private `_ComponentStepReturn`-style vocabulary and repeated union annotations | **renamed** and **moved public** | `vercor.components.ComponentStepReturn = Mapping[str, RuntimeArray] | StepResult` is the single owner-package alias used by public step contracts. It is not a new root alias. |
| `DataComponent.from_step` inheriting a constructor that returned an unrelated active component | **removed** | The path now raises a focused `TypeError`; data-only construction is `DataComponent.from_fields`. |
| Public-looking inherited runtime-store lifecycle methods | **renamed** and **moved private** | Underscored bridge methods keep `FieldStore`, `ComponentRuntimeState`, and `ExchangeContract` out of component-author APIs. |
| Lifecycle dispatch split between structural objects and convenience components | **merged** | Structural initialization is user `initialize`, refresh, spec hook with the original user object, refresh. Convenience components retain one lifecycle dispatch. |
| Rebuilding component normalization, exchange contracts, topology, dispatch groups, and run order in `initial_state`, `run`, and output | **merged** | One `PreparedCoupling` is built after lifecycle initialization and reused. |
| Runtime validation split across entry points | **merged** | Prepared contracts/topology validate created and supplied states; configuration fingerprints reject direct mutation after preparation. |
| Public mutators and direct component mutation as two setup paths | **kept separate** | `Coupler.add_component`, `add_exchange(s)`, and `set_run_order` intentionally invalidate preparation; unsupported direct mutation raises `CouplerError`. |
| `SurfaceMaskPolicy` special-case branch versus the public topology protocol | **merged** | Every policy follows `applies` then `build`; surface masks are one built-in policy. |
| Duplicate patch application and retained `SurfaceExchangeMasks` | **removed** | Patches apply once, validate keys/shapes, and derived surface masks remain local to construction. |
| Host loop, scanned loop, thin backend classes, and custom-backend lazy imports | **merged** | Loop/adaptation implementations live in `vercor._runtime.backends`; selection stays in `runner`. |
| Generic and JAXGCM period accumulation plus backend-specific write paths | **merged** | Shared output schemas, immutable sum/count accumulators, coalesced boundaries, and host writers support compiled/host execution. Model-specific extraction stays private beside the JAXGCM adapter. |
| Native Veros/CAMulator accumulation versus traced session accumulation | **kept separate** | `_ComponentOutputAdapter` preserves their native host layouts/cadence; `_PeriodOutputSession` is immutable and JAX-compatible for generic/JAXGCM output. A private marker prevents duplicate generic schemas. Both reuse shared variable/file primitives. |
| Parallel setup-package lazy registries | **removed** | `vercor.setups` owns the sole public lazy export table. Private setup packages are implementation owners, not alternate facades. |
| Setup-specific configuration mixed into `RuntimeOptions` or `Settings` | **kept separate** | Runtime policy, traced model constants, component contracts, and model construction have distinct owners. |
| Manual paired-JCM configuration copying | **merged** | `dataclasses.replace` changes only missing forcing data and preserves caller values/subclasses. |
| Exchange recipes defined privately and reexported publicly | **moved public** | Recipe constants now live directly in `vercor.recipes`, their public owner. |
| Registry discovery, Pydantic models, service containers, and setup extraction | **deferred** | They are speculative for current customization cases and are **nice to improve** only with concrete demand. |

## 3. Bad design decisions

| Priority | Design problem | Consequence | Concrete fix |
| --- | --- | --- | --- |
| **must change** | Treating VerCOR subclasses as the implicit plugin contract | External models had to inherit internals or mimic undocumented behavior. | Make structural `ComponentLike` canonical and normalize every registered object through one validated private bridge. |
| **must change** | Sending a normalized wrapper rather than the original user object to structural lifecycle hooks | Identity-based state, mocks, and plugin-owned attributes behaved incorrectly. | Call user `initialize`, refresh declarations, then call `ComponentSpec.lifecycle.initialize` exactly once with the original user object; payload, prefill, and validation hooks use that object too. |
| **must change** | Letting data-only construction accept an active step callback | `DataComponent.from_step` silently changed component kind. | Raise `TypeError` with the three supported alternatives. |
| **must change** | Rebuilding static runtime artifacts at several public operations | Lifecycle hooks could repeat, validation could disagree, and setup cost was duplicated. | Prepare one frozen `PreparedCoupling` and reuse it everywhere. |
| **must change** | Allowing direct component configuration changes after preparation | A run could use stale contracts or topology. | Fingerprint normalized and original objects and raise `CouplerError`; supported `Coupler` mutators invalidate safely. |
| **must change** | Branching on the concrete `SurfaceMaskPolicy` type | Custom topology implementations could not receive equivalent behavior. | Adapt every `TopologyPolicy` uniformly and validate patch keys and target-grid shapes. |
| **must change** | Silently sharing duplicate exchange topology keys | Different declarations could overwrite one another's regridder or masks. | Reject duplicates with guidance to merge their field declarations. |
| **must change** | Splitting loop implementations between runner, backend classes, and circular imports | Selection, stepping, cancellation, and return validation drifted. | Put loops and custom-driver adaptation in `backends.py`; keep `runner.py` as selection/delegation only. |
| **must change** | Falling back to step zero or clock start for malformed custom-backend driver calls | Custom workflows could execute the wrong model time without an error. | Require a concrete scalar integer-convertible value, reject booleans/fractions, and require `0 <= step < clock.steps`. |
| **must change** | Performing period output as backend-specific per-step side effects | Compiled runs missed output or put Python I/O inside traced execution. | Precompute cadence boundaries, accumulate in JAX chunks, and transfer/write completed reductions only at host boundaries. |
| **must change** | Loading or configuring optional models during ordinary package import | Core users paid dependency and side-effect costs for unused setups. | Keep factory modules light and defer JCM/Dinosaur, Veros, CREDIT/Torch/TensorFlow imports and environment setup until invocation. |
| **must change** | Mixing four configuration responsibilities | Static controls could be traced and model-specific values could leak into generic runtime contracts. | Use the ownership split documented below: `RuntimeOptions`, `Settings`, `ComponentSpec`, setup dataclasses. |
| **must change** | Shipping test tooling as a runtime dependency and omitting PEP 561 metadata | Normal installs were heavier and external plugins could not reliably type-check. | Move tools to `test`/`dev` extras and ship `vercor/py.typed`; test the installed wheel. |
| **nice to improve** | No automatic plugin discovery | Applications must import and inject plugin objects explicitly. | Keep explicit injection in 3.1; consider entry points only if independent deployments show repeated discovery needs. |
| **nice to improve** | Dataclass validation is hand-written | Complex future configuration schemas could need richer parsing. | Keep frozen dataclasses now; consider Pydantic only when external text configuration becomes a real requirement. |
| **nice to improve** | Forcing indexes, raw interpolators, and periodic endpoint defaults retain historical shapes | Some specialized extension cases remain less uniform. | Evaluate a forcing-index protocol, raw interpolator removal, and endpoint changes for a future breaking release with measured use cases. |
| **nice to improve** | Private modules are numerous | Navigation can be difficult despite focused ownership. | Defer setup extraction and wholesale reorganization; private layout can evolve without changing public contracts. |

## 4. Public API redesign

### Ownership and configuration

The stable core owner modules are `vercor.components`, `vercor.runtime`,
`vercor.topology`, `vercor.coupling`, `vercor.exchanges`,
`vercor.regridding`, `vercor.grids`, `vercor.fields`, `vercor.state`,
`vercor.output`, and `vercor.setups`. `vercor.types`, `vercor.dtypes`, and
`vercor.jax_logging` are supporting public typing/precision/logging modules.
Existing public modules including `vercor.settings`, `vercor.clock`,
`vercor.calendar`, `vercor.recipes`, `vercor.diagnostics`, and the interpolator
facades remain valid; the frozen list is not an instruction to delete them.

Configuration ownership is exact:

- `RuntimeOptions` owns static execution, topology, dtype, model-year, and
  runtime policy.
- `Settings` owns traced physics and component/model constants. It remains a
  mutable setup-time container so components can intentionally update values
  before preparation.
- `ComponentSpec` owns fields, lifecycle, execution capability, and output
  contract.
- `Spinup`, `JAXGCMConfig`, `VerosConfig`, `CAMulatorConfig`, and
  `JCMLandAtmosphereConfig` own model-specific construction only.

The main contracts have these signatures (annotations abbreviated only where a
public type alias is already named):

```python
ComponentStepReturn = Mapping[str, RuntimeArray] | StepResult

class ComponentLike(Protocol):
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec
    def initial_fields(self) -> Mapping[str, RuntimeArray]: ...
    def initialize(self, context: SetupContext) -> None: ...
    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> ComponentStepReturn: ...

Component.from_step(
    name: str,
    grid: RectilinearGrid,
    step: Callable[..., ComponentStepReturn],
    *,
    spec: ComponentSpec | None = None,
    payload: Any | None = None,
    settings: Settings | None = None,
) -> Component

HostComponent.from_step(
    name: str,
    grid: RectilinearGrid,
    step: Callable[..., ComponentStepReturn],
    *,
    spec: ComponentSpec | None = None,
    payload: Any | None = None,
    settings: Settings | None = None,
) -> HostComponent

DataComponent.from_fields(
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object] | None = None,
    *,
    settings: Settings | None = None,
    spec: ComponentSpec | None = None,
    import_policy: FieldImportPolicy | None = None,
) -> DataComponent
```

`DataComponent.from_step(...) -> NoReturn` exists only to produce a focused
error. `ComponentLike` is a static structural `Protocol`, not a promise that
arbitrary objects can be classified with `isinstance`.

`ComponentSpec(inputs=(), outputs=(), defaults=None, *, execution="jax",
lifecycle=None, output=None)` is immutable. The `inputs`/`outputs` sequences are
deduplicated, `defaults` is copied into a read-only mapping, and `execution` is
`"jax"` or `"host"`. `LifecycleHooks(initialize, create_payload, prefill,
validate)` is the only public lifecycle customization path. Structural hook
arguments are the original user object. Public contexts are `SetupContext`,
`StepContext`, `PrefillContext`, `PrefillResult`, and `ValidationContext`; the
last exposes a public `ComponentState`, never a runtime store.

### Execution precedence

Execution selection is deterministic:

1. `ComponentSpec.execution` declares a component capability.
2. `HostComponent.from_step` forces its resulting spec to `"host"`.
3. `RuntimeOptions(execution="auto")` chooses JAX unless any component requires
   host execution.
4. `execution="jax"` rejects host components.
5. `execution="host"` runs every component through the Python host loop.
6. An `ExecutionBackend` object delegates orchestration to
   `backend.run(state, *, context, driver)` and must return `RunState`.

A custom backend may call
`RuntimeDriver.step_component(state, component, *, step)`. The driver validates
the `RunState`, prepared component name, scalar/concrete/integer-convertible
step, and range. Booleans, fractional reals, tracers that cannot materialize,
arrays with dimensions, negative values, and `step >= clock.steps` raise
`CouplerError`; there is no step-zero fallback.

Custom backends currently reject configured period output before invocation
because `ExecutionBackend` has no public period-session hook. Built-in JAX and
host modes share the period session. A period-enabled run also rejects traced
`RunState` leaves and tells differentiated callers to disable output; output-free
runs keep the one-shot differentiable scan.

### Coupling, topology, state, and output

`CouplerSpec` is a frozen reusable recipe; `Coupler` is a configured runtime
session. Their representative signatures are:

```python
CouplerSpec(
    *,
    components: Sequence[ComponentLike],
    exchanges: Sequence[Exchange] = (),
    run_order: Sequence[str] = (),
    runtime: RuntimeOptions | None = None,
)
CouplerSpec.build(clock: Clock, *, logger=None, log_level="INFO") -> Coupler

Coupler(
    clock: Clock,
    *,
    components: Iterable[ComponentLike] = (),
    exchanges: Iterable[Exchange] = (),
    run_order: Sequence[str] = (),
    runtime: RuntimeOptions | None = None,
    logger: LoggerLike | None = None,
    log_level: int | str = "INFO",
)
Coupler.initial_state(*, prefill_missing: bool = True) -> RunState
Coupler.run(state: RunState | None = None) -> RunState
Coupler.write_outputs(
    state: RunState,
    *,
    output_dir: Path = Path("."),
    filename_template: str = "{component}.runtime_fields.nc",
    write_snapshots: bool = True,
) -> None
```

`add_component`, `add_exchange`, `add_exchanges`, and `set_run_order` return the
same `Coupler` and safely invalidate preparation. `Exchange(source, target,
fields, *, regrid=bilinear, label=None)` accepts scalar names or public
`VectorField` declarations. Public `Regridder`/`RegridderFactory` protocols and
the `bilinear`/`conservative` factories make interpolation replaceable without
exposing concrete private wrappers.

`TopologyPolicy.applies(TopologyContext) -> bool` and
`build(TopologyContext) -> ExchangeTopologyPatch` are the complete topology
extension. Patch keys are `(source, target, regrid_key)` and values must match
the target grid. `SurfaceMaskPolicy` is the bundled ATM/OCN/LND implementation,
not a special runtime type branch.

`RunState` is immutable and opaque; direct construction raises `TypeError`.
Use `component(name)`, `components(names=None)`, or immutable
`replace_fields(component, fields)`. `ComponentState` exposes `field`, `fields`,
and `iter_fields` across the public `state`, `received`, and `sent` scopes.

`OutputConfig(snapshot_writer=None, period=None)` belongs on `ComponentSpec`.
Snapshot writers receive `SnapshotContext` with public `ComponentInfo`,
`ComponentState`, payload, path, time, and logger. Generic period output samples
declared runtime fields; an empty `PeriodOutput.variables` uses
`ComponentSpec.outputs`. Session-managed generic, JAXGCM, and current Veros
period files are emitted during `Coupler.run()` in the current working
directory. CAMulator writes native period files beneath its configured
`save_location` hierarchy. `write_outputs(output_dir=...)` writes final runtime
views and registered snapshots; it does not redirect period files.

### Short component and backend example

```python
from collections.abc import Mapping
from datetime import datetime

from vercor import Clock, ComponentSpec, Coupler, RuntimeOptions
from vercor.components import Component
from vercor.grids import RectilinearGrid
from vercor.runtime import ExecutionContext, RunState, RuntimeDriver
from vercor.types import RuntimeArray

grid = RectilinearGrid.uniform(
    "demo", nlon=2, nlat=2,
    longitude=(0.0, 360.0), latitude=(-90.0, 90.0),
)

def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
    return {"temperature": fields["temperature"] + 1.0}

model = Component.from_step(
    "MODEL", grid, step,
    spec=ComponentSpec(outputs=("temperature",), defaults={"temperature": 280.0}),
)

class SequentialBackend:
    def run(
        self, state: RunState, *, context: ExecutionContext, driver: RuntimeDriver
    ) -> RunState:
        for index in range(context.clock.steps):
            for name in context.run_order:
                state = driver.step_component(state, name, step=index)
        return state

coupler = Coupler(
    Clock(datetime(2000, 1, 1), 3600.0, 2),
    components=(model,),
    run_order=("MODEL",),
    runtime=RuntimeOptions(execution=SequentialBackend()),
)
state = coupler.run()
```

Migration from a valid 3.0 component normally requires no source change. Prefer
annotating new callback return types with `ComponentStepReturn`; replace the
invalid `DataComponent.from_step` path with the adapter matching the component
kind.

## 5. Private API redesign

The following internals support the public contracts but are not compatibility
surfaces. Their names, layouts, and signatures may change in any minor release
as long as public behavior remains stable.

| Private owner | Responsibility | Public contract supported |
| --- | --- | --- |
| `vercor.components._adapter` | `_ComponentAdapter`, structural validation, `normalize_component(component)` | `ComponentLike`, original-object lifecycle identity, `Coupler.add_component` |
| `vercor.components._contracts`, `_callable_wrappers`, `_lifecycle_api`, `_runtime_fields`, `_runtime_validation` | Author value normalization, callable arity adaptation, lifecycle dispatch, mapping/store translation, validation | Convenience component constructors and `ComponentStepReturn` |
| `vercor._runtime.prepared` | `PreparedCoupling`, preparation, frozen contracts/topology/dispatch, configuration fingerprints, x64 capability enablement | `Coupler.initial_state`, `run`, `write_outputs` consistency |
| `vercor._runtime.initialization` and `preparation` | Lifecycle initialization, contract/topology construction, state creation, supplied-state validation, sent-store priming | Public setup lifecycle and state entry points |
| `vercor._runtime.contracts`, `stores`, `state`, `component_state`, `field_transfer`, `validation`, `state_validation` | Immutable runtime stores/states and receive/send/shape enforcement | Opaque `RunState` and public `ComponentState` views |
| `vercor._runtime.dispatch_context`, `exchange_dispatch`, `driver` | Destination-grouped exchanges and receive/step/send orchestration | Built-in loops and `RuntimeDriver.step_component` |
| `vercor._runtime.topology_policy`, `exchange_topology`, `surface_masks`, `topology_state`, `topology` | Uniform public-policy adaptation, duplicate/key/shape checks, regridders/masks, frozen maps | `TopologyPolicy`, `SurfaceMaskPolicy`, `ExchangeTopologyPatch` |
| `vercor._runtime.runner` | Mode selection, host compatibility/warning, signal scope, delegation | `RuntimeOptions.execution` precedence |
| `vercor._runtime.backends` | Compiled scan, output-enabled chunks, host loop, custom backend adapter, strict driver validation | JAX/host/custom execution and interrupt propagation |
| `vercor.output._session` | `_PeriodOutputSchema`, immutable `_PeriodOutputAccumulator`/`_PeriodOutputSession`, boundary/filename plans, early field/tracer/backend checks, native-host schema exclusion | Backend-consistent generic/JAXGCM `PeriodOutput` without duplicating native Veros/CAMulator writes |
| `vercor.output._period`, `_dataset`, `_period_files`, `_netcdf` | Sum/count math, calendar coordinates, dataset decoration, file lifecycle, final host transfer | `OutputVariable`, period files, existing NetCDF layout |
| `vercor.output._runtime` | Final runtime-view and registered snapshot orchestration | `Coupler.write_outputs` and `SnapshotWriter` |
| `vercor.setups._lazy_imports` | The one setup lazy-resolution helper | Lightweight `vercor.setups` imports |
| `vercor.setups._data`, `_slab`, `_jcm`, `_external` | Built-in construction, private model payloads/extractors, optional dependency boundaries | Public setup configs and factories |
| `vercor.setups._external.veros_runtime_settings` and `camulator_runtime_settings` | Invocation-time model/environment configuration | Side-effect-free core/setup imports |
| `vercor._regridders` and private interpolator helpers | Concrete implementations and precomputed geometry/weights | Public regridder protocols/factories |

`PreparedCoupling` is built after lifecycle initialization so its fingerprints
capture the effective configuration. It stores read-only component/contract
mappings, tuple exchanges/run order, frozen `RuntimeTopologyMaps`, one
`RuntimeDispatchContext`, runtime controls, and the interrupt controller.
`initial_state`, supplied-state validation, execution, and output all consume
that object; none reconstructs contracts independently.

Precision intentionally follows implemented JAX semantics rather than a fake
per-session global lock. An x64 coupler may enable the process-wide JAX x64
capability before normalization. A float32 coupler continues to request explicit
float32 VerCOR allocations even if that process capability is already enabled.

Output internals keep model state on device. Without period output the runtime
uses the unchanged one-shot compiled scan. For session-managed output, a static
union of step, day, month, and year boundaries splits the scan into pure JAX
chunks. Immutable sum/count sessions cross chunk boundaries; only completed
reductions cross to the file writer. JAXGCM's model-specific schema extracts
native payload values while generic schemas read declared runtime fields.
Veros/CAMulator remain host-only and keep their native step-managed adapter; the
session owner explicitly skips them to prevent validation drift and duplicate
files. Filenames are collision-safe without changing unique historical names.

These pieces remain private because their dataclass fields, grouping strategy,
fingerprinting algorithm, chunking, accumulator representation, and optional
model integrations are implementation choices, not contracts plugin authors
should have to track.

## 6. Setup-agnostic plugin architecture

The recommended mechanism is a structural `Protocol` plus explicit dependency
injection. A user imports their component/backend/topology/config factory and
passes the resulting object to `Coupler` or `CouplerSpec`. This has four useful
properties: normal Python composition, static type checking without inheritance,
simple fakes, and no process-global ordering or discovery state.

Frozen dataclasses are used for `ComponentSpec`, `RuntimeOptions`, topology
patches, output policies, setup configs, and runtime state. `Settings` is the
intentional setup-time mutable exception for traced physics/model constants.
Small factories and adapters reduce boilerplate but do not define an alternate
runtime contract. Dependency injection is direct; there is no global registry,
entry-point discovery, Pydantic dependency, or service container in 3.1.

Supported realistic setups are:

- **Built-in default:** construct slab/data components from `vercor.setups`, or
  invoke an optional JAXGCM, Veros, or CAMulator factory with its config.
- **User JAX model:** implement `ComponentLike` or use
  `Component.from_step`; return mapping updates or `StepResult` with a stable
  payload PyTree.
- **User host model:** declare `ComponentSpec(execution="host")` structurally or
  use `HostComponent.from_step`. Auto execution selects the host loop.
- **User component/config:** keep model configuration in a plugin-owned frozen
  dataclass and have a normal factory return a structural component. VerCOR does
  not need to understand that config format.
- **Alternative backend/workflow:** implement `ExecutionBackend`, iterate the
  public `ExecutionContext.clock` and `run_order`, and call `RuntimeDriver`.
  Returning anything but `RunState` is an error. Period output must be disabled
  until the public backend contract gains an explicit session hook.
- **Alternative topology:** implement `TopologyPolicy` and return masks only
  for keys in `TopologyContext.exchange_keys`; ordinary runs may use
  `topology=None`.
- **Alternative pipeline:** compose `CouplerSpec` recipes, build sessions with
  different clocks, supply a validated initial `RunState`, and use immutable
  `replace_fields` for controlled perturbations.
- **Lifecycle/output:** install `LifecycleHooks` and an `OutputConfig` on the
  component spec. Hooks see public contexts; snapshots never see runtime stores.
- **Testing/mocking:** use a 1x1 or 2x2 `RectilinearGrid`, structural fake
  components, a sequential backend, and an empty topology patch. No private
  imports or monkeypatching of a registry is necessary.
- **Independent packaging:** depend on `vercor>=3.1.0`, import only public owner
  modules, include type information, and test against an installed wheel. The
  repository's `tests/fixtures/public_plugin` is the executable reference.

The tradeoff is deliberate explicitness. Applications own import/discovery and
object lifetime; VerCOR owns validation and execution. Entry-point discovery
would be useful only if deployments need third-party name-based loading without
application imports. Pydantic would be useful only if schema-driven external
configuration becomes central. Neither is required to plug in a model today.

## 7. Compatibility plan

### VerCOR 3.0 to 3.1

All valid 3.0 root exports and the canonical module workflows remain available.
This includes `Coupler`, `CouplerSpec`, component classes/contracts, clocks,
grids, fields, exchanges, runtime options/backends, state views, output
contracts, settings, and the existing setup factory/config names. The root does
not gain aliases for owner-only topology, regridding, setup, or typing symbols.

Intentional error tightening:

- `DataComponent.from_step` raises instead of producing an active component.
- Invalid structural names, grids, specs, methods, initial-field results, and
  execution modes fail during normalization/preparation.
- Direct component configuration mutation after preparation raises
  `CouplerError`; public coupler mutators remain supported.
- Duplicate topology keys, unknown patch keys, wrong mask shapes, forced JAX
  with host components, malformed/out-of-range driver steps, and non-`RunState`
  backend results raise explicit errors.
- Period output rejects traced runs and custom execution backends until their
  public contracts can represent the I/O session.
- Enabled CAMulator spinup is rejected because no implementation exists.

There are no 3.x deprecation shims. Invalid silent behavior is corrected
directly; any future removal of a valid v3 contract waits for 4.0 and requires
an external plugin compatibility matrix plus a migration guide.

Packaging changes are additive for users: pytest/build/lint/type tools live in
extras; `vercor/py.typed` ships in wheel and sdist; CI tests installed Python
3.12/3.13 artifacts in base, JCM, and Veros environments. CAMulator still
requires the separately installed NCAR MILES-CREDIT project and is deliberately
unpinned until a compatible release is verified.

JCM/JAXGCM naming is retained for compatibility. JCM is the model/ecosystem name
used by paired setup helpers and resources; `JAXGCMConfig`/`make_jax_gcm`
identify the existing adapter API. Renaming either family in 3.1 would create
cost without removing architectural coupling, so documentation explains the
relationship instead.

### 2.x -> 3.x migration table

| 2.x path | 3.x path | Note |
| --- | --- | --- |
| `vercor.config.RuntimeOptions` | `vercor.runtime.RuntimeOptions` | Runtime contracts have a canonical public owner. |
| `RuntimeOptions(surface_masks=...)` | `RuntimeOptions(topology=SurfaceMaskPolicy(...))` | Import policies from `vercor.topology`; use `None` for setup-agnostic graphs. |
| Surface mask fields on `Coupler` | A `TopologyPolicy` returning `ExchangeTopologyPatch` | Runtime topology details are private. |
| `vercor.setups.JAXGCMConfig`, `VerosConfig`, `CAMulatorConfig`, and `JCMLandAtmosphereConfig` | Same owner-module imports | These existing setup configuration imports remain stable. |
| `vercor.recipes.CouplerSpec` | `vercor.coupling.CouplerSpec` | Recipes now contain exchange field constants only. |
| `RectilinearGrid.uniform(...)` and `.from_coordinates(...)` | Same constructors | Existing grid construction remains valid; direct coordinates stay keyword-only. |
| Generic component factory helpers | `Component.from_step`, `HostComponent.from_step`, `DataComponent.from_fields` | Pick the execution/data kind explicitly. |
| Runtime stores or component view internals | `RunState.component(name)` and `ComponentState.field(s)` | Runtime state is opaque and immutable. |
| `Coupler.write_outputs(...)` | Same method | The public final-output path remains stable; period files are emitted during `run`. |
| Setup implementation imports | Factories from `vercor.setups` | Private `_data`/`_external` modules are not plugin contracts. |

Release strategy: publish 3.1.0 as a minor release, run the external installed
plugin fixture against the built wheel, and document every error tightening in
release notes. Preserve valid 3.x behavior throughout the series; collect
real-world plugin cases before considering 4.0 changes.

## 8. Final rewritten API

### Complete proposed public API

The proposal below is the implemented 3.1 API, not pseudocode.

- `vercor` keeps its 48-symbol 3.0 convenience surface unchanged:
  `AssetError`, `Coupler`, `CouplerError`, `RunState`, `Component`,
  `ComponentLike`, `ComponentInfo`, `ComponentError`, lifecycle hook aliases,
  `ComponentState`, `CouplerSpec`, `ExecutionBackend`, `ExecutionContext`,
  `FieldImportPolicy`, `LifecycleHooks`, `DataComponent`, `DTypePolicy`,
  `ExchangeError`, `ComponentSpec`, `GridError`, `HostComponent`,
  `KEEP_PAYLOAD`, output types, prefill/validation contexts, `RegridderError`,
  `RuntimeOptions`, `RuntimeDriver`, `Settings`, setup/step contexts,
  `StepResult`, `Clock`, calendar datetime types, `RectilinearGrid`, `Exchange`,
  `VectorField`, and `vector`.
- `vercor.components`: `Component`, `ComponentLike`, `ComponentInfo`,
  `ComponentStepReturn`, `ComponentSpec`, `DataComponent`, `HostComponent`,
  `FieldImportPolicy`, `LifecycleHooks`, the four hook aliases,
  `SetupContext`, `StepContext`, `StepResult`, `KEEP_PAYLOAD`,
  `PrefillContext`, `PrefillResult`, and `ValidationContext`.
- `vercor.runtime`: `RuntimeOptions`, `DTypePolicy`, `ExecutionMode`,
  `ExecutionBackend`, `ExecutionContext`, `RuntimeDriver`, plus compatible
  `RunState` and `ComponentState` reexports.
- `vercor.topology`: `ExchangeKey`, `TopologyContext`, `TopologyPolicy`,
  `ExchangeTopologyPatch`, and `SurfaceMaskPolicy`.
- `vercor.coupling`: `Coupler`, `CouplerSpec`, and the compatible `Exchange`
  reexport.
- `vercor.exchanges`: canonical `Exchange` owner.
- `vercor.regridding`: `Regridder`, `RegridderFactory`, `bilinear`, and
  `conservative`.
- `vercor.grids`: `RectilinearGrid`, including keyword-only direct
  construction, `uniform`, and `from_coordinates`.
- `vercor.fields`: `COMMON_FIELD_NAMES`, `ExchangeField`, `VectorField`, and
  `vector`.
- `vercor.state`: `RunState`, `ComponentState`, `FieldScope`, and
  `FieldLookupScope`.
- `vercor.output`: `OutputConfig`, `OutputFrequency`, `OutputVariable`,
  `PeriodOutput`, `SnapshotContext`, and `SnapshotWriter`.
- `vercor.setups`: `Spinup`, `JAXGCMConfig`, `VerosConfig`,
  `CAMulatorConfig`, `JCMLandAtmosphereConfig`, `JCMInputs`,
  `JCMLandAtmosphereSetup`, `load_jcm_inputs`, paired JCM and four slab
  factories, and lazy CAMulator, ERA5, ERA-Interim, JAXGCM, JCM-land, and Veros
  factories.
- `vercor.types`: `RuntimeArray`.
- `vercor.dtypes`: `DTypePolicy`, `PrecisionPolicy`, shape/policy protocols,
  dtype queries, conversion helpers, and JAX allocation helpers.
- `vercor.jax_logging`: `LoggerLike`, `JaxCallbackLogger`, canonical constants,
  logger configuration/level helpers, host emission, and `setup_logger`.
- `vercor.settings`: `Settings`, `SettingSpec`, and documented default metadata.
- `vercor.clock`/`vercor.calendar`: `Clock` and public calendar values/helpers.
- `vercor.recipes`: the canonical `*_FIELDS` exchange recipe constants.
- `vercor.diagnostics` and public interpolator facades remain supported for
  their existing diagnostic/interpolation workflows.

Primary public relationships are:

```text
ComponentLike / convenience adapters
        -> CouplerSpec (reusable recipe) or Coupler (session)
        -> Prepared public configuration: Exchange + RuntimeOptions
        -> Coupler.initial_state / run
        -> immutable RunState -> ComponentState views
        -> Coupler.write_outputs for final views and snapshots
```

The main usage examples are the structural/backend example in section 4, the
scenario patterns in section 6, and the six runnable public-only snippets in
the repository `README.md`. The independently packaged fixture is the normative
example for external type checking and artifact isolation.

### Complete proposed private API

The implemented private relationship map is:

```text
vercor.components._adapter.normalize_component
    -> normalized Component objects
    -> vercor._runtime.prepared.prepare_coupling
       -> initialization + contracts + topology + dispatch + fingerprints
       -> PreparedCoupling
          -> preparation.create_runtime_state / prepare_runtime_state
          -> runner.run_coupler_runtime
             -> backends.run_compiled_scanned_runtime
             -> backends.run_compiled_period_output_runtime
             -> backends.run_host_runtime / run_host_period_output_runtime
             -> backends.run_custom_backend
          -> output._runtime final views/snapshots

generic/JAXGCM output-enabled built-in run
    -> output._session.build_period_output_plan
    -> _PeriodOutputSchema + _PeriodOutputSession + coalesced boundaries
    -> pure JAX sum/count accumulation inside each backend chunk
    -> output._period_files / output._netcdf at host boundaries

native Veros/CAMulator output-enabled host run
    -> private setup-owned _ComponentOutputAdapter records during model steps
    -> output session recognizes the private ownership marker and skips schemas
    -> shared output._period_files / output._netcdf primitives
```

Actual private owners are the focused `vercor.components._*` author/runtime
bridges; `vercor._runtime` contracts, stores, state, time, validation, topology,
preparation, dispatch, progress, interrupt, runner, backend, and facade modules;
`vercor.output._period`, `_dataset`, `_component_adapter`, `_session`,
`_period_files`, `_netcdf`, and `_runtime`; `vercor._regridders` plus private
interpolator helpers; and setup implementation packages under
`vercor.setups._data`, `_slab`, `_jcm`, and `_external`.

Bundled JAXGCM, Veros, and CAMulator payloads, native field extractors,
coordinates, metadata, spinup, and environment configuration remain private
because their external model versions and native state layouts can evolve
independently of `ComponentLike`. Setup lazy loading is likewise private; users
depend only on the factory/config names in `vercor.setups`.

No internal class or function in this subsection is a supported plugin import.
VerCOR may combine, rename, or reorganize them in 3.x while preserving the
public signatures, lifecycle order, execution precedence, state semantics,
output formats, and valid 3.0 workflows described above.
