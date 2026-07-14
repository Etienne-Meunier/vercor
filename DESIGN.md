# VerCOR: Design Specification

A fully differentiable coupler in JAX for different Earth system models written in JAX.

The evidence-based 3.1 public/private inventory and compatibility baseline are
maintained in
[`docs/api-architecture-review.md`](docs/api-architecture-review.md). This file
records the current implementation design; the complete v4 review rewrite is
deferred to the documentation/release milestone.

---

## 1. Goals and Non-Goals

### Goals

- **End-to-end differentiable**: exact gradients of any output with respect to any
  coupled models parameters, via JAX reverse-mode AD.
- **No global arrays or mutable state**: all data is passed explicitly via function arguments and return values.
- **Accelerated**: pure functions are JIT-compiled with `jax.jit` decorator, 
single device parallelism via `jax.vmap` where applicable, use `jax.lax.scan` for iterative methods/solvers, 
use `jax.lax.fori_loop`, `jax.numpy.where` and  `jax.lax.cond` etc. to avoid Python control flow.
- **Modularity**: clean, modular code structure for easy maintenance and extension.
- **Documentation**: comprehensive docstrings and usage examples.
- **Testing**: extensive unit tests for correctness and regression prevention.

## 2. Architecture Overview

### Modular design

The codebase is organized into modules corresponding to physical, numerical and different coupled models/components.

Interpolation, exchangers, grids, model components, output routines etc. are all separate modules with well-defined interfaces. 
This allows different agents to work on different components in parallel and makes testing easier.

Public interpolation is reached through `vercor.regridding` factories and
regridder protocols. Private helper owners under `vercor._interpolators` keep
lower-level concerns separate:
`_bilinear_geometry` owns spherical geometry and orientation checks,
`_bilinear_weights` owns target-to-source cell lookup and bilinear weights, and
`_bilinear_extrapolation` owns nearest/IDW fill policy plus valid-source mask
normalization. Production code outside `vercor._interpolators` should depend on
the public regridder boundary, not these private helpers. The removed primary
`vercor.interpolators` package is not a compatibility facade.

Core output coordination lives in the private runtime/output packages. Public
providers only sample frames, snapshot writers receive a coordinator-allocated
path, and model-step implementations do not own cadence or period file paths.
Logging is injected through public contexts rather than hidden globals.

### Pure functional style

Differentiable physics and runtime kernels are pure and operate on immutable
PyTrees. Host adapters, optional model setup, logging, interruption handling,
and explicitly requested output are deliberate side-effect boundaries and are
kept outside traced kernels. Not every helper is or should be decorated with
`jax.jit`; the execution backend owns the transform boundary.

### One-shot JIT scanned runtime

The default differentiable workflow uses a one-shot `jax.jit` wrapper around
the scanned runtime. VerCOR does not own a persistent compiled-runtime cache
and does not expose state-buffer donation controls through `Coupler.run()`.
The default output-free workflow remains one JIT-wrapped scan. A custom static
workflow is split only where consecutive component schedules differ, with each
uniform chunk preserving absolute clock indices. One run-local jitted executor
is reused for every chunk with the same component schedule; JAX recompiles that
callable naturally when chunk input shapes differ. Clock steps, JAX step
indices, forcing-selection metadata, and progress messages are built once per
run and sliced for each chunk. Configured period output uses
one-step chunks so the core can sample every post-step state, retain immutable
JAX sum/count accumulators, and perform host writes only at precomputed cadence
boundaries. Model state remains JAX-backed. Because period output is an I/O
workflow, traced `RunState` leaves are rejected when it is enabled, while
output-free runs remain differentiable.
Configuration objects still need stable shapes and PyTree structures because
JAX traces the scanned runtime at the run boundary.

**Keep traced runtime inputs stable:**
- Define model/containers at module top-level so identities are easy to reason
  about.
- Mark non-array metadata as static so it is not traced.
- Keep argument pytrees small and consistent, with fixed fields and stable
  mapping keys.
- Prefer deterministic RNG splitting at the runtime boundary.

**Anti-pattern to avoid:** constructing fresh containers every call with
changing non-array fields, such as dicts with keys that appear or disappear.

### Input / Output

Explicit run output is coordinated by the runtime and implemented by private
`vercor.output` helpers. Computational kernels remain decoupled from NetCDF and
host transfer. VerCOR 4 currently writes NetCDF output; a general restart-file
contract is not part of the v4 alpha API.

JAXGCM, Veros, CAMulator, third-party period output, native snapshots, and final
runtime-view files are written directly with `h5netcdf`, bypassing xarray
conversion so providers preserve calendar timestamps, native coordinates and
metadata, and runtime-field attrs. Passing no `OutputTarget` performs no I/O.
One core coordination path owns provider sampling, uniform variable filtering,
immutable JAX PyTree sum/count accumulation, cadence, collision-safe paths,
snapshots, host transfer, and writes. Model steps and execution backends never
own cadence or file paths. The static boundary plan allocates filenames across
every provider and boundary; repeated same-date records gain deterministic time,
absolute-step, and schema discriminators instead of overwriting files.
Providers sample the post-step state and receive a zero-based step index plus
the end-of-step model time. Period filenames/time coordinates and final
snapshot contexts use that same represented-state time; zero-step snapshots use
the clock start. The first provider frame fixes variable, coordinate, metadata,
sample-axis, time-axis, and dimension-order schema for the run, and later drift
is rejected with a component-scoped error.

Bundled model-specific providers live beside their setup adapters under
`vercor.setups._external` and adapt native model objects to `OutputFrame`.
Runtime-field providers and bundled/third-party native providers use the same
selection and coordinator path. VerCOR-owned samples, accumulators, means, and
runtime-view fields remain JAX-backed until `vercor.output._netcdf` delegates
the final transfer to `vercor._host_arrays`. NetCDF time-coordinate
values intentionally remain host `int64` arrays because the CF microsecond
offsets can overflow JAX integers when `jax_enable_x64` is disabled.

### Data flow: PyTree-based result objects

Shared PyTree mechanics live in private `vercor._pytree.PyTreeNodeMixin`.
Immutable
classes registered with `@jax.tree_util.register_pytree_node_class` should
inherit from the mixin and declare `pytree_children` for traced fields plus
`pytree_aux_data` for static metadata. The mixin reconstructs objects without
rerunning constructors, and classes with derived static attributes can restore
them in `_pytree_post_unflatten()`.

Every module returns an immutable PyTree container, usually a frozen dataclass,
containing arrays and objects.

No mutable state. No side effects.

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RectilinearGrid(PyTreeNodeMixin):
    pytree_children = (
        "longitude",
        "latitude",
        "longitude_edges",
        "latitude_edges",
        "binary_mask",
    )
    pytree_aux_data = ("name",)

    name: str
    longitude: Array
    latitude: Array
    longitude_edges: Array
    latitude_edges: Array
    binary_mask: Array
    ...
```

### Public and runtime API boundary

VerCOR intentionally separates user-facing orchestration objects from the
immutable runtime containers used during traced integration.

- Public orchestration API: the root exports exactly `Clock`, `Coupler`,
  `Exchange`, `RectilinearGrid`, `RunState`, and `RuntimeOptions`. Component,
  runtime-extension, topology, output, regridding, state-view, and setup
  contracts remain in their canonical public owner modules. `Coupler(...)` is
  the sole primary assembly path: it owns immutable snapshots of the component,
  exchange, and run-order collections while retaining the original author
  objects, which are treated as immutable configuration. It exposes read-only
  public views. Reconfiguration requires a new coupler; there are no primary
  mutators or `CouplerSpec`.
  Each `Exchange` owns a stable public `route_id` plus an injected
  `regridder_factory`. The default route ID is `"source->target"`; repeated
  endpoints therefore require explicit distinct IDs, and all route IDs are
  validated globally before lifecycle setup or factory invocation.
  Component-name sequences are normalized internally to immutable tuples.
  An empty run order is explicit setup-only semantics: setup, validation,
  topology, state construction, and output preparation occur, but the runtime
  loop advances no component. Component names are setup-owned identifiers rather than a
  fixed enum, so custom coupled runs can use names outside the bundled
  ATM/OCN/LND/ICE conventions. `RuntimeOptions` owns static coupler runtime
  policy, including dtype, workflow, execution backend, model-year length, and
  optional topology policy. Pass
  `RuntimeOptions(topology=vercor.topology.SurfaceMaskPolicy())` to opt into
  the bundled ATM/OCN/LND surface-mask topology, and leave `topology=None` for
  setup-agnostic custom exchanges. A public `Workflow.build(WorkflowContext)`
  returns one exact ascending `StepPlan` per clock step; each step may reorder
  or omit registered components without duplicating or inventing names. The
  default `SequentialWorkflow` repeats the constructor run order. Public custom
  execution backends implement `vercor.runtime.ExecutionBackend.execute(...)`
  and receive an `ExecutionContext`, core-defined `ExecutionChunk`, and
  `RuntimeDriver`; the private runtime context is not part of the public
  contract. Backends must consume every supplied plan exactly once through
  `RuntimeDriver.run_step(state, plan)` and return `RunState`. Supplied states, every
  state entering the public driver, and backend-returned states are validated
  against the private prepared binding: exact component/store/route names,
  grids and coordinates, array shapes and dtypes, and mask constraints must
  match. The driver rejects forged, reordered, repeated, skipped, or
  out-of-chunk plans before it dispatches the normal receive/step/send pipeline;
  custom orchestration may include host-backed components. Component lifecycle initialization runs for
  any non-empty configured component graph, including custom single-component
  or no-exchange workflows. `vercor.state.RunState` is opaque: users inspect results
  through `RunState.component(name) -> ComponentState` and
  `ComponentState.field(...)`/`fields(...)`, while runtime stores remain
  private. `vercor.grids.RectilinearGrid`, `vercor.fields.VectorField`, and
  `vercor.exchanges.Exchange` are owned by their public facade modules rather
  than private implementation modules. `RectilinearGrid.uniform(...)` builds
  generated equally spaced rectilinear grids, and
  `RectilinearGrid.from_coordinates(...)` builds grids from explicit
  coordinate arrays. Direct `RectilinearGrid(...)` construction requires
  keyword-only coordinate arguments to avoid accidental longitude/latitude
  swaps.
- VerCOR 4 component-author API: `vercor.components.Component` is a
  runtime-checkable structural protocol with `name`, `RectilinearGrid`, frozen
  `ComponentSpec`, and `step(fields, context, payload=None)`. Authors implement
  that protocol directly or compose `CallableComponent`; data-only adapters use
  `DataComponent`. `ComponentSpec` exclusively owns declared `inputs`,
  `outputs`, immutable `initial_fields`, the `execution` capability,
  `LifecycleHooks`, `TransferPolicy`, and `OutputSpec`. Scalar author values
  expand on the component grid and every prepared field is normalized once to
  the runtime dtype. `LifecycleHooks.setup` runs once against the original
  author object and may return `SetupResult(fields, payload)`; prefill and
  validation use typed immutable contexts/results. Runtime payloads and NumPy
  author arrays are defensively copy-owned at preparation. Standard payload
  containers are rebuilt per runtime state, NumPy leaves are copied, and opaque
  object leaves are deep-copied or rejected if they cannot be owned.
  `StepResult()`
  preserves payload, an explicit payload replaces it, and host execution may
  clear or restructure it; compiled JAX execution requires replacements to
  retain the setup payload's PyTree structure. One private
  declaration-to-binding adapter owns setup, normalization, and runtime hook
  dispatch. There is no primary-v4 `ComponentLike`, `HostComponent`, component
  `initialize`/`initial_fields`, constructor payload, mutable authoring mixin,
  separate payload-creation hook, or duplicate output/import-policy property.
- Internal runtime API: the `vercor._runtime` package owns
  `FieldStore`, `ComponentRuntimeState`, runtime contexts, dispatch
  contexts, and runtime helper functions. Public `RunState` is owned by
  `vercor.state` while carrying immutable runtime component states and static
  metadata through JAX tracing. These containers are required for
  differentiability and stable scan carry structure. Runtime
  field stores live in `vercor._runtime.stores` and own name membership, mapping
  roundtrips, fallback reads, and replacement of existing fields while
  preserving established dtypes. Receive/send contract construction lives in
  `vercor._runtime.contracts`, exchange dispatch lives in
  `vercor._runtime.exchange_dispatch`, static dispatch context construction
  lives in `vercor._runtime.dispatch_context`, which stores exchanges grouped by
  destination so per-component dispatch receives destination-specific work,
  runtime step metadata lives in `vercor._runtime.time`, component state
  creation lives in `vercor._runtime.component_state`, field receive/send mechanics live in
  `vercor._runtime.field_transfer`, component/store runtime validation lives in
  `vercor._runtime.validation`, and configured runtime-state/topology validation
  lives in `vercor._runtime.state_validation`. Runtime coupler-state assembly
  and runtime-contract refresh live in `vercor._runtime.coupler_state`; exchange
  surface-role lookup lives in `vercor._runtime.component_topology`; runtime
  topology data contracts live in
  `vercor._runtime.topology_state`, including grouped `RuntimeTopologyMaps`,
  whose copied mappings are read-only, and `ExchangeTopologyState`. Generic exchange
  regridder/identity-mask map construction lives in
  `vercor._runtime.exchange_topology`; public topology policies are adapted by
  `vercor._runtime.topology_policy` through the single uniform
  `build(context)` protocol. Policy patches must use configured route IDs and
  target-grid shapes. Duplicate route IDs are rejected rather than silently
  sharing a regridder. Scalar routes require the public `Regridder` capability;
  vector routes require `VectorRegridder`, and mixed routes require both.
  Optional atmosphere/ocean/land
  surface-mask creation and validation live in
  `vercor._runtime.surface_masks` behind `vercor.topology.SurfaceMaskPolicy`.
  Derived surface-mask values remain local to policy construction rather than
  retained as runtime state.
  Runtime exchange
  validation checks that exchanged fields are declared by the sending and
  receiving components instead of requiring the advisory common field
  vocabulary. `vercor._runtime.topology` remains the orchestration boundary
  that composes those owners and returns the explicit topology state for the
  runtime facade to store. Public `RunState` carries private component
  alignment metadata, component states, and fractional masks through
  `jax.lax.scan`; binary masks
  remain in `RuntimeTopologyMaps` for final output and topology bookkeeping.
  Setup-time component precision synchronization, initialization context construction, component
  setup validation, runtime contract validation, and topology handoff live in
  `vercor._runtime.initialization`. `vercor._runtime.prepared.PreparedCoupling`
  is the single frozen post-lifecycle boundary for normalized components,
  exchanges, run order, contracts, topology maps, destination-grouped dispatch,
  clock, runtime-normalized physical constants, runtime options, and interrupts.
  It contains no reflective configuration fingerprint or public prepared graph.
  Author objects are treated as immutable configuration after `Coupler`
  construction; callers create a new coupler for changed configuration rather
  than mutating or rebuilding preparation. Runtime state creation, supplied-state
  validation, and initial sent-store priming live in
  `vercor._runtime.preparation` and reuse the prepared contracts and dispatch
  context without rebuilding either.
  Frozen `RuntimeRunContext` execution inputs live in
  `vercor._runtime.run_context`; it carries only static execution metadata and
  shared runtime controllers, not compiled-runtime cache state.
  Shared host/scanned progress messages plus traced callbacks live in
  `vercor._runtime.progress`, and the interrupt controller lives in
  `vercor._runtime.interrupts`. JAX process x64 is treated as a capability:
  an x64 Coupler may enable it before component arrays are normalized, while a
  float32 Coupler keeps explicit float32 VerCOR allocations even when the
  process capability is already enabled.
  Compiled scanned execution, the Python host loop, custom backend adaptation,
  and validated public driver adaptation live in `vercor._runtime.backends`.
  `vercor._runtime.execution` owns workflow validation, schedule chunking,
  scheduled-component backend selection, host compatibility warnings,
  per-chunk result validation, and core-owned period-output boundaries.
  `vercor._runtime.runner` owns the interrupt signal scope and delegates plan
  construction and execution to that coordinator. High-level runtime orchestration
  for the public `Coupler` facade lives in `vercor._runtime.facade`: prepared
  runtime-state reuse, run-context construction, host/scanned execution,
  runtime views, and final output delegation enter through this module instead
  of direct `Coupler` imports of runtime implementation helpers.
  Runtime component metadata and read-only field resolution for
  diagnostics/output live in `vercor.state`. `ComponentState.field(...)`,
  `ComponentState.fields(...)`, and `ComponentState.iter_fields(...)` own
  explicit state/received/sent views while diagnostics keep any candidate-order
  selection private. Private helpers keep runtime containers out of public
  diagnostics/output APIs. `Coupler` exposes `initial_state()` and `run()` for
  runtime-state creation.
  `RunState.component(...)` and `RunState.components(...)` are the public
  component-view factories; `RunState.replace_fields(...)` is the sole public
  immutable state-update operation.
  Final runtime output iteration, output-mask naming/selection, and view
  writing live in private `vercor.output._runtime` helpers. `Coupler.run`
  accepts one optional `OutputTarget`; `output=None` and an all-disabled target
  perform no I/O. The private runtime facade validates enabled output runs,
  rejects traced states before providers or writers run, and coordinates period,
  final-field, and snapshot work across the private output path. `Coupler.write_outputs()`
  and `Coupler.finalize()` are absent from the primary v4 contract.
  The `vercor._runtime` package initializer does not reexport runtime containers
  or helper functions; internal code should import from the focused owner
  modules listed above.
  payload pytrees carried through `jax.lax.scan` must preserve every leaf's
  shape and dtype between input and output; per-step slices or adapted forcing
  objects should be local values unless they are shape-stable runtime state.
  Internal runtime containers are not exported from the package top level.
  Public `RunState`/`ComponentState` are owned by `vercor.state`.

### Setup adapters and shared ownership

Reusable concrete adapters live under the canonical packaged namespace
`vercor.setups`. Runnable assembly scripts live under `examples/`; in-repo code
should not depend on a top-level `setups` package. Setup adapters use
`SetupContext`, `StepContext`, and plain runtime-array mappings at their author
boundary instead of importing runtime context/store internals directly.
The public setup facade exports concrete factory functions, setup-specific
configuration dataclasses (`Spinup`, `JAXGCMConfig`, `VerosConfig`,
`CAMulatorConfig`, and `JCMLandAtmosphereConfig`), and the
`JCMInputs`/`load_jcm_inputs(...)` loader for reusable JCM coordinate, terrain,
and forcing inputs. The root package stays core-only and does not reexport
bundled setup configuration. Setup subpackages no longer advertise lazy module
objects in their `__all__` lists or maintain parallel lazy registries:
`vercor.setups` is the sole lazy export table. Resolving a factory attribute
loads only its lightweight factory module. JCM/Dinosaur, Veros, and
CREDIT/Torch/TensorFlow imports plus Veros/CAMulator runtime configuration begin
only when that factory is invoked. Deep adapter modules live under underscore
packages for package-internal tests and optional-dependency boundaries, but
supported user workflows enter through `vercor.setups`.
Examples and setup factories assemble complete runs through `Coupler(...)`
with direct `Exchange(source, target, fields,
regridder_factory=...)` declarations. Shared exchange
field recipes live in `vercor.recipes` with `*_FIELDS` names. Short recipe aliases and setup orchestration helpers
such as `ExchangeSpec`, `build_coupler()`, `build_exchanges()`, and
`add_exchange_specs()` have been removed. Public exchange configuration types,
including `ExchangeField`, are owned by `vercor.fields`; `vercor.exchanges`
exports only the public `Exchange` class. Public regridding protocols,
including `Regridder` and `RegridderFactory`, are owned by
`vercor.regridding`; concrete
bilinear/conservative regridder classes remain private implementation details.
`Regridder` exposes scalar `regrid(field)` plus source/target grid metadata;
`VectorRegridder` extends that capability with `regrid_vector(u, v)`.
Regridders are not callable.

Core helper ownership follows the same boundary. Calendar constants,
model-calendar datetime values, leap-year logic, and month/day conversion live
in `vercor.calendar`. Daily forcing-index policy, including noleap and 360-day
calendar mapping to forcing-file day indexes, lives in `vercor.forcing_index`.
The common exchange field vocabulary lives in `vercor.fields` as the advisory
`COMMON_FIELD_NAMES`; custom field names are valid when declared through
`ComponentSpec` or seeded component fields.
Rectilinear grid construction, center-to-edge geometry, and grid identity checks
live in `vercor.grid_geometry`; mask math lives in `vercor.grid_masks`, while
surface-role lookup for the optional built-in surface-mask policy is private to
`vercor._runtime.component_topology`. Generic hybrid/sigma-coordinate pressure
and altitude helpers live in `vercor.fluxes.vertical_coordinates`, and generic
JAXGCM PyTree transforms live beside that adapter in private
`vercor.setups._external._jax_gcm_pytree`. Flux helper modules keep local JAX
array normalization helpers explicitly named as JAX conversion boundaries so
they are not confused with host-array or NumPy transfer helpers.
Adapter-specific runtime and file-output policy lives beside adapters in focused
helpers instead of in factory/bootstrap modules. Exported adapter helpers use
plain package-internal names in their owner modules; underscored helpers
remain local implementation details and are not listed in external adapter
`__all__` exports. User code reaches these adapters through the
`vercor.setups` factory facade. JAXGCM runtime payload, prefill, validation, stepping, and
host recording live in
`vercor.setups._external.jax_gcm_runtime`, which consumes the setup object through
concrete setup-state annotations rather than a duplicate local protocol.
`vercor.setups._external.jax_gcm_state` owns JAXGCM setup-time model resources,
spinup policy, initialization, and the canonical `JCMState` bundle. JAXGCM and
Veros spinup are controlled only by `Spinup.enabled`, independent of component
names, run order, or whether a counterpart component is present.
`vercor.setups._external.jax_gcm` remains a private factory implementation that
constructs setup state and binds runtime-owned lifecycle hooks directly without
reexporting state bundles or owning runtime payload/setup-state internals.
JAXGCM output extraction, coordinate adaptation, and unit metadata live in
`vercor.setups._external.jax_gcm_output`. JAXGCM, Veros, and CAMulator factories
install ordinary `OutputProvider` objects that return `OutputFrame` samples;
their setup/runtime adapters retain only the latest native state needed for
sampling and never own cadence, output paths, or period writes. Final native
snapshots are registered through `OutputSpec.snapshot_writer`; JAXGCM samples
the final runtime payload's `JCMState`, while Veros and CAMulator sample their
latest native host state.

The stable `vercor.output` contract is `OutputContext`, `OutputFrame`,
`OutputProvider`, `OutputSpec`, `OutputTarget`, `OutputVariable`, `PeriodOutput`,
`SnapshotContext`, and `SnapshotWriter`. `OutputSpec.period is None` disables
period sampling for that component. `PeriodOutput.variables` is applied after
every provider sample with identical empty/subset/unknown/duplicate semantics
for runtime fields, bundled native models, and third-party providers. Custom
`ExecutionBackend` objects use the same core-owned sampling, immutable
sum/count accumulation, cadence, and write boundaries as built-in backends;
they receive no output-session hook. Snapshot writers receive only public
component/state metadata plus the payload.

`vercor.output._session` owns the sole immutable period accumulator and
run-local output plan/session. `vercor.output._period` owns finite-value mean
math and cadence predicates, `vercor.output._dataset` owns shared coordinate
helpers, `vercor.output._runtime` owns final runtime views and snapshots, and
`vercor.output._netcdf` is the sole host/file write primitive. There is no
mutable native accumulator, hidden prepared-schema marker, component adapter,
or second period-file lifecycle.
Surface-temperature cleanup and output-field mapping live in
`vercor.setups._external.jax_gcm_fields`. Veros host-runtime flux application and
substep orchestration live in `vercor.setups._external.veros_runtime` with
concrete setup-state annotations.
`vercor.setups._external.veros_gcm_state` owns Veros setup-time model resources,
spinup policy, grid derivation, lifecycle callbacks, and the latest native host
state used for sampling, while `vercor.setups._external.veros_gcm` remains the
private factory implementation. Native Veros variable metadata, ghost-cell
removal, coordinates, and spatial-axis ordering live in
`vercor.setups._external.veros_output` behind an ordinary provider. It returns
frames only; the shared coordinator applies variable selection, accumulation,
cadence, paths, host transfer, and writes. Final Veros snapshots reuse the same
native extraction helpers through `OutputSpec.snapshot_writer`.
Veros host-state mutation helpers and the named tuple-compatible
`VerosForcingFields` container live in `vercor.setups._external.veros_state`.
Veros backend settings are imported only inside the explicit configuration
function called once by the invoked factory, before its private implementation
loader runs. Importing `veros_output`, `veros_fluxes`, or `veros_state` never
configures the runtime. CAMulator
prediction-block and runtime step orchestration live in
`vercor.setups._external.camulator_runtime` with concrete setup-state
annotations, with tensor staging in
`vercor.setups._external.camulator_tensors` and field mapping in
`vercor.setups._external.camulator_fields`. CAMulator wind
artifact filtering keeps public configuration and log-and-skip failure policy in
`vercor.setups._external.camulator_wind_filter`, while private PyTorch
mask/kernel construction and selected tensor mutation live in
`vercor.setups._external._camulator_wind_filtering`.
`vercor.setups._external.camulator_gcm_state` owns CAMulator atmosphere
setup-time model resources, timestep alignment, field seeding, and lifecycle
callbacks, while `vercor.setups._external.camulator` remains the thin public
factory. CAMulator runtime environment defaults are applied at that invoked
factory boundary before heavy imports, and enabled CAMulator spinup is rejected
there because no spinup path is implemented. Public external setup factories group spinup and period-output options
as `Spinup` and `PeriodOutput` instead of parallel keyword bundles. The paired
JCM land/atmosphere helper takes a single `JCMLandAtmosphereConfig`, whose
`atmosphere` field carries the `JAXGCMConfig`; legacy parallel JCM setup
keywords are not public API. It uses `dataclasses.replace` so generated forcing
replaces only a missing forcing value while preserving caller config subclasses
and explicit forcing objects. The ERA5/JCM example accepts injected ocean,
JCM-input, and clock objects and provides short-run and initial-state-only CLI
modes without changing its default workflow.
CAMulator retains the latest native prediction block for its ordinary output
provider and final snapshot writer. Tensor reshaping, metadata handling,
configured native-variable extraction, and coordinate adaptation live in
`vercor.setups._external.camulator_output`; the adapter does not choose cadence,
construct paths, accumulate means, or write period files.

### Distribution and external plugin boundary

Runtime metadata excludes test tooling; `test` and `dev` extras own pytest,
formatting, lint, typing, and build dependencies. `vercor/py.typed` marks the
wheel and sdist as PEP 561 typed packages. The independently packaged fixture
under `tests/fixtures/public_plugin` imports only public VerCOR modules and
exercises structural JAX/host components, original-object lifecycle hooks, a
custom sequential backend, a custom topology policy, and snapshot output
against an installed wheel. CI builds the VerCOR wheel/sdist plus the
independently packaged public-plugin wheel once, then tests installed base, JCM,
and Veros environments on Python 3.12 and 3.13 by consuming those downloaded
artifacts rather than invoking build tooling in each matrix cell.
Local verification uses the same three-artifact bundle, with an offline build
fallback only when artifact paths are not supplied. Setup subprocess probes and
public-plugin mypy run from the installed site-packages root or an external
temporary use site, so checkout sources cannot satisfy an installed-artifact
check. CAMulator is omitted from the install matrix until a compatible NCAR
MILES-CREDIT release is verified.

`vercor.assets` owns generic cache, download, and checksum validation only, with
asset-specific registries and product vocabulary kept outside the generic cache
layer. Concrete forcing product registries and `get_forcing_data(...)` defaults
live with setup data adapters in `vercor.setups._data.assets`. `vercor.forcing_data`
owns the NetCDF forcing-variable read boundary, including mapping-key
resolution, variable lookup, file-to-runtime axis transpose, and optional
latitude-axis flip.
Diagnostics are split into `vercor.diagnostics.fields`, `vercor.diagnostics.tables`, and
`vercor.diagnostics.plotting`, with field lookup delegated to `vercor.state`
and `vercor.diagnostics` preserving the public reexport surface.

CAMulator runtime field contracts, optional CREDIT/postblock loading, forcing cursors,
tensor accessors, runtime stepping, output, wind filtering, land forcing, and
initialization are split across
`vercor.setups._external.camulator_contracts`,
`vercor.setups._external.camulator_imports`,
`vercor.setups._external.camulator_forcing`,
`vercor.setups._external.camulator_tensors`,
`vercor.setups._external._camulator_wind_filtering`,
`vercor.setups._external.camulator_stepper`,
`vercor.setups._external.camulator_runtime`,
`vercor.setups._external.camulator_output`,
`vercor.setups._external.camulator_wind_filter`,
`vercor.setups._external.camulator_land`, and
`vercor.setups._external.camulator_init`. Runtime adapter code may import these
private focused modules directly; user code should use the public
`vercor.setups` factory facade. The old one-hop CAMulator state and wind-filter
facades have been removed. CAMulator tensor channel metadata is stored internally as
typed `TensorVariableIndex` values in `camulator_tensors`, while
`StateVariableAccessor.get_var_index(...)` is the canonical metadata lookup for
callers that inspect tensor channels. CAMulator wind-filter configuration
validates with explicit exceptions and avoids mutable function defaults so tests
and callers see stable failure modes independent of Python optimization
settings. Low-level wind mask artifacts and in-place tensor filtering stay
behind the private wind-filtering owner; public setup factories and examples
should not import that private module directly.

### Configuration ownership

The primary v4 API has no `Settings` container and no
`vercor.physical_constants` module. `vercor.physics.PhysicalConstants` owns
the frozen traced SI values passed through setup and step contexts;
`RuntimeOptions.dtype` is the sole runtime precision owner; `ComponentSpec`
owns component fields, lifecycle, transfer, execution capability, and current
output policy; setup- or plugin-specific options belong in frozen dataclasses
owned by that setup or plugin. Native third-party objects such as Veros may
retain their own `.settings`; those are adapter state, not VerCOR's public
configuration contract.

### Precision and dtype policy

VerCOR-owned array dtypes are centralized in `vercor.dtypes`. Real-valued JAX
and NumPy arrays use `RuntimeOptions.dtype`: `enable_x64=False` maps to 32-bit
real arrays and `enable_x64=True` maps to 64-bit real arrays. Runtime
preparation casts physical constants and component-owned grid/data arrays at
one boundary before state creation. Helpers without an explicit runtime policy
follow the active JAX `jax_enable_x64` configuration; conversion helpers
preserve already-typed real arrays when no policy is supplied.
Integer/index arrays use the canonical 32-bit index dtype in both
real-precision modes to keep sparse metadata and interpolation indices compact.

Production kernels and adapters should use the dtype helpers rather than
hard-coded `jnp.float64`, `jnp.float32`, `jnp.float_`, `jnp.int64`, or
`jnp.int32` annotations. At host boundaries that require NumPy dtype objects,
derive them explicitly with `np.dtype(jax_real_dtype(policy))` or
`np.dtype(jax_index_dtype(policy))`. NumPy remains restricted to explicit host
and dtype boundaries. Output providers keep VerCOR-owned values JAX-backed and
return immutable `OutputFrame` samples. `vercor.output._netcdf` calls
`vercor._host_arrays` only when a non-JAX consumer such as `h5netcdf` requires
a host array.

### Logging across JAX runtime transforms

The coupler logger is callback-backed through `jax.debug.callback`, so runtime
hooks can emit diagnostics inside `jax.lax.scan`, `jax.jit`, and automatic
differentiation transforms. Coupler logging levels are configured at
instantiation with `Coupler(..., log_level=...)`; disabled levels are filtered
before callbacks enter the traced graph. Runtime hooks should pass traced values
as logger arguments, for example `logger.info("Mean SST: {}", jnp.mean(sst))`,
instead of converting tracers with `float(...)` or `int(...)`.
`vercor.jax_logging` is the public logging facade for logging contracts,
constants, setup helpers, host emission, and the callback-backed logger. The
implementation is split across private `vercor._logging` owner modules:
`config` owns canonical Python logger configuration, `protocols` owns
logger-like contracts and level checks, `host` owns host-side formatting and
emission, and `callback` owns traced-value partitioning plus
`JaxCallbackLogger`. Production code outside the facade should import from
`vercor.jax_logging`, not from `vercor._logging`.
Initialization, runtime, and output helpers that are reached outside a
coupler context use the default `VerCOR` Python logger from
`vercor.jax_logging.get_default_logger()`. Helpers reached from
`Coupler.initial_state()`, `Coupler.run()`, or
component runtime contexts receive the coupler logger explicitly instead of
writing directly to stdout.
The host and scanned coupler runtime paths share progress formatting and traced
callback helpers in `vercor._runtime.progress`. The scanned path precomputes
datetime and timestep labels on the host, then selects the per-step label inside
ordered callbacks so progress logging remains traceable without putting Python
datetime objects in the scan carry.
When `Coupler.run()` selects the Python host runtime because one or more
host-backed components are present, VerCOR emits one warning before the runtime
loop starts. The warning names the host-backed components so users can see why
the full coupled loop is not differentiable.

### Runtime interruption across host and scanned integrations

`Coupler.run()` provides an internal runtime interrupt controller to
`vercor._runtime.runner`, which owns the signal scope, while the execution
coordinator, public driver adapter, and scanned/custom backend boundaries own
runtime cancellation checkpoints. During a run, `SIGINT`, `SIGTERM`, and `SIGTSTP` request graceful
runtime cancellation and are restored to their previous handlers when the run
exits. The host runtime checks the controller at step and component boundaries.
The controller also installs a temporary nonblocking wakeup fd so signals
delivered while the main thread is inside a compiled XLA call are recorded
before Python signal handlers run. The JIT-scanned runtime inserts explicit
ordered `jax.debug.callback` checkpoints at the same boundaries; those callbacks
drain the wakeup fd and observe terminal shortcut commands independently of
logging level. Interrupt callback failures are translated back to a
`KeyboardInterrupt` subclass, while unrelated JAX runtime failures are
preserved.

---

## 3. Module Specifications

### 3.1 Physical and runtime configuration

**File**: `vercor/physics.py`

`PhysicalConstants` is the frozen JAX PyTree of traced SI-valued physical
constants. Setup and step contexts receive this canonical object; there is no
parallel parameter container or primary `physical_constants` module.

**Files**: `vercor/runtime/__init__.py` and `vercor/dtypes.py`

`RuntimeOptions` owns static backend, workflow, topology, model-year, and
precision policy. The public runtime module also owns the frozen planning
containers and workflow/backend/driver protocols. `DTypePolicy` is owned by `vercor.dtypes`, and
`RuntimeOptions.dtype` is the sole runtime precision selection.

**Files**: `vercor/components/contracts.py` and `vercor/setups/config.py`

`ComponentSpec` owns one component's fields, lifecycle, transfer, execution
capability, and current output declaration. Bundled setup configuration is
owned by frozen setup-specific dataclasses; third-party plugins use their own
frozen dataclasses and inject the constructed components explicitly.

This separation is deliberate: JAX traces the children of
`PhysicalConstants`, while runtime policy and component/setup structure remain
static. Physics kernels must not branch in Python on traced physical values.

## 4. Validation and Testing

### Test design philosophy

The test harness is the most important part of this project. Without high-quality
tests, autonomous agents will solve the wrong problem.

1. **Tests must be nearly perfect.** Agents will optimize for whatever the tests
   measure. If a test is wrong or has loose tolerances, agents will produce code
   that passes the bad test but gives wrong physics. Invest more time in the test
   harness than in the code it tests.

2. **Tests must give concise, actionable feedback.** Print the max relative error
   and where it occurs, not full arrays. Pre-compute aggregate statistics.
   Log details to files, not stdout, to avoid context window pollution.

3. **Tests must be fast by default.** Every test file supports a `--fast` mode
   (~10% subsample) for rapid iteration. Full validation runs before commits.

4. **Tests must decompose monolithic tasks.** Test sub-components independently:
   - Regridding with mock meshes and fields
   - Different clock functionalities and options
   - Coupler stepping with mock models and fields
   - Fluxes computations with mock meshes and fields
   - Exchanges of fields between models with mock models, meshes and fields
   - Input / Output
   - etc.
   This lets different agents work on different subsystems.

5. **Tests must enable bisection.** When solution disagrees, we need to find the
   first module in the pipeline that diverges from original code. The test suite
   should make this easy by testing every intermediate quantity, not just
   the final output. This is the "oracle bisection" pattern.

### Test hierarchy

We use a layered testing approach, from unit tests to full pipeline validation.

**level 1**: Unit tests (fast)

**level 2**: Module tests.
For each module, pre-generate reference data and check agreement.

**level 3**: Gradient tests

For each module, verify that AD gradients match finite-difference gradients.

**level 5**: End-to-end integration tests (from runnable scripts in `examples/`)

---

## 5. Performance Strategy

### JIT compilation

The default output-free JAX workflow executes the complete clock through one
JIT-compiled scan. Custom workflows reuse one run-local jitted scan executor
per distinct component schedule, including across output cadence boundaries.
`RunState` and traced physical values are array-bearing PyTree inputs;
`RuntimeOptions`, component declarations, routing, and other shape-controlling
configuration remain static in the private prepared coupling. Configuration
must therefore keep stable names, shapes, dtypes, and payload PyTree structure
for the duration of a coupler. A changed configuration requires construction
of a new `Coupler` and may trigger a new compilation.
