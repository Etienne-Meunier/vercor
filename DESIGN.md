# VerCOR: Design Specification

A fully differentiable coupler in JAX for different Earth system models written in JAX.

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

For bilinear rectilinear interpolation, `BilinearRectilinearInterpolator` is the
public PyTree and user-facing facade. Private helper owners under
`vercor.interpolators` keep lower-level concerns separate:
`_bilinear_geometry` owns spherical geometry and orientation checks,
`_bilinear_weights` owns target-to-source cell lookup and bilinear weights, and
`_bilinear_extrapolation` owns nearest/IDW fill policy plus valid-source mask
normalization. Production code outside `vercor.interpolators` should depend on
the public interpolator or regridder boundary, not these private helpers.

The output module handles all data saving and logging, ensuring a clean separation between computation and I/O.

### Pure functional style

All functions are pure, jitted with `jax.jit` decorator and stateless. No mutable global state. No side effects.
This is critical for JAX compatibility and makes reasoning about the code easier.
Each function takes explicit inputs and returns explicit outputs, which can be easily tested and debugged.

### One-shot JIT scanned runtime

Pure differentiable coupled runs use a one-shot `jax.jit` wrapper around the
scanned runtime. VerCOR does not own a persistent compiled-runtime cache and
does not expose state-buffer donation controls through `Coupler.run()`.
That single-scan path remains unchanged when no component configures period
output. Configured period output precomputes all component cadence boundaries,
coalesces them into ordered scan chunks, and carries immutable JAX sum/count
accumulators between chunks. Completed reductions cross to the host writer only
between chunks; model state remains JAX-backed. Because period output is an I/O
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

All I/O is handled by a dedicated module that reads/writes from/to disk.
The core computational modules are completely decoupled from file formats and storage details.
This allows us to easily swap out the I/O layer if needed, and keeps the core logic clean and focused on the physics.

The output is done in a structured format, such as NetCDF, HDF5, that can be easily read by visualization tools and post-processing scripts.

Model restart files are supported and written in compact HDF5 format using `h5py`.

Current example output snapshots are also written in HDF5. JAXGCM, Veros, and
configured CAMulator averaged period outputs, external-component native
snapshots, CAMulator forecast increments, and final runtime-view NetCDF files
are written directly with `h5netcdf`,
bypassing xarray conversion so adapters can preserve VerCOR calendar
timestamps, shape-derived JCM coordinates, native Veros/CAMulator metadata, and
runtime field attrs. Shared
period-output adapter state, record/write orchestration, cadence, calendar time
encoding, dataset coordinate helpers, accumulation, variable containers,
mean-output conversion, single-record snapshot storage, period-file write
lifecycle, and NetCDF writing live in `vercor.output`;
private `vercor.output._session` owns backend-neutral static output schemas,
immutable JAX PyTree sessions/accumulators, generic runtime-field extraction,
early selected-field validation, coalesced clock boundaries, and host-boundary
writes. Generic schemas default an empty `PeriodOutput.variables` selection to
declared outputs and write `{component}.averages.YYYY-MM-DD.nc`. The static
boundary plan allocates filenames globally across every schema and boundary.
A path requested once remains unchanged; repeated records from one schema use a
deterministic time plus absolute-step discriminator, while paths shared by
multiple schemas also use a path-safe component token plus stable schema index.
Thus unique date-only and existing sub-daily names remain compatible without
allowing one schema to overwrite another. Non-grid dimensions use stable
variable-qualified names because NetCDF dimensions are dataset-global, while
`nlat` and `nlon` remain shared grid axes.
model-specific output helpers live beside their setup adapters in
`vercor.setups._external` and adapt native model objects into that shared output
boundary. Setup-state constructors instantiate the private
`_ComponentOutputAdapter` directly from model-specific output constants and
helpers; output modules do not keep one-case adapter factories.
VerCOR-owned period output samples, accumulators, extracted variables, mean
variables, and runtime-view fields stay JAX-backed until the file boundary;
`vercor.host_arrays` owns the final host transfer. NetCDF time-coordinate
values intentionally remain host `int64` arrays because the CF microsecond
offsets can overflow JAX integers when `jax_enable_x64` is disabled.

### Data flow: PyTree-based result objects

Shared PyTree mechanics live in `vercor.pytree.PyTreeNodeMixin`. Immutable
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

- Public orchestration API: `Coupler`, `Exchange`, `Clock`, grids, regridders,
  and bundled concrete components are the objects users compose when
  configuring a coupled run. `Coupler` and setup helpers accept plain
  component-name sequences for run order and normalize them internally to
  immutable tuples. Component names are setup-owned identifiers rather than a
  fixed enum, so custom coupled runs can use names outside the bundled
  ATM/OCN/LND/ICE conventions. `RuntimeOptions` owns static coupler runtime
  policy, including dtype, execution backend, model-year length, and optional
  topology policy. Pass
  `RuntimeOptions(topology=vercor.topology.SurfaceMaskPolicy())` to opt into
  the bundled ATM/OCN/LND surface-mask topology, and leave `topology=None` for
  setup-agnostic custom exchanges. Public custom execution backends implement
  `vercor.runtime.ExecutionBackend` and receive an `ExecutionContext` plus
  `RuntimeDriver`; the private runtime context is not part of the public
  contract. Custom backends must return `RunState`. The driver validates the
  state, prepared component name, and concrete in-range scalar step before it
  dispatches the normal receive/step/send pipeline; custom orchestration may
  include host-backed components. Component lifecycle initialization runs for
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
- Component-author API: `Component`, `DataComponent`, and `HostComponent` are
  the stable extension points. Custom adapters should use the class-level
  authoring constructors where possible: `DataComponent.from_fields()` for
  data-only fields, `Component.from_step()` for pure callable JAX models, and
  `HostComponent.from_step()` for Python host-side models. These constructors
  declare fields through `spec=ComponentSpec(...)`: `inputs` declare fields the
  model reads, `outputs` declare fields the model writes, `defaults` declare
  concrete runtime defaults for fields the model reads or updates, and
  `execution` declares whether the component runs through the differentiable
  JAX path or the Python host path. Scalar default and seeded values expand to
  grid-shaped constants.
  `SetupContext` and `StepContext` are public setup and step context payloads
  passed to author callbacks, with canonical ownership in
  `vercor.components.contexts`.
  `from_model()`, `default_fields`, `HostRuntimeComponent`, and
  component-prefixed context names have been removed from the public API.
  Component factory lifecycle customization is nested in
  `spec=ComponentSpec(..., lifecycle=LifecycleHooks(...),
  output=OutputConfig(...))`; individual hook and output keywords are not
  public constructor inputs. Runtime prefill and validation hooks receive typed
  `PrefillContext`, `PrefillResult`, and `ValidationContext` objects instead
  of mutable runtime-store dictionaries. Lifecycle hook type aliases
  (`ComponentInitializeHook`, `ComponentCreatePayloadHook`,
  `ComponentPrefillHook`, and `ComponentValidateHook`) remain public
  component-author type contracts and are reexported from `vercor.components`
  and `vercor`.
  `ComponentSpec`, `component.spec`, and `declare_fields()` provide the same
  vocabulary and read-only introspection for subclasses. `DataComponent` owns
  data-import behavior through
  `DataComponent.from_fields(..., import_policy=FieldImportPolicy(...))`;
  model-oriented `ComponentSpec` does not carry data selection policy.
  `field_names` exposes setup-time seeded field names in insertion order;
  direct `component.data` and `component.setup_metadata` mutation is not
  public API. Subclass constructors can use `seed_field(...)`,
  `seed_fields(...)`, `update_settings(...)`, and `grid_field_defaults(...)`
  to build validated grid-shaped default-field mappings with scalar expansion
  and field-specific overrides. Components that write native snapshots receive
  a typed `OutputConfig(snapshot_writer=...)` spec. Snapshot writers receive
  `SnapshotContext`, which exposes public `ComponentInfo`, the public
  `ComponentState`, component payload, output path, time, and logger without
  exposing normalized component adapters or `ComponentRuntimeState`. Mutable
  period-output adapters live under
  `vercor.output._component_adapter`.
  `DataComponent` seeding
  automatically records seeded fields as declared outputs, so data-only
  components remain introspectable whether fields are declared up front or added
  through constructor seeding. The removed `wrap()` classmethods and
  `make_*_component()` factory functions have been removed. The module-level
  `data_component()`, `differentiable_component()`, and `host_component()`
  factory helpers have also been removed. Component authors should use
  class-level `from_fields()` / `from_step()` constructors, or subclasses with
  `declare_fields(...)`. `vercor.components` and `vercor` reexport the
  component-author facade.
  `ComponentLike` is the canonical structural extension contract: structural
  components provide a non-empty name, `RectilinearGrid`, `ComponentSpec`,
  callable `initial_fields`/`initialize`/`step`, and mapping initial fields.
  Structural initialization calls the user object's `initialize`, refreshes
  its public state, calls the spec lifecycle initializer with that original
  object, and refreshes again; payload, prefill, and validation hooks likewise
  receive the original object. `DataComponent.from_step()` is intentionally
  rejected because data-only components do not execute steps.
  `vercor.components.contracts` owns public author-facing context, result,
  spec, lifecycle-hook types, and `ComponentStepReturn`; the alias is reexported
  by `vercor.components` but not the root facade. `ValidationContext` exposes
  public `ComponentState`, while underscored runtime lifecycle bridges keep
  runtime state/store contracts off inherited public component surfaces.
  Internal callable/field-normalization type aliases remain private underscored
  names. `vercor.components.base` owns only
  the abstract differentiable `Component` contract, `vercor.components.data`
  owns `DataComponent`, and `vercor.components.host` owns `HostComponent`.
  Field-name de-duplication lives in private `vercor._field_names`, and
  component authoring methods for field declarations, setup seeding, and
  settings updates live in private `vercor.components._field_authoring`.
  Lifecycle hook storage lives on the private `_lifecycle_hooks` component
  field; constructors use one public `LifecycleHooks` value and callable/data
  wrappers assign it directly. Default lifecycle dispatch lives in private
  `vercor.components._lifecycle_api`, and constructor-installed hooks are
  stored in one private container rather than as ad-hoc component attributes.
  Author-value normalization lives in private
  `vercor.components._contracts`; public constructor option normalization lives
  in private `vercor.components._constructor_options`; callable signature
  adaptation, shared callable construction options, and shared callable runtime
  mechanics live in private `vercor.components._callable_wrappers`, which
  carries lifecycle hooks as that container and delegates hook
  precedence/default payload fallback to the lifecycle mixin. The concrete
  callable-backed
  differentiable wrapper is owned by `vercor.components.base`, and the concrete
  callable-backed host wrapper is owned by `vercor.components.host`, keeping
  each runtime kind beside its public abstract base. Private helper modules use
  type-only `Component` annotations where they need concrete component shape.
  Host/scanned runtime selection is driven by the public
  `ComponentSpec.execution` value, so structural custom components can request
  host execution without subclassing a VerCOR base class or depending on a
  private marker protocol. `HostComponent` always enforces host execution;
  `RuntimeOptions.execution` selects the backend for the whole coupled run and
  retains precedence over per-component capability. Component-facing
  runtime-field adapters and runtime-store mutation helpers live in private
  `vercor.components._runtime_fields`, and component-facing required-field
  validation lives in private `vercor.components._runtime_validation`.
  Component host/scanned execution policy lives in internal
  `vercor.components.runtime_execution`, and setup validation lives in internal
  `vercor.components.setup_validation`, giving runtime modules explicit
  component-owned bridge modules instead of importing private component
  internals. These internals are not exported from `vercor.components`.
  Subclasses should call the base constructor with `name`, `grid`, and optional
  `settings`; raw setup `data` and `setup_metadata` are initialized internally
  rather than accepted as public constructor inputs. The private `_data` store
  contains grid fields, not a
  general metadata store: all entries must use one of the canonical layouts
  `(nLat, nLon)`, `(nTime, nLat, nLon)`, `(nLev, nLat, nLon)`, or
  `(nTime, nLev, nLat, nLon)`. Setup and runtime-state creation validate this
  contract before traced execution. Subclasses should seed fields with
  `seed_field()` or `seed_fields()` for scalar or array-like author values
  rather than mutating `data` directly; step methods should implement the
  mapping-based `step(fields, context, payload=None)` contract and return a
  field-update mapping or `StepResult(fields, payload)` when the runtime payload
  must be replaced. Private runtime adapters in
  `vercor.components.runtime_execution` translate between those public mappings
  and `FieldStore` membership, zero-like fallback, and existing-field
  replacement mechanics owned by the runtime.
  `seed_declared_defaults()` seeds fields from a component's declared
  defaults, and the base `initialize()` hook now does this automatically when
  subclasses do not need custom setup. Runtime prefill hooks are adapted through
  private `vercor.components._runtime_fields` helpers for ordinary
  output/default fields. Non-grid
  metadata such as hybrid-level coefficients belongs on component attributes or
  runtime payloads. Factory-created setup adapters should put non-runtime setup
  metadata in private `_setup_metadata` rather than attaching ad-hoc attributes
  to the component object. Examples include forcing-file provenance and
  diagnostic coefficients that should not enter runtime field validation or JAX
  scan state.
  `Component` for differentiable active models and implement `step(...)`. Use
  `DataComponent` for forcing/static data adapters that intentionally keep the
  shared no-op runtime step and do not create plotting-only runtime fields.
  Derived diagnostics, such as a combined land/sea surface temperature used only
  for plots, belong in diagnostics or setups. Use `HostComponent` for
  non-differentiable adapters and implement the same mapping-based `step(...)`;
  host-backed adapters must run through `Coupler.run()` so VerCOR can select the
  Python host runtime path. Optional
  hooks include `initialize()`, `create_runtime_payload()`,
  `prefill_runtime_state_fields()`, and `validate_runtime_state()`. Callable
  wrappers may accept `(fields)`, `(fields, context)`, or
  `(fields, context, payload)` and return either a field-update mapping or
  `StepResult(fields, payload)` when the runtime payload must
  be replaced. Callable-backed differentiable and host components share
  signature normalization and step-result application helpers. Internal
  normalized callable adapters use names that spell out which author arguments
  they forward, while
  `Component.from_step()` and `HostComponent.from_step()` construct their own
  private runtime-kind wrappers directly. Both declare their runtime contract
  with the same `ComponentSpec` path used by subclasses, and apply step results
  through the runtime-owned field replacement helpers.
  Runtime prefill and validation depend only on `inputs`, `outputs`, and
  `defaults`.
  These helpers still enforce the same stable runtime-state
  contract: updated fields must already exist through seeded data, declared
  outputs/defaults, or exchange prefill, and scanned payload pytrees must keep
  stable shapes and dtypes.
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
  `vercor._runtime.topology_policy` through the uniform
  `applies(context)` then `build(context)` protocol. Policy patches must target
  configured topology keys and target-grid shapes. Duplicate exchange topology
  keys are rejected rather than silently sharing a regridder. Optional atmosphere/ocean/land
  surface-mask creation and validation live in
  `vercor._runtime.surface_masks` behind `vercor.topology.SurfaceMaskPolicy`.
  Derived surface-mask values remain local to policy construction rather than
  retained as runtime state.
  Runtime exchange
  validation checks that exchanged fields are declared by the sending and
  receiving components instead of requiring the advisory common field
  vocabulary. `vercor._runtime.topology` remains the orchestration boundary
  that composes those owners and returns the explicit topology state for the
  runtime facade to store. Public `RunState` carries
  component states and fractional masks through `jax.lax.scan`; binary masks
  remain in `RuntimeTopologyMaps` for final output and topology bookkeeping.
  Setup-time component precision synchronization, initialization context construction, component
  setup validation, runtime contract validation, and topology handoff live in
  `vercor._runtime.initialization`. `vercor._runtime.prepared.PreparedCoupling`
  is the single frozen post-lifecycle boundary for normalized components,
  exchanges, run order, contracts, topology maps, destination-grouped dispatch,
  clock/settings/options, interrupts, and component configuration fingerprints.
  Component configuration mutation after preparation is rejected instead of
  silently rerunning lifecycle hooks. Runtime state creation, supplied-state
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
  Compiled and pure scanned execution, the Python host loop, custom backend
  adaptation, and validated public driver adaptation live in
  `vercor._runtime.backends`. `vercor._runtime.runner` owns only run-mode
  selection, host-component compatibility and warnings, interrupt signal scope,
  and delegation to those backend implementations. High-level runtime orchestration
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
  `RunState.component(...)` and `RunState.components(...)` are the only public
  component-view factories.
  Final runtime output iteration, output-mask naming/selection, and
  view writing live in private `vercor.output._runtime` helpers, with
  `vercor._runtime.facade` validating and delegating output writes for
  `Coupler.write_outputs()`. `Coupler.finalize()` has been removed; users write
  outputs through `Coupler.write_outputs()`. `Coupler` delegates to the runtime
  facade and remains
  the public setup/output facade rather than the owner of runtime adapter
  mechanics.
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
Examples and setup factories assemble runs through `Coupler(...)`,
`Coupler.add_exchange(...)`, `Coupler.add_exchanges(...)`, and direct
`Exchange(source, target, fields, regrid=...)` declarations. Shared exchange
field recipes live in `vercor.recipes` with `*_FIELDS` names. Short recipe aliases and setup orchestration helpers
such as `ExchangeSpec`, `build_coupler()`, `build_exchanges()`, and
`add_exchange_specs()` have been removed. Public exchange configuration types,
including `ExchangeField`, are owned by `vercor.fields`; `vercor.exchanges`
exports only the public `Exchange` class. Public regridding protocols,
including `Regridder` and `RegridderFactory`, are owned by
`vercor.regridding`; concrete
bilinear/conservative regridder classes remain private implementation details.
Regridders expose explicit `regrid(field)` and `regrid_vector(u, v)` methods
and are not callable.

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
`vercor.setups._external.jax_gcm_output`; `JAXGCMSetupState` owns a private
`_ComponentOutputAdapter` that streams prediction objects into the shared
JAX-backed sum/count period accumulator instead of retaining all period samples
or calling xarray adapters. JAXGCM-specific output helpers construct the
configured adapter and delegate prediction extraction, coordinate/metadata
builders, accumulation, cadence checks, and file writes through the shared
adapter record boundary. Final JAXGCM snapshots are registered by the external
factory and are written from the final runtime payload's `JCMState`, not from
runtime data fields or declared component outputs.
For coupled `run()` period output, the JAXGCM factory installs a private
setup-owned schema that extracts native prediction-equivalent variables from
the post-step payload `JCMState`. Host and chunked scanned backends therefore
share one session path while preserving `jcm.averages.YYYY-MM-DD.nc`, native
dimension ordering, and metadata; model steps no longer perform period-file
side effects. Coordinate-dependent physics cache setup occurs before JIT and
the installed sample extractor is pure. When output is enabled, the private
`JAXGCMRuntimePayload` also carries a shape-stable immutable per-step output
accumulator built from raw prediction-time sums and finite counts. Session
merging therefore weights runtime and spinup samples identically, including
NaNs and multi-time predictions.
Shared output extension primitives for adapter authors are exported from
`vercor.output`: `OutputVariable`, `PeriodOutput`, `OutputConfig`,
`SnapshotContext`, and `SnapshotWriter`. `OutputConfig.period is None`
disables period output; `PeriodOutput(frequency="step")` requests every-step
period output. Custom `ExecutionBackend` objects are rejected when period
output is configured until the public backend contract provides a
period-session hook. Snapshot writers receive only public component/result views and
the component payload. Shared cadence, calendar time metadata, dataset
coordinate discovery, period-sample/output conversion, period-average file
orchestration, and direct `h5netcdf` writing live in private
`vercor.output._period`, `vercor.output._dataset`,
`vercor.output._component_adapter`, `vercor.output._session`,
`vercor.output._period_files`, and
`vercor.output._netcdf`.
Surface-temperature cleanup and output-field mapping live in
`vercor.setups._external.jax_gcm_fields`. Veros host-runtime flux application and
substep orchestration live in `vercor.setups._external.veros_runtime` with
concrete setup-state annotations.
`vercor.setups._external.veros_gcm_state` owns Veros setup-time model resources,
spinup policy, grid derivation, and lifecycle callbacks, while
`vercor.setups._external.veros_gcm` remains the private factory implementation.
The shared output-period accumulator stores one running sum plus one
finite-value count array per variable as JAX arrays, preserving current
`nanmean` semantics without retaining every timestep. Opt-in Veros period-output
extraction, native Veros variable metadata handling, ghost-cell removal, and
write-time native Veros spatial-axis ordering policy live in
`vercor.setups._external.veros_output`; `VerosGCMSetupState` owns the same
private `_ComponentOutputAdapter`, and `vercor.setups._external.veros_runtime`
streams selected snapshots through the Veros output helper, which delegates
accumulation, cadence checks, and file writes to the shared adapter record
boundary with the same day/month/year cadence policy used by JAXGCM. Veros
final snapshots use the same native-state extraction helpers through a
component-registered snapshot writer.
average files keep VerCOR's
lowercase `time` dimension while matching native Veros spatial NetCDF dimension
order, and the accumulator averages only across recorded runtime samples rather
than reducing horizontal or vertical axes. Private Veros output helpers keep
variable and coordinate extraction names parallel to make data-variable versus
coordinate-variable responsibilities explicit.
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
CAMulator forecast-increment output remains the default when
`PeriodOutput.frequency` is unset; when it is `day`, `month`, or `year`,
`CAMulatorGCMSetupState` owns the same private `_ComponentOutputAdapter` and
`vercor.setups._external.camulator_runtime` streams native prediction tensors
through the CAMulator output helper, which delegates average accumulation,
cadence checks, and file writes to the shared adapter record boundary.
CAMulator records the latest native prediction as a single snapshot
record in both increment-output and period-output modes, and final snapshots
reuse the same adapter/output builders without falling back to VerCOR runtime
fields. CAMulator tensor reshaping, metadata handling, output filtering from
`predict.save_vars`, average-file path/coordinate adaptation, and
forecast-increment writing live in `vercor.setups._external.camulator_output`.

### Distribution and external plugin boundary

Runtime metadata excludes test tooling; `test` and `dev` extras own pytest,
formatting, lint, typing, and build dependencies. `vercor/py.typed` marks the
wheel and sdist as PEP 561 typed packages. The independently packaged fixture
under `tests/fixtures/public_plugin` imports only public VerCOR modules and
exercises structural JAX/host components, original-object lifecycle hooks, a
custom sequential backend, a custom topology policy, and snapshot output
against an installed wheel. CI builds wheel/sdist once, then tests installed
base, JCM, and Veros environments on Python 3.12 and 3.13 by consuming the
downloaded artifacts rather than rebuilding them in each matrix cell. Setup
subprocess probes and public-plugin mypy run from the installed site-packages
root or an external temporary use site, so checkout sources cannot satisfy an
installed-artifact check. CAMulator is omitted from the install matrix until a
compatible NCAR MILES-CREDIT release is verified.

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

### Settings container

VerCOR uses one metadata-backed `Settings` class for both coupler-level
and component-level settings. `vercor.settings.DEFAULT_SETTINGS` stores the
defaults as `SettingSpec(value, description, units)` namedtuple records; unitless
settings use `"-"` for units. Each `Coupler` and each `Component` receives an
independent `Settings()` instance populated from those defaults at
construction time, so setup-time changes on one owner do not leak into another.

Settings support direct attribute reads and assignments: `settings.enable_x64`
and similar attribute reads resolve setting values dynamically through
`__getattr__`, and assigning an existing attribute updates only that value
through `__setattr__`. Known default
settings are declared as class-level annotations so static type checkers retain
useful types without per-setting runtime property descriptors. Constructor
keyword arguments may only override known default settings, so misspelled
physics/configuration names fail eagerly. New custom settings must be introduced
explicitly with `add()` or passed through `Settings(custom={...})`; existing
settings should be updated with `set()` where production code is making an
intentional configuration change. Use
`get()`, `get_metadata()`, and `as_dict()` for explicit lookup and
introspection. `dir(settings)` includes default and custom setting names for
introspection. The obsolete `ComponentSettings` and `VercorSettings` aliases
have been removed; use `Settings` directly.

### Precision and dtype policy

VerCOR-owned array dtypes are centralized in `vercor.dtypes`. Real-valued JAX
and NumPy arrays use the `Settings.enable_x64` precision switch whenever a
settings object is available: `False` maps to 32-bit real arrays and `True` maps
to 64-bit real arrays. Runtime initialization through
`Coupler.initial_state()`, `Coupler.run()`, or `Coupler.write_outputs()` treats
the coupler setting as the run-level precision policy, synchronizes component
settings to that policy, and recasts component-owned grid/data arrays before
runtime state creation. Helpers that create arrays without a settings object
follow the active JAX global
`jax_enable_x64` configuration; conversion helpers preserve an already-typed
real array when no settings object is supplied.
Integer/index arrays use the canonical 32-bit index dtype in both
real-precision modes to keep sparse metadata and interpolation indices compact.

Production kernels and adapters should use the dtype helpers rather than
hard-coded `jnp.float64`, `jnp.float32`, `jnp.float_`, `jnp.int64`, or
`jnp.int32` annotations. At host boundaries that require NumPy dtype objects,
derive them explicitly with `np.dtype(jax_real_dtype(policy))` or
`np.dtype(jax_index_dtype(policy))`. NumPy remains restricted to explicit host
and dtype boundaries. File-output adapters should keep VerCOR-owned values
JAX-backed and delegate external component period-average orchestration to
`vercor.output._component_adapter`, period-file writes to
`vercor.output._period_files`, and final file-transfer conversion to
`vercor.output._netcdf`, which calls `vercor.host_arrays` only when a non-JAX
consumer, such as `h5netcdf` or a host-backed model runtime, requires a host
array.

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
Initialization, runtime, and finalization helpers that are reached outside a
coupler context use the default `VerCOR` Python logger from
`vercor.jax_logging.get_default_logger()`. Helpers reached from
`Coupler.initial_state()`, `Coupler.run()`, `Coupler.write_outputs()`, or
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
`vercor._runtime.runner`, which owns the signal scope, while
`vercor._runtime.backends` owns host, scanned, and custom runtime cancellation
checkpoints. During a run, `SIGINT`, `SIGTERM`, and `SIGTSTP` request graceful
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

### 3.1 Constants and Parameters

**File**: `constants.py`

For physical constants, such as gravitational acceleration, gas constant, etc.

**File**: `parameters.py`

For runtime parameters, such as coupled run identifier, precision, time interpolation type,
type of year (leap, noleap, 360day), etc.

Two parameter containers:

```python
@dataclass(frozen=True)
class PhysicsParameters:
    """All fields are JAX-traceable floats."""
    # Scalars
    gravity: float
    rhoAir: float
    rgas: float
    latvap: float
    zref: float
    mwdair: float
    ...


@dataclass(frozen=True)
class RuntimeOptions:
    """Static coupler runtime policy. NOT traced by JAX."""
    dtype: DTypePolicy
    execution: str | ExecutionBackend
    topology: TopologyPolicy | None
    model_year_seconds: float


@dataclass(frozen=True)
class FieldImportPolicy:
    """Static component field-import policy."""
    daily_selection: bool
    time_interpolation: bool
```

The split between `PhysicsParameters` (traced) and `ControlParameters` (static) is critical:
JAX traces through `PhysicsParameters` for AD, while `ControlParameters` controls array
shapes and solver settings that must be compile-time constants.

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

The entire `run_simulation()` function should be JIT-compiled:

Since `ControlParameters` is static (controls array shapes), it should be passed via
`static_argnums` or as a `static_field` in an Equinox module.

First call will be slow (~30-60s for XLA compilation). Subsequent calls with the
same shapes will be fast.
