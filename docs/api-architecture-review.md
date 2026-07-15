# VerCOR 0.4.0a1 API architecture review

This review records the shipped alpha architecture. Public compatibility is
defined by canonical owner modules, their `__all__` manifests, live signatures,
and behavior tests—not by the layout of private implementation modules.

## 1. Executive summary

VerCOR 0.4 is a deliberate protocol-first break from 0.3. A configured
`Coupler` normalizes structural components once, validates stable exchange
routes and topology, asks a workflow for an exact clock-step plan, executes
core-defined chunks through a backend and validated driver, and coordinates all
requested output at the run boundary.

The package root contains six conveniences only: `Clock`, `Coupler`, `Exchange`,
`RectilinearGrid`, `RunState`, and `RuntimeOptions`. Advanced contracts each
have one canonical owner. `ComponentSpec` is the sole component declaration;
`PhysicalConstants` is the traced physics PyTree; `RuntimeOptions` owns static
execution policy; setup configuration belongs to the setup or plugin.

The differentiable default is `Coupler.run(output=None)`. That path performs no
I/O and retains JVP and reverse-mode behavior. An `OutputTarget` opts into the
single coordinator for provider sampling, selection, immutable accumulation,
host transfer, period files, final fields, and snapshots. Optional JCM, Veros,
and CAMulator dependencies remain lazy. CAMulator is source-tested but is not
installed or pinned for this alpha.

## 2. Duplication map

| Earlier overlap | VerCOR 0.4 owner | Resolution |
| --- | --- | --- |
| Inherited, callable, host, and data authoring hierarchies | `vercor.components` | One structural `Component` protocol plus `CallableComponent` and `DataComponent`. |
| Defaults, initialization, payload creation, transfer, and output properties | `ComponentSpec` | One immutable declaration and one `LifecycleHooks.setup` result. |
| Coupler recipes, mutators, and facade reexports | `vercor.coupler.Coupler` | Constructor-only assembly with immutable configuration views. |
| Callable-derived exchange keys and topology tuple keys | `Exchange.route_id` | Stable route identity used by topology and validation. |
| Backend-specific scheduling | `Workflow` and the private execution coordinator | One validated plan and core-owned chunks for every backend. |
| Backend/native output sessions | `vercor.output._session` | One immutable accumulator and one run-level output lifecycle. |
| Public and internal state mutation helpers | `RunState.replace_fields` | One immutable public replacement operation; alignment remains private. |
| Setup import registries | `vercor.setups` | One lazy public facade; implementation imports occur at factory invocation. |

These consolidations are intentionally narrow. Scalar and vector regridding
remain separate capabilities. Provider sampling and file writing remain
separate boundaries. Physics values and static precision policy remain
separate because JAX traces the former and uses the latter to control arrays.

## 3. Bad design decisions

The review identified and removed the following failure-prone designs:

- Broad root reexports made ownership ambiguous and allowed aliases to drift.
- Mutable coupler assembly made a prepared graph disagree with visible
  configuration.
- Reflection over author objects created an unstable prepared snapshot and
  encouraged hidden mutable configuration.
- Component-specific output markers and native accumulators duplicated cadence
  and silently skipped output under compiled execution.
- Custom backends that could call arbitrary component steps made schedule,
  cancellation, and output boundaries unverifiable.
- Callable-derived exchange identity collided for repeated endpoints and could
  not address topology patches reliably.
- Shape-only foreign-state checks admitted changed coordinates, dtypes, and
  invalid masks.
- Eager optional-model imports made core import depend on unused frameworks and
  their process configuration.

The corresponding 0.4 rules are constructor-only assembly, explicit route IDs,
exact workflow plans, validated driver calls, strict state schemas, one output
coordinator, and lazy setup factories. Ambiguous fan-in remains an error. A
fan-in reducer, public prepared graph, registry, entry-point discovery,
Pydantic model, and fractional subcycling are not part of this release.

## 4. Public API redesign

The machine-readable inventory below is ordered by canonical owner. CI executes
it against each live module and independently checks central constructor
signatures from installed artifacts.

<!-- public-api-manifest:start -->
```json
{
  "vercor": ["Clock", "Coupler", "Exchange", "RectilinearGrid", "RunState", "RuntimeOptions"],
  "vercor.assets": ["VERCOR_ASSETS_BASE_URL", "ensure_registered_asset"],
  "vercor.calendar": ["CalendarDate", "DAYS_PER_MONTH_360", "DAYS_PER_MONTH_GREGORIAN_LEAP", "DAYS_PER_MONTH_GREGORIAN_NO_LEAP", "DateTime360", "DateTime365", "ModelDateTime", "YearType", "day_of_year_from_month_day", "is_leap_year", "model_year_seconds", "month_day_from_day_of_year", "year_type_for_calendar"],
  "vercor.clock": ["Clock"],
  "vercor.components": ["CallableComponent", "Component", "ComponentSpec", "DataComponent", "LifecycleHooks", "PrefillContext", "PrefillResult", "SetupContext", "SetupResult", "StepContext", "StepResult", "TransferPolicy", "ValidationContext"],
  "vercor.coupler": ["Coupler"],
  "vercor.diagnostics": ["ComponentMetric", "combine_surface_temperatures", "component_vector_speed", "plot_component_scalar_vector_comparison", "print_component_field_means_table", "safe_component_nanmean", "total_surface_temperature"],
  "vercor.dtypes": ["DTypePolicy", "PrecisionPolicy", "ShapeLike", "as_jax_index_array", "as_jax_real_array", "dtype_policy", "jax_arange", "jax_full", "jax_index_dtype", "jax_linspace", "jax_ones", "jax_real_dtype", "jax_zeros"],
  "vercor.exceptions": ["AssetError", "ComponentError", "CouplerError", "ExchangeError", "GridError", "RegridderError"],
  "vercor.exchanges": ["Exchange"],
  "vercor.field_layout": ["CANONICAL_DATA_LAYOUTS", "canonical_data_layout_description", "canonical_grid_field_shape", "canonical_grid_field_shape_error", "canonicalize_time_last_level_field", "canonicalize_time_last_surface_field", "is_canonical_grid_field_shape", "validate_canonical_grid_field_shape", "validate_component_data_layout"],
  "vercor.fields": ["COMMON_FIELD_NAMES", "ExchangeField", "VectorField", "vector"],
  "vercor.fluxes": ["cdn", "compute_air_density", "compute_hybrid_pressure_levels", "compute_hybrid_sigma_full_level_altitudes", "compute_ocean_surface_fluxes", "compute_potential_temperature", "compute_sigma_pressure_levels", "get_altitudes_hybrid_sigma_levels", "get_altitudes_sigma_levels", "psimhu", "psixhu", "qsat", "qsat_august_eqn", "shr_flux_atmIce"],
  "vercor.forcing_data": ["read_forcing"],
  "vercor.forcing_index": ["ForcingYearType", "daily_forcing_day_of_year", "daily_forcing_index", "day_of_year_360_to_gregorian", "gregorian_month_lengths", "noleap_day_of_year"],
  "vercor.grid_geometry": ["centers_to_edges", "grids_identical"],
  "vercor.grid_masks": ["check_remap_conservation", "check_total_lnd_ocn_mask_sum", "compute_land_mask", "compute_ocn_lnd_masks_on_atm_grid", "create_lnd_mask_from_ocn"],
  "vercor.grids": ["RectilinearGrid"],
  "vercor.jax_logging": ["CANONICAL_LOG_DATE_FORMAT", "CANONICAL_LOG_FORMAT", "DEFAULT_LOGGER_NAME", "JaxCallbackLogger", "LoggerLike", "configure_python_logger", "effective_log_level", "emit_host_log", "get_default_logger", "logger_enabled_for", "normalize_log_level", "setup_logger"],
  "vercor.output": ["OutputContext", "OutputFrame", "OutputProvider", "OutputSpec", "OutputTarget", "OutputVariable", "PeriodOutput", "SnapshotContext", "SnapshotWriter"],
  "vercor.physics": ["PhysicalConstants"],
  "vercor.recipes": ["ATMOSPHERE_TO_DATA_OCEAN_FIELDS", "ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS", "ATMOSPHERE_TO_LAND_BASIC_FIELDS", "ATMOSPHERE_TO_LAND_RADIATION_FIELDS", "ATMOSPHERE_TO_LAND_STATE_FIELDS", "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS", "ATMOSPHERE_TO_OCEAN_STATE_FIELDS", "ATMOSPHERE_TO_VEROS_FORCING_FIELDS", "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS", "JCM_LAND_TO_ATMOSPHERE_FIELDS", "LAND_TO_ATMOSPHERE_SOIL_FIELDS", "LAND_TO_ATMOSPHERE_SURFACE_FIELDS", "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS", "OCEAN_TO_SEAICE_SURFACE_FIELDS", "SEAICE_TO_OCEAN_FIELDS", "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS", "SLAB_ATMOSPHERE_TO_OCEAN_FIELDS", "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS"],
  "vercor.regridding": ["Regridder", "RegridderFactory", "VectorRegridder", "bilinear", "conservative"],
  "vercor.runtime": ["ExecutionBackend", "ExecutionChunk", "ExecutionContext", "ExecutionPlan", "RuntimeDriver", "RuntimeOptions", "SequentialWorkflow", "StepPlan", "Workflow", "WorkflowContext"],
  "vercor.setups": ["CAMulatorConfig", "JAXGCMConfig", "JCMLandAtmosphereConfig", "JCMLandAtmosphereSetup", "JCMInputs", "Spinup", "VerosConfig", "load_jcm_inputs", "make_slab_atmosphere", "make_slab_land", "make_slab_ocean", "make_slab_seaice", "make_jcm_land_atmosphere", "make_camulator_gcm", "make_camulator_land", "make_era5_atmosphere", "make_era5_land", "make_era5_ocean", "make_erainterim_ocean", "make_jax_gcm", "make_jcm_land", "make_veros_gcm"],
  "vercor.state": ["ComponentState", "FieldLookupScope", "FieldScope", "RunState"],
  "vercor.time_selection": ["datetime_to_seconds_in_year", "get_periodic_interval"],
  "vercor.topology": ["ExchangeTopologyPatch", "SurfaceMaskPolicy", "TopologyContext", "TopologyPolicy"],
  "vercor.types": ["RuntimeArray"]
}
```
<!-- public-api-manifest:end -->

The central signatures are constructor-only and keyword-delimited where policy
could otherwise be confused with data:

```text
Coupler(clock, *, components=(), exchanges=(), run_order=(), runtime=None,
        constants=None, logger=None, log_level="INFO")
Exchange(source, target, fields, *, route_id=None, regridder_factory=bilinear)
ComponentSpec(inputs=(), outputs=(), initial_fields=None, *, execution="jax",
              lifecycle=None, transfer=None, output=None)
RuntimeOptions(dtype=DTypePolicy(), backend="auto", workflow=SequentialWorkflow(),
               topology=None)
OutputTarget(directory, *, write_period=True, write_final_fields=True,
             write_snapshots=True)
```

The five signatures above are a readable representative sample. The complete,
static executable inventory is
[`tests/contracts/vercor-0.4.0a1-public-signatures.json`](../tests/contracts/vercor-0.4.0a1-public-signatures.json):
it covers all 150 concrete callable exports in canonical non-root owner
manifests and 55 public class/protocol-call methods. Every
normalized value includes parameter order and kind, defaults, resolved public
annotations, and the return annotation. Source and isolated installed-artifact
tests require exact key-set equality and execute the same frozen contracts.

Model-year policy is calendar-owned rather than runtime-owned:
`Clock.calendar` selects the calendar, and `vercor.calendar` resolves each
timestamp's canonical year type and duration for runtime forcing metadata.

`RunState` is opaque. Its public operations are `component`, `components`, and
`replace_fields`; callers never receive runtime stores, topology maps, or a
prepared graph. Public annotations resolve without importing private symbols.

## 5. Private API redesign

Private layout is free to change and is not a compatibility contract. The
following inventory is complete for `0.4.0a1` and documents responsibility,
not import permission.

```text
vercor._field_names
vercor._host_arrays
vercor._interpolators
vercor._interpolators._bilinear_extrapolation
vercor._interpolators._bilinear_geometry
vercor._interpolators._bilinear_weights
vercor._interpolators.bilinear_rectilinear
vercor._interpolators.conservative_remap_rectilinear
vercor._logging
vercor._logging.callback
vercor._logging.config
vercor._logging.host
vercor._logging.protocols
vercor._pytree
vercor._regridders
vercor._regridders.base
vercor._regridders.bilinear
vercor._regridders.conservative
vercor._run_order
vercor._runtime
vercor._runtime.backends
vercor._runtime.component_state
vercor._runtime.component_topology
vercor._runtime.contracts
vercor._runtime.coupler_state
vercor._runtime.dispatch_context
vercor._runtime.driver
vercor._runtime.exchange_dispatch
vercor._runtime.exchange_topology
vercor._runtime.execution
vercor._runtime.facade
vercor._runtime.field_transfer
vercor._runtime.initialization
vercor._runtime.interrupts
vercor._runtime.preparation
vercor._runtime.prepared
vercor._runtime.progress
vercor._runtime.run_context
vercor._runtime.runner
vercor._runtime.state
vercor._runtime.state_validation
vercor._runtime.stores
vercor._runtime.surface_masks
vercor._runtime.time
vercor._runtime.topology
vercor._runtime.topology_policy
vercor._runtime.topology_state
vercor._runtime.validation
vercor.components._adapter
vercor.components._callable_wrappers
vercor.components._contracts
vercor.components._protocol
vercor.components._runtime_fields
vercor.components.base
vercor.components.contexts
vercor.components.contracts
vercor.components.data
vercor.components.runtime_execution
vercor.components.setup_validation
vercor.diagnostics.fields
vercor.diagnostics.plotting
vercor.diagnostics.tables
vercor.fluxes.bulk_formula_cesm
vercor.fluxes.utilities
vercor.fluxes.vertical_coordinates
vercor.output._dataset
vercor.output._netcdf
vercor.output._period
vercor.output._runtime
vercor.output._session
vercor.setups._data
vercor.setups._data._component_helpers
vercor.setups._data._field_helpers
vercor.setups._data.assets
vercor.setups._data.era5_atmosphere
vercor.setups._data.era5_land
vercor.setups._data.era5_ocean
vercor.setups._data.erainterim_ocean
vercor.setups._data.jcm_land
vercor.setups._external
vercor.setups._external._camulator_wind_filtering
vercor.setups._external._jax_gcm_pytree
vercor.setups._external.camulator
vercor.setups._external.camulator_contracts
vercor.setups._external.camulator_fields
vercor.setups._external.camulator_forcing
vercor.setups._external.camulator_gcm_state
vercor.setups._external.camulator_imports
vercor.setups._external.camulator_init
vercor.setups._external.camulator_land
vercor.setups._external.camulator_output
vercor.setups._external.camulator_runtime
vercor.setups._external.camulator_runtime_settings
vercor.setups._external.camulator_stepper
vercor.setups._external.camulator_tensors
vercor.setups._external.camulator_wind_filter
vercor.setups._external.jax_gcm
vercor.setups._external.jax_gcm_fields
vercor.setups._external.jax_gcm_output
vercor.setups._external.jax_gcm_runtime
vercor.setups._external.jax_gcm_state
vercor.setups._external.jax_gcm_tools
vercor.setups._external.veros_fluxes
vercor.setups._external.veros_gcm
vercor.setups._external.veros_gcm_state
vercor.setups._external.veros_output
vercor.setups._external.veros_runtime
vercor.setups._external.veros_runtime_settings
vercor.setups._external.veros_setup
vercor.setups._external.veros_state
vercor.setups._jcm
vercor.setups._lazy_imports
vercor.setups._slab
vercor.setups._slab.atmosphere
vercor.setups._slab.land
vercor.setups._slab.ocean
vercor.setups._slab.seaice
vercor.setups._time_helpers
vercor.setups.config
```

### Foundations and numerical implementations

- Field/runtime helpers: `vercor._field_names`, `vercor._host_arrays`,
  `vercor._pytree`, and `vercor._run_order`.
- Interpolation: `vercor._interpolators`,
  `vercor._interpolators._bilinear_extrapolation`,
  `vercor._interpolators._bilinear_geometry`,
  `vercor._interpolators._bilinear_weights`,
  `vercor._interpolators.bilinear_rectilinear`, and
  `vercor._interpolators.conservative_remap_rectilinear`.
- Logging: `vercor._logging`, `vercor._logging.callback`,
  `vercor._logging.config`, `vercor._logging.host`, and
  `vercor._logging.protocols`.
- Regridding: `vercor._regridders`, `vercor._regridders.base`,
  `vercor._regridders.bilinear`, and `vercor._regridders.conservative`.

### Runtime coordinator

The private runtime package consists of `vercor._runtime` and these focused
owners: `backends`, `component_state`, `component_topology`, `contracts`,
`coupler_state`, `dispatch_context`, `driver`, `exchange_dispatch`,
`exchange_topology`, `execution`, `facade`, `field_transfer`, `initialization`,
`interrupts`, `preparation`, `prepared`, `progress`, `run_context`, `runner`,
`state`, `state_validation`, `stores`, `surface_masks`, `time`, `topology`,
`topology_policy`, `topology_state`, and `validation`, each beneath
`vercor._runtime`.

`prepared` owns the single immutable post-setup binding. It contains normalized
components, routes, contracts, topology, clock, constants, and static runtime
policy; it contains neither reflective author snapshots nor a public prepared
graph. `execution` validates the workflow and owns chunk boundaries.
`backends` adapts JAX, host, and custom executors. `driver` is the only component
dispatch route exposed through the public driver wrapper. State validation
covers exact components, fields, payload structure, route maps, coordinates,
shapes, dtypes, and finite mask constraints before and after external backend
calls.

### Component, diagnostics, flux, and output implementations

- Components: `vercor.components._adapter`, `_callable_wrappers`, `_contracts`,
  `_protocol`, `_runtime_fields`, `base`, `contexts`, `contracts`, `data`,
  `runtime_execution`, and `setup_validation`.
- Diagnostics: `vercor.diagnostics.fields`, `plotting`, and `tables`.
- Fluxes: `vercor.fluxes.bulk_formula_cesm`, `utilities`, and
  `vertical_coordinates`.
- Output: `vercor.output._dataset`, `_netcdf`, `_period`, `_runtime`, and
  `_session`.

The component adapter is the only declaration-to-runtime normalization
boundary. The output session is the only cadence and mean-accumulation owner;
`_netcdf` is the only file-writing primitive. There is no hidden output marker,
second component adapter, duplicate accumulator, or native period-file path.

### Bundled setup implementations

- Data package: `vercor.setups._data`, `_component_helpers`, `_field_helpers`,
  `assets`, `era5_atmosphere`, `era5_land`, `era5_ocean`,
  `erainterim_ocean`, and `jcm_land`.
- External package: `vercor.setups._external`, `_camulator_wind_filtering`,
  `_jax_gcm_pytree`, `camulator`, `camulator_contracts`, `camulator_fields`,
  `camulator_forcing`, `camulator_gcm_state`, `camulator_imports`,
  `camulator_init`, `camulator_land`, `camulator_output`, `camulator_runtime`,
  `camulator_runtime_settings`, `camulator_stepper`, `camulator_tensors`,
  `camulator_wind_filter`, `jax_gcm`, `jax_gcm_fields`, `jax_gcm_output`,
  `jax_gcm_runtime`, `jax_gcm_state`, `jax_gcm_tools`, `veros_fluxes`,
  `veros_gcm`, `veros_gcm_state`, `veros_output`, `veros_runtime`,
  `veros_runtime_settings`, `veros_setup`, and `veros_state`.
- Remaining setup owners: `vercor.setups._jcm`, `vercor.setups._lazy_imports`,
  `vercor.setups._slab`, `vercor.setups._slab.atmosphere`,
  `vercor.setups._slab.land`, `vercor.setups._slab.ocean`,
  `vercor.setups._slab.seaice`, `vercor.setups._time_helpers`, and
  `vercor.setups.config`.

Public setup access is always through `vercor.setups`. Private factories defer
JCM/Dinosaur, Veros, CREDIT, Torch, and TensorFlow imports until invocation.
JAXGCM, Veros, and CAMulator expose ordinary output providers and snapshot
writers; they do not own cadence, paths, period accumulation, or writes.

## 6. Setup-agnostic plugin architecture

Plugins are ordinary Python packages that inject objects explicitly. There is
no registry or entry-point discovery. The independently built fixture under
`tests/fixtures/public_plugin` proves the complete boundary using only public
imports:

1. a plugin-owned frozen configuration and assembly factory;
2. structural JAX and host components plus setup payload replacement;
3. a structural scalar regridder and injected factory on an explicit route ID;
4. a non-empty route-keyed topology patch;
5. a plugin workflow and chunk-oriented custom backend;
6. immutable `RunState.replace_fields` before driver execution; and
7. per-step provider output and a final snapshot.

The plugin wheel is installed next to built VerCOR artifacts in a clean target,
its smoke runs outside the checkout, and its package plus external use site pass
strict mypy. CI repeats native 0.4 plugin lanes on Python 3.12 and 3.13. The
frozen 0.3 plugin is retained only as a historical artifact: its metadata and
source boundary are inspected, but it is not executed against 0.4.

Bundled slab, JCM, and Veros factories return ordinary structural components
and use the same constructor and output contracts. CI has installed base, JCM,
and Veros lanes. CAMulator remains lazy and source-tested because a compatible
external release is not yet pinned.

## 7. Compatibility plan

VerCOR 0.4 is intentionally source-breaking and this alpha does not ship a 0.3
adapter namespace. Task 9 was explicitly skipped. Applications migrate imports
and construction directly using `docs/migration-0.3-to-0.4.md`; primary 0.4 modules
remain alias-free.

The frozen `tests/contracts/vercor-0.3.2-public-api.json` and frozen 0.3 plugin
wheel remain historical evidence only. They define what changed and guard
against rewriting history, but they do not promise that a legacy application
runs on 0.4. No earlier API is restored.

Compatibility within the 0.4.x line is defined by canonical public owner
manifests, signatures, public-only plugin behavior, output-free gradients, and
installed wheel/sdist tests. Private module names in section 5 are descriptive
and may change without a deprecation cycle.

## 8. Final rewritten API

The final alpha data flow is:

```text
components + exchanges + clock + RuntimeOptions
                    |
                    v
       private immutable preparation
                    |
                    v
 Workflow -> ExecutionPlan -> validated chunks
                    |
                    v
       backend -> RuntimeDriver.run_step
                    |
                    v
              immutable RunState
                    |
          OutputTarget supplied?
             /             \
           no               yes
      no host I/O      one output coordinator
```

The release contract is `0.4.0a1`, Python 3.12+, a six-symbol root, one owner
per advanced public symbol, constructor-only coupling, protocol-first
components, stable route IDs, exact workflow plans, chunk backends, strict
state validation, opaque immutable public state, and one opt-in output
coordinator. Wheel and installed-sdist probes verify the packaged surface and
PEP 561 marker outside the checkout.

Deferred features stay deferred: no registry, entry-point discovery, Pydantic,
fan-in reducer, public prepared graph, fractional subcycling, or CAMulator
dependency pin. No tag, push, publication, or release upload is part of
preparing this alpha.
