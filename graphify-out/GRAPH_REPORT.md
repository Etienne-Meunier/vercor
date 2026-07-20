# Graph Report - VerCOR  (2026-07-20)

## Corpus Check
- 271 files · ~234,985 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4281 nodes · 12503 edges · 175 communities (159 shown, 16 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 1053 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f6bef7a1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Representative Near Surface State Over
- Single Private Normalization Bridge For
- Make Grid
- Cache Coords
- Custom Component Wrapping
- Run Every Step Plan In
- With Fields
- Regrid Vector
- Canonicalize Time Last Level Field
- Interrupts Components
- Workflow Runtime Execution Tests
- Run Data Driver
- Patch The Configured Route With
- Configuration For The Bundled Camulator
- Return The Configured Runtime Logger
- Field Names
- Compute Sigma Pressure Levels
- Has Identical Grids
- Profile Runtime
- Regrid Vector
- Create A Complete Immutable Coupling
- Dtypes Components
- Jax Gcm Output
- Return Setup Owned Field And
- Api Boundaries
- Field Transfer
- Ver Cor 0 4 Unified
- Focused Tests For The Sole
- Grid Geometry
- Model Year Seconds
- Run A Coupler Through The
- Regrid Vector
- Create Runtime State From Coupler
- Run Camulator With Veros
- Opaque Mutable Leaf Used To
- Build Distributions
- Select Fast Cases
- Public Post Step Component View
- Canonicalize External Typing Aliases
- Fields Components
- Apply Conservative Scalar Regridding
- Return The Immutable Runtime Policy
- Veros Output
- Assert Allclose Compact
- Assertions Components
- V0 4 Public Api
- Struct Time
- Run Jcm With Era5Data
- Get Loc
- Repository Wide Contracts For Ver
- Calendar Components
- Capture Logger Output
- Bilinear Rectilinear
- Ver Cor 0 4 Route
- Base Class For Exceptions Inside
- Conftest Components
- Init Context
- Regrid Vector
- Array To Host
- Advance The Private Host Backed
- Assets Components
- Mark Variables That Are Recognized
- Convert Numpy Jax Arrays To
- Extrapolate Scalar Field
- Shared H5Netcdf Output Writer For
- Reject Any Enabled I O
- Base Class For Exceptions Related
- Day Of Year
- Shared Net Cdf Writer Boundary
- Initialize Camulator Runtime Fields
- Apply Scalar
- Build Output Plan
- Iter Fields
- Stable Public Output Contracts For
- Run Optional Author Validation With
- Public Payload Passed To Component
- Make Example Grid
- Class Body Source
- Apply Wind Filter To Tensor
- Canonical Data Layout Description
- Boundary Tests For Lazy Bundled
- Atmosphere Components
- First Development
- Is Period End
- V0 4 Physics
- Iterator Over Simulation Time Steps
- Make Veros Gcm
- External Tools Coverage
- Tests For The Injectable Jcm
- Camulator Stepper
- Actual Averaging Window Start
- Package Import Cycles
- Structural Differentiable Component Implemented Outside
- Apply Scalar
- Return Typed Indexing Metadata For
- Load Optional Setup Factories Only
- Run Every Readme Python Block
- Set Up And Return The
- Initialize Camulator Forcing Cursor
- Camulator Init
- Make Differentiable Model
- Emit A Debug Message
- Build Input With Forcing
- Controlled Pytest Parallelization Implementation Plan
- Plot Component Scalar Vector Comparison
- Validate The Exact Structural Component
- Initial State
- Initialize Model State Helpers And
- Camulator Contracts
- Bundled Native Output Paths Use
- Veros Runtime Settings
- Camulator Imports
- Return A 4D Anisotropic Gaussian
- Load Jcm Coords Terrain Forcing
- Set Diagnostics
- Use Site
- Return Normalized Declarations For Private
- Create A Public Component State
- Positive Current Contract Testing
- Smoke Components
- Production Numpy Boundaries
- Dummy Grid Component
- Reject Invalid Nested Callbacks At
- Return The Registered Component Name
- Unified Opt In Output
- Calendar Year Ownership Design
- Return Selected Default Jcm Parameter
- Protocol First Components
- Release Procedure
- Runnable Ver Cor Example Drivers
- Vercor Public Plugin
- Private Runtime Modules Should Import
- Path Components
- Private Logging Implementation Modules For
- Validate Cadence And Freeze A
- Concrete Regridder Implementation Package
- Internal Runtime Package Runtime Containers
- Copy Mutable Caller Provided Mappings
- Private Owner Package For Bundled
- Private Owner Package For Bundled
- Route Keyed Topology
- Spherical Bilinear Interpolation
- 2026-04-29
- 2026-05-14
- File Structure
- run_camulator_prediction_block
- normalize_component_step_callable
- Milestone Timeline
- Test-suite performance optimization design
- Veros Duplicate-Dimension Output Bug-Fix Design
- canonicalize_external_typing_aliases
- _RecordingLogger
- 2026-04-30
- 2026-05-05
- VerCOR 0.4.0a1 Protocol-First API Design
- VerCOR Pre-1.0 Versioning Correction Design
- .request
- .__init__
- Veros Output Universe Bug-Fix Design
- 2026-05-06
- VerCOR Progress Archive: 2026-05-16 to 2026-07-14
- File Map
- Controlled Pytest Parallelization Implementation Plan
- Test-suite Artifact Reuse Implementation Plan
- VerCOR Pre-1.0 Versioning Correction Implementation Record
- Period-Average Window-Start Timestamp Implementation Plan
- Global Constraints
- Global Constraints
- Considered approaches
- Considered approaches
- .__init__
- .scanned_checkpoint
- over-engineering-audit-2026-06-12.md
- over-engineering-audit-2026-06-29.md

## God Nodes (most connected - your core abstractions)
1. `Coupler` - 258 edges
2. `assert_allclose_compact()` - 224 edges
3. `Clock` - 218 edges
4. `make_test_grid()` - 206 edges
5. `Exchange` - 181 edges
6. `RectilinearGrid` - 178 edges
7. `ComponentSpec` - 157 edges
8. `CouplerError` - 157 edges
9. `DataComponent` - 132 edges
10. `2026-04-23` - 132 edges

## Surprising Connections (you probably didn't know these)
- `test_top_level_exports_public_orchestration_and_component_author_api()` --indirect_call--> `DataComponent`  [INFERRED]
  tests/test_api_boundaries.py → vercor/components/data.py
- `test_data_component_rejects_active_step_factory()` --indirect_call--> `DataComponent`  [INFERRED]
  tests/test_public_api_contracts.py → vercor/components/data.py
- `test_runtime_topology_maps_are_frozen_read_only_views()` --indirect_call--> `RuntimeTopologyMaps`  [INFERRED]
  tests/test_runtime_facade_boundaries.py → vercor/_runtime/topology_state.py
- `ToyHostModel` --uses--> `Coupler`  [INFERRED]
  examples/custom_component_wrapping.py → vercor/coupler.py
- `ToyHostModel` --uses--> `Exchange`  [INFERRED]
  examples/custom_component_wrapping.py → vercor/exchanges.py

## Import Cycles
- 1-file cycle: `vercor/components/_contracts.py -> vercor/components/_contracts.py`
- 1-file cycle: `vercor/setups/_external/veros_state.py -> vercor/setups/_external/veros_state.py`

## Hyperedges (group relationships)
- **VerCOR 0.4 Architecture** — design_vercor_0_4_design_specification, docs_api_architecture_review_vercor_0_4_api_architecture_review, docs_superpowers_plans_2026_07_13_vercor_0_4_api_protocol_first_rewrite [INFERRED 0.95]
- **Release Evidence Pipeline** — github_workflows_python_package_python_package_ci, docs_releasing_release_procedure, progress_vercor_progress [INFERRED 0.85]
- **VerCOR Output Contract Evolution** — docs_superpowers_specs_2026_07_13_vercor_0_4_api_design_unified_output_coordinator, docs_superpowers_specs_2026_07_14_veros_output_universe_design_veros_output_universe_bug_fix_design, docs_superpowers_specs_2026_07_14_veros_duplicate_dimension_output_design_veros_duplicate_dimension_output_bug_fix_design, docs_superpowers_specs_2026_07_17_period_average_window_start_design_period_average_window_start_timestamp_design [INFERRED 0.85]
- **Measured Test-Suite Optimization** — docs_superpowers_specs_2026_07_16_test_suite_parallelization_design_controlled_pytest_parallelization_design, docs_superpowers_specs_2026_07_16_test_suite_performance_design_test_suite_performance_optimization_design, docs_superpowers_plans_2026_07_16_test_suite_parallelization_controlled_pytest_parallelization_implementation_plan, docs_superpowers_plans_2026_07_16_test_suite_performance_test_suite_artifact_reuse_implementation_plan [INFERRED 0.85]
- **VerCOR 0.4 Contract Consolidation** — docs_superpowers_specs_2026_07_13_vercor_0_4_api_design_vercor_0_4_0a1_protocol_first_api_design, docs_superpowers_specs_2026_07_15_calendar_year_ownership_design_calendar_year_ownership_design, docs_superpowers_specs_2026_07_15_vercor_versioning_design_vercor_pre_1_0_versioning_correction_design, docs_superpowers_specs_2026_07_17_vercor_0_4_deprecation_cleanup_design_vercor_0_4_deprecation_cleanup_design [INFERRED 0.85]

## Communities (175 total, 16 thin omitted)

### Community 0 - "Representative Near Surface State Over"
Cohesion: 0.08
Nodes (58): test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions(), test_era5_atmosphere_helpers_support_jit_and_gradients(), _finite_difference_scalar_grad(), _ocean_state(), Array, ndarray, Representative near-surface state over ocean for bulk-flux tests., test_cdn_and_stability_functions_are_well_behaved() (+50 more)

### Community 1 - "Single Private Normalization Bridge For"
Cohesion: 0.05
Nodes (45): make_differentiable_model(), Update the custom flux on the host runtime path., Wrap a pure JAX callable as a differentiable VerCOR component., _NameItem, Update fields and replace the public runtime payload., Any, Callable convenience adapter for the structural component protocol., Minimal runtime step context passed to component step boundaries. (+37 more)

### Community 2 - "Make Grid"
Cohesion: 0.06
Nodes (83): make_test_grid(), NDArray, test_data_and_callable_factories_return_core_contract_instances(), _coupler(), _prepared_prefill_binding(), Any, Focused coverage for the 0.4 component adapters and private runtime bridge., test_callable_component_adapts_supported_step_arities() (+75 more)

### Community 3 - "Cache Coords"
Cohesion: 0.07
Nodes (54): BaseException, SimpleNamespace, _accumulate_frames(), _ConstructedVerosState, _FakeDynamicsPrediction, _FakeForcing, _FakePhysicsModule, _FakeSettings (+46 more)

### Community 4 - "Custom Component Wrapping"
Cohesion: 0.06
Nodes (57): make_custom_coupler(), make_data_forcing(), make_host_model(), Wrap a Python host-side model while keeping VerCOR runtime fields explicit., Small structural component using the public Component protocol., Minimal custom backend that delegates component stepping to RuntimeDriver., Assemble custom-named components without the built-in surface-mask policy., Wrap static or time-dependent forcing fields without a runtime step. (+49 more)

### Community 5 - "Run Every Step Plan In"
Cohesion: 0.08
Nodes (48): _JaxChunkExecutor, has_period_output(), Return whether the explicit target enables any declared period output., Reject any enabled I/O run carrying traced state leaves., validate_output_run_state_not_traced(), build_jax_chunk_executor(), build_runtime_execution_data(), execute_custom_chunk() (+40 more)

### Community 6 - "With Fields"
Cohesion: 0.08
Nodes (29): _rebuild_state(), _replace_source_grid(), _state_with_duplicate_component_name(), _state_with_nonfinite_mask(), _state_with_out_of_range_mask(), _state_with_unexpected_binary_mask(), _state_with_unexpected_grid_edges(), _state_with_wrong_component_order() (+21 more)

### Community 7 - "Regrid Vector"
Cohesion: 0.06
Nodes (72): StreamHandler, StringIO, capture_logger_output(), RuntimeArray, RecordingRegridder, Build runtime state from a Coupler's components for focused tests., runtime_state_from_coupler_components(), _canonical_handler() (+64 more)

### Community 8 - "Canonicalize Time Last Level Field"
Cohesion: 0.14
Nodes (24): test_era5_land_layout_uses_shared_jax_helpers(), test_jcm_land_layout_uses_shared_jax_helpers(), test_ocean_mask_helpers_accept_jax_arrays(), test_shared_field_helpers_canonicalize_surface_fields_and_positive_masks(), test_shared_masked_surface_field_helper_supports_jit_and_gradients(), test_total_surface_temperature_diagnostic_uses_runtime_view_fields(), canonicalize_time_last_level_field(), canonicalize_time_last_surface_field() (+16 more)

### Community 9 - "Interrupts Components"
Cohesion: 0.14
Nodes (22): JaxRuntimeError, KeyboardInterrupt, NoReturn, _block_until_ready(), _make_pure_coupler(), MonkeyPatch, Signals, test_checkpoint_observes_wakeup_fd_signal_without_python_handler() (+14 more)

### Community 10 - "Workflow Runtime Execution Tests"
Cohesion: 0.14
Nodes (31): Any, MonkeyPatch, Workflow runtime execution tests., test_alternating_jax_workflow_uses_absolute_steps_and_builds_metadata_once(), test_auto_and_forced_host_backend_selection_preserve_behavior(), test_backend_must_execute_every_plan_in_its_chunk(), test_custom_backend_must_return_run_state_for_each_chunk(), test_custom_backend_period_output_is_accumulated_and_written_by_core() (+23 more)

### Community 11 - "Run Data Driver"
Cohesion: 0.07
Nodes (26): Run every step plan in one core-defined chunk., Advance every plan in one core-defined chunk., Build a complete static plan through public workflow contracts., Return the configured entries as an execution plan., Record the chunk and dispatch all of its plans in order., ExecutionContext, Protocol, Public workflow planning and runtime execution extension contracts. (+18 more)

### Community 12 - "Patch The Configured Route With"
Cohesion: 0.10
Nodes (21): Patch the configured route with an explicit target-shaped mask., Any, ndarray, _RecordingTopologyPolicy, test_custom_topology_policy_builds_once_and_patches_route_maps(), test_topology_policy_patch_accepts_valid_binary_and_fractional_masks(), test_topology_policy_patch_rejects_invalid_mask_values(), test_topology_policy_patch_rejects_unknown_keys_and_wrong_shapes() (+13 more)

### Community 13 - "Configuration For The Bundled Camulator"
Cohesion: 0.07
Nodes (39): _camulator_output_conf(), _camulator_prediction(), _make_coupler(), Any, datetime, MonkeyPatch, Path, Tensor (+31 more)

### Community 14 - "Return The Configured Runtime Logger"
Cohesion: 0.12
Nodes (15): _is_dynamic_callback_value(), JaxCallbackLogger, _partition_dynamic(), _partition_dynamic_kwargs(), Any, Small logger wrapper that emits messages through ``jax.debug.callback``., Return the wrapped Python logger name., Return the wrapped Python logger level. (+7 more)

### Community 15 - "Field Names"
Cohesion: 0.07
Nodes (40): test_flatten_fields_and_append_unique(), _ComponentBinding, Single private normalization bridge for protocol-first components., Immutable runtime binding produced once after setup and dtype selection., Return prepared field names in stable insertion order., Private callable-signature normalization for component adapters., Private application of public step results to immutable runtime stores., Any (+32 more)

### Community 16 - "Compute Sigma Pressure Levels"
Cohesion: 0.09
Nodes (33): Any, Return the updated field and a replacement immutable payload., test_unconfigured_real_conversion_preserves_existing_array_dtype(), as_jax_real_array(), Convert ``value`` to a JAX array using VerCOR's real dtype policy., compute_sigma_pressure_levels(), Compute pressure levels from sigma levels and normalized surface pressure., _physical_constants_for_dtype() (+25 more)

### Community 17 - "Has Identical Grids"
Cohesion: 0.05
Nodes (54): Public-only VerCOR extension fixture., PluginAssembly, PluginConfig, PluginFactory, PluginRegridder, PluginRegridderFactory, PluginWorkflow, Any (+46 more)

### Community 18 - "Profile Runtime"
Cohesion: 0.14
Nodes (17): ArgumentParser, _block_until_ready(), build_parser(), _format_result(), main(), profile_runtime(), Any, Run a small timing profile for the scanned runtime. (+9 more)

### Community 19 - "Regrid Vector"
Cohesion: 0.12
Nodes (20): test_common_exchange_recipes_are_centralized_for_examples(), test_fields_facade_owns_vector_field_contract(), ExchangeField, Create an exchange with an explicit or endpoint-derived route ID., _flatten_field_items(), _normalize_field_items(), ExchangeField, Two named runtime fields that should be transferred as one vector. (+12 more)

### Community 20 - "Create A Complete Immutable Coupling"
Cohesion: 0.18
Nodes (15): _ComponentDeclaration, prepare_component(), Run setup once and freeze normalized fields/payload in a runtime binding., Validated public declaration retained until runtime preparation., Minimal setup context passed to component initialization hooks., Freeze the public run-order sequence without splitting text., SetupContext, initialize_coupler_runtime() (+7 more)

### Community 21 - "Dtypes Components"
Cohesion: 0.11
Nodes (36): ShapeLike, test_dtype_policy_disable_x64_maps_real_arrays_to_float32(), test_dtype_policy_enable_x64_maps_real_arrays_to_float64(), test_index_dtype_is_int32_for_both_real_precision_modes(), test_numpy_and_jax_helpers_agree_on_dtype_policy(), as_jax_index_array(), dtype_policy(), DTypePolicy (+28 more)

### Community 22 - "Jax Gcm Output"
Cohesion: 0.09
Nodes (43): _additional_coordinate_values(), _default_physics_module(), _float0_leaf_to_nan(), _float0s_to_nans(), _infer_shape_to_dims(), _iter_data_items(), jax_gcm_coordinate_variables(), jax_gcm_data_variables_with_unit_metadata() (+35 more)

### Community 23 - "Return Setup Owned Field And"
Cohesion: 0.07
Nodes (33): SpinupResult, test_align_model_timestep_rejects_non_divisible_model_step(), test_align_model_timestep_returns_coupling_timestep_and_substeps(), Any, Return setup-owned field and payload state from a setup callback.      ``fields`, Flatten fields and payload while preserving mapping order., Restore a setup result from JAX PyTree leaves., SetupResult (+25 more)

### Community 24 - "Api Boundaries"
Cohesion: 0.07
Nodes (39): Path, Private runtime modules should import Component only for type checking., test_bilinear_interpolator_removes_unused_cartesian_helper(), test_callable_component_has_one_step_normalization_owner(), test_camulator_adapters_share_runtime_cursor_state_transition_helper(), test_camulator_gcm_factory_passes_runtime_step_directly(), test_component_base_internals_are_private_modules(), test_component_contract_modules_share_field_name_deduplication_owner() (+31 more)

### Community 25 - "Field Transfer"
Cohesion: 0.11
Nodes (29): test_runtime_state_is_separate_from_public_component_objects(), test_step_result_payload_sentinel_preserves_runtime_payload_by_default(), _runtime_component_state(), test_field_store_replacements_reject_shape_changes(), test_gradient_flows_through_component_payload(), _RuntimeSendComponent, test_runtime_component_and_coupler_state_are_pytrees(), test_runtime_component_state_preserves_optional_payload_under_jit() (+21 more)

### Community 26 - "Ver Cor 0 4 Unified"
Cohesion: 0.19
Nodes (43): _api(), _component(), _coupler(), _period_spec(), Any, MonkeyPatch, ndarray, Path (+35 more)

### Community 27 - "Focused Tests For The Sole"
Cohesion: 0.08
Nodes (36): _ClockStep, Array, _sample_sum_and_counts(), build_output_plan(), _coordinate_dtypes(), _coordinate_parts(), _coordinate_shapes(), initial_output_session() (+28 more)

### Community 28 - "Grid Geometry"
Cohesion: 0.14
Nodes (11): _coordinate_allclose(), grids_identical(), _RuntimeArray, Return whether two rectilinear grids have the same shape and coordinates., _BaseRegridder, Any, Return whether source and target grids are identical., Return the destination grid using public target terminology. (+3 more)

### Community 29 - "Model Year Seconds"
Cohesion: 0.13
Nodes (30): test_array_only_pytree_round_trip_uses_declared_children(), datetime, test_datetime_to_seconds_in_year_for_datetime(), test_datetime_to_seconds_in_year_for_model_datetime_with_arithmetic(), test_forcing_index_rejects_unknown_year_type(), test_get_periodic_interval_exact_cycle_boundary_resets_to_first_record(), test_get_periodic_interval_exact_last_record_boundary_wraps_to_first(), test_get_periodic_interval_wraps_with_time_beyond_cycle() (+22 more)

### Community 30 - "Run A Coupler Through The"
Cohesion: 0.06
Nodes (93): build_slab_coupler(), Build and initialize a small pure-JAX slab coupler for profiling., create_runtime_state_from_coupler(), prepared_coupling(), Any, RuntimeArray, Create, prime, and validate state using the coupler's installed topology., Return the Coupler's canonical prepared runtime boundary. (+85 more)

### Community 31 - "Regrid Vector"
Cohesion: 0.14
Nodes (32): _block_until_ready(), _component_state(), _identity_factory(), _IdentityRegridder, _make_initial_state(), _make_output_component(), _make_period_output_coupler(), _period_target() (+24 more)

### Community 32 - "Create Runtime State From Coupler"
Cohesion: 0.02
Nodes (132): 2026-04-23, Compile Cache and Safe Donation Runtime Audit, Coupler / Veros / Clock Coverage Expansion, Coverage Outcome, Eighth JAX Translation Slice 8A: CAMulator Boundary, Eighth JAX Translation Slice 8B: JAX-First Example Drivers, Eleventh JAX Translation Slice 11A: ERA5 Land Adapter, Eleventh JAX Translation Slice 11B: JAX-Backed Forcing Read Boundary (+124 more)

### Community 33 - "Run Camulator With Veros"
Cohesion: 0.11
Nodes (27): _default_clock(), main(), _parse_args(), Run the bundled JCM atmosphere/land setup with ERA5 ocean forcing., Parse optional short-run and initialization-only CLI controls., Run the example or return its prepared initial state., Return the historic noleap clock, optionally with a shorter run., Namespace (+19 more)

### Community 34 - "Opaque Mutable Leaf Used To"
Cohesion: 0.12
Nodes (35): _api(), _MutablePayload, _one_step_coupler(), Any, Opaque mutable leaf used to prove host payload ownership., test_callable_component_uses_declared_outputs_and_setup_payload(), test_component_is_the_structural_protocol(), test_component_mapping_arguments_reject_non_mappings() (+27 more)

### Community 35 - "Build Distributions"
Cohesion: 0.09
Nodes (30): TempPathFactory, build_distributions(), BuiltDistributions, _cached_build_pythonpath(), _existing_distributions(), install_local_target(), Path, Offline distribution build/install helpers for artifact-boundary tests. (+22 more)

### Community 36 - "Select Fast Cases"
Cohesion: 0.13
Nodes (35): Protocol, select_fast_cases(), SelectFastCases, ArithmeticCase, ClockIterationCase, StringCase, test_clock_360_rejects_day_31_start(), test_clock_360_rolls_microseconds_across_year_boundary() (+27 more)

### Community 37 - "Public Post Step Component View"
Cohesion: 0.13
Nodes (34): OutputContext, Public post-step component view supplied to an output provider.      ``step`` is, camulator_average_coordinate_variables(), camulator_average_data_variables(), camulator_output_provider(), camulator_period_output_variables(), _CAMulatorOutputProvider, _configured_level_values() (+26 more)

### Community 38 - "Canonicalize External Typing Aliases"
Cohesion: 0.11
Nodes (25): _canonical_public_callable_names(), _canonical_public_method_names(), _documented_public_manifest(), _normalized_signature(), _public_signature_contract(), Executable documentation and release contracts for VerCOR 0.4.0a1., Return every concrete callable from canonical non-root owner manifests., Return public class/protocol behavior, excluding inherited exceptions. (+17 more)

### Community 39 - "Fields Components"
Cohesion: 0.07
Nodes (41): CaptureFixture, test_component_vector_speed_reads_runtime_component_view(), test_component_vector_speed_uses_jax_arrays(), test_runtime_array_to_host_is_canonical_host_transfer(), _diagnostic_component_order(), test_grids_identical_detects_equal_and_unequal_grids(), test_plot_component_scalar_vector_comparison_accepts_callable_scalar(), test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes() (+33 more)

### Community 40 - "Apply Conservative Scalar Regridding"
Cohesion: 0.17
Nodes (25): _grid(), Any, MonkeyPatch, test_conservative_factory_forwards_remapper_options(), test_conservative_factory_returns_conservative_rectilinear_regridder(), test_conservative_regridder_api_does_not_expose_noop_fill_value(), test_identical_grid_regridder_remains_scalar_only(), test_public_conservative_factory_exposes_radius_km_only() (+17 more)

### Community 41 - "Return The Immutable Runtime Policy"
Cohesion: 0.17
Nodes (10): Exception, Public planning, validation, and scheduling contracts for 0.4 workflows., test_runtime_module_owns_only_the_v0_4_workflow_contracts(), test_runtime_options_validates_v0_4_extension_contracts(), test_sequential_workflow_builds_an_empty_plan_for_zero_steps(), test_sequential_workflow_builds_one_frozen_plan_per_clock_step(), test_static_planning_containers_copy_sequence_inputs_to_tuples(), test_workflow_must_return_an_execution_plan() (+2 more)

### Community 42 - "Veros Output"
Cohesion: 0.13
Nodes (32): _active_output_variable_names(), _attrs_for_variable(), _coordinate_dimension_is_extractable(), _current_timestep_index(), _drop_timestep_dim(), _extract_coordinate_variable(), _extract_variable(), extract_veros_output_snapshot() (+24 more)

### Community 43 - "Assert Allclose Compact"
Cohesion: 0.18
Nodes (20): float64, _cell_areas(), _make_remapper(), NDArray, RuntimeArray, test_apply_scalar_shape_mismatch_raises_value_error(), test_apply_scalar_supports_jax_jit_linearity_and_gradients(), test_constant_field_preserved_on_refinement_conservation_mode() (+12 more)

### Community 44 - "Assertions Components"
Cohesion: 0.08
Nodes (48): assert_allclose_compact(), assert_array_equal_compact(), _format_index(), Any, Assert numerical closeness with concise, greppable diagnostics., Assert exact equality with the same compact diagnostics., test_private_bilinear_weight_helper_reproduces_periodic_dateline_indices(), Any (+40 more)

### Community 45 - "V0 4 Public Api"
Cohesion: 0.06
Nodes (42): test_breaking_api_cleanup_removes_transitional_public_surfaces(), test_runtime_state_runs_validation_hooks_outside_run_order(), _InterruptingHostComponent, _NoopRuntimeComponent, test_custom_backend_signal_aborts_through_shared_controller(), test_host_runtime_signal_aborts_through_shared_controller(), _clock(), _component() (+34 more)

### Community 46 - "Struct Time"
Cohesion: 0.09
Nodes (9): struct_time, _ModelDateTimeBase, month_day_from_day_of_year(), Self, timedelta, Return a ``datetime.timetuple()``-compatible model-calendar value., Format the datetime using ``datetime.strftime``-style directives., Return ``(month, day)`` for one-based day-of-year. (+1 more)

### Community 47 - "Run Jcm With Era5Data"
Cohesion: 0.08
Nodes (42): build_coupler(), Build the example coupler, reusing supplied model/data objects., test_camulator_constructor_builds_jax_backed_grid(), MonkeyPatch, test_make_jcm_land_atmosphere_accepts_jax_gcm_config(), MonkeyPatch, test_make_jcm_land_atmosphere_accepts_preloaded_inputs(), test_make_jcm_land_atmosphere_patches_mask_and_options() (+34 more)

### Community 48 - "Get Loc"
Cohesion: 0.14
Nodes (14): Path, slice, _RecordingLogger, test_camulator_runtime_cursor_initializes_indexes_and_advances(), test_grid_field_defaults_returns_defaults_with_overrides(), test_initialize_camulator_forcing_cursor_accepts_integer_index(), test_initialize_camulator_forcing_cursor_returns_index_and_warns_on_mismatch(), test_load_jcm_inputs_facade_returns_named_payload() (+6 more)

### Community 49 - "Repository Wide Contracts For Ver"
Cohesion: 0.15
Nodes (28): _forbidden_api_tokens(), _forbidden_exact_release_labels(), _forbidden_release_shorthand_labels(), _ownership_matrix_line(), MonkeyPatch, Path, Repository-wide contracts for VerCOR's supervised pre-1.0 versioning., Return exact old labels only when the line establishes VerCOR ownership. (+20 more)

### Community 50 - "Calendar Components"
Cohesion: 0.16
Nodes (14): log_scanned_component_progress(), log_scanned_step_progress(), datetime, ModelDateTime, RuntimeArray, timedelta, Return the shared host/scanned runtime step progress message., Return the shared host/scanned runtime component progress message. (+6 more)

### Community 51 - "Capture Logger Output"
Cohesion: 0.16
Nodes (22): MonkeyPatch, test_check_remap_conservation_handles_skip_and_mismatch(), test_check_total_lnd_ocn_mask_sum_success_and_failure(), test_compute_ocn_lnd_masks_on_atm_grid_clips_and_builds_binary_land_mask(), test_create_lnd_mask_from_ocn_accepts_jax_backed_masks(), Base class for exceptions during regridding operations., RegridderError, check_remap_conservation() (+14 more)

### Community 52 - "Bilinear Rectilinear"
Cohesion: 0.02
Nodes (104): 2026-05-15: Conservative Helper and Compatibility Cleanup, 2026-05-15: Maintainability Follow-Up, 2026-05-15: Runtime Helper Consolidation, 2026-05-26: Code Organization Audit Implementation, 2026-05-26: Ownership Boundary Refactor Follow-Up, 2026-05-26: Refactoring Campaign Ownership Split, 2026-05-27: Boundary Cohesion Refactor, 2026-05-27: Callable Component Boundary Refactor (+96 more)

### Community 53 - "Ver Cor 0 4 Route"
Cohesion: 0.12
Nodes (38): test_coupler_rejects_string_run_order(), _clock(), _components(), _coupler(), _interleaved_route_coupler(), Any, VerCOR 0.4 route identity, topology, regridding, and state contracts., _state_coupler() (+30 more)

### Community 54 - "Base Class For Exceptions Inside"
Cohesion: 0.11
Nodes (31): _array_is_traced(), _array_leaf_metadata(), Any, _raise_if_false(), Validate exact runtime store names, grid shapes, and dtypes., Reject missing, extra, duplicate, or reordered runtime names., Validate one differentiable component payload's static PyTree schema., Return array shape/dtype metadata when a payload leaf is numeric. (+23 more)

### Community 55 - "Conftest Components"
Cohesion: 0.11
Nodes (23): CaseT, Config, FixtureRequest, Item, Parser, fast_mode(), pytest_addoption(), pytest_collection_modifyitems() (+15 more)

### Community 56 - "Init Context"
Cohesion: 0.22
Nodes (21): CoverageCouplerStub, datetime, _fake_jcm_land_inputs(), _install_data_driver_factory_fakes(), _prepare_component_for_test(), Any, datetime, MonkeyPatch (+13 more)

### Community 57 - "Regrid Vector"
Cohesion: 0.21
Nodes (20): _make_grid(), Any, MonkeyPatch, test_bilinear_factory_forwards_interpolator_options(), test_bilinear_factory_returns_bilinear_rectilinear_regridder(), test_regridder_constructor_propagates_interpolator_options(), test_regridder_constructor_sets_interpolator_and_grids(), test_regridder_has_identical_grids_false_for_different_coords() (+12 more)

### Community 58 - "Array To Host"
Cohesion: 0.24
Nodes (13): MonkeyPatch, test_transposed_host_array_uses_canonical_host_transfer(), array_to_host(), host_int64_array(), Any, NDArray, RuntimeArray, Transfer an array-like value to host memory for I/O-only consumers. (+5 more)

### Community 59 - "Advance The Private Host Backed"
Cohesion: 0.11
Nodes (25): GlobalFourDegreeSetup, Any, Advance the private host-backed Veros ocean boundary., step_veros_runtime(), CustomGlobalFourDegree, Veros global 4-degree setup with VerCOR-controlled forcing fields., advance_veros_substeps(), apply_veros_forcing_fields() (+17 more)

### Community 60 - "Assets Components"
Cohesion: 0.22
Nodes (23): MonkeyPatch, Path, test_asset_base_url_normalizes_and_handles_empty(), test_download_asset_writes_response_bytes(), test_ensure_registered_asset_downloads_when_cached_md5_invalid(), test_ensure_registered_asset_errors_without_base_url(), test_ensure_registered_asset_raises_and_deletes_on_md5_mismatch(), test_ensure_registered_asset_uses_valid_cached_file() (+15 more)

### Community 61 - "Mark Variables That Are Recognized"
Cohesion: 0.12
Nodes (15): _append_indexed_variables(), _mark_unavailable_variables(), slice, CAMulator tensor indexing and xarray-to-Torch staging helpers., Mark variables that are recognized by config but absent from a tensor type., Build index mappings for every supported tensor type., Build indices for pure state tensors with no forcing or diagnostics., Typed channel metadata for one CAMulator tensor variable. (+7 more)

### Community 62 - "Convert Numpy Jax Arrays To"
Cohesion: 0.11
Nodes (33): ScalarPhysicsValue, test_compute_hybrid_pressure_levels_matches_hybrid_definition(), _float_dtype_of(), dtype, ndarray, Convert numpy/jax arrays to a numpy.ndarray (without requiring jax)., With same T and pressure thickness, adding humidity increases Tv and thus thickn, Specific humidity q should not be treated as water-vapor mixing ratio. (+25 more)

### Community 63 - "Extrapolate Scalar Field"
Cohesion: 0.11
Nodes (21): extrapolate_scalar_field(), Array, Return source points available for scalar interpolation or extrapolation., Fill target points from nearest or IDW valid source values., valid_scalar_source_mask(), all_negative(), all_positive(), great_circle_distance_rad() (+13 more)

### Community 64 - "Shared H5Netcdf Output Writer For"
Cohesion: 0.21
Nodes (20): File, Path, test_write_netcdf_dataset_logs_filename_when_logger_is_supplied(), test_write_netcdf_dataset_rejects_conflicting_dimension_sizes(), test_write_netcdf_dataset_writes_scalar_data_variables(), _coordinate_variables(), _mean_variables(), Path (+12 more)

### Community 65 - "Reject Any Enabled I O"
Cohesion: 0.17
Nodes (19): RuntimeArray, Create immutable runtime state from component setup objects., runtime_state_from_components(), prime_runtime_outgoing(), Populate outgoing stores once before the first exchange dispatch., _final_snapshot_time(), datetime, ModelDateTime (+11 more)

### Community 66 - "Base Class For Exceptions Related"
Cohesion: 0.14
Nodes (16): test_top_level_exports_public_exceptions(), test_centers_to_edges_and_compute_land_mask_edge_cases(), test_exchange_stores_factory_and_formatting_without_create_wrapper(), test_exchange_uses_wrapped_factory_name_and_keeps_partial_options(), test_helper_kernels_support_jax_jit(), test_rectilinear_grid_owns_grid_behavior_without_private_base(), test_rectilinear_grid_pytree_round_trip_preserves_arrays(), GridError (+8 more)

### Community 67 - "Day Of Year"
Cohesion: 0.10
Nodes (29): StrEnum, test_calendar_owns_canonical_year_types_and_durations(), test_calendar_resolves_year_type_from_existing_clock_policy(), test_calendar_year_helpers_reject_foreign_policy_values(), _Time, CalendarDate, day_of_year_from_month_day(), is_leap_year() (+21 more)

### Community 68 - "Shared Net Cdf Writer Boundary"
Cohesion: 0.04
Nodes (40): [0.4.0a1] - 2026-07-14, Added, Changed, Changelog, Compatibility, Removed, 1. Executive summary, 2. Duplication map (+32 more)

### Community 69 - "Initialize Camulator Runtime Fields"
Cohesion: 0.14
Nodes (20): _camulator_output_array(), initialize_camulator_runtime_fields(), map_camulator_prediction_arrays(), map_camulator_prediction_to_runtime_fields(), prepare_camulator_dynamic_forcing_chunk(), prepare_camulator_sst_input(), prepare_camulator_surface_forcing(), Any (+12 more)

### Community 70 - "Apply Scalar"
Cohesion: 0.07
Nodes (35): test_registered_pytree_classes_inherit_shared_flatten_methods(), test_remapper_pytree_round_trip_uses_declared_metadata_only(), test_static_pytree_metadata_round_trip_uses_declared_aux_fields(), ConservativeRectilinearRemapper, Any, Array, RuntimeArray, Ensure latitude bounds are monotonically increasing. (+27 more)

### Community 71 - "Build Output Plan"
Cohesion: 0.06
Nodes (30): 1. Code Organization, 1. Tests are everything, 2. Code Style, 2. Concise test output (context window hygiene), 3. Fast tests to avoid time blindness, 3. Testing, 4. Keep PROGRESS.md current (agent orientation), 5. Prevent regressions (CI discipline) (+22 more)

### Community 72 - "Iter Fields"
Cohesion: 0.05
Nodes (41): FieldLookupScope, FieldScope, test_boundary_redesign_removes_remaining_duplicate_public_helpers(), test_runtime_private_state_uses_public_domain_vocabulary(), test_state_constructors_do_not_expose_runtime_stores(), test_prepared_coupling_owns_single_normalized_runtime_boundary(), test_runtime_topology_state_groups_read_only_maps(), _replace_component_store() (+33 more)

### Community 73 - "Stable Public Output Contracts For"
Cohesion: 0.17
Nodes (13): Return ``(nlat, nlon)`` for horizontal grid-shaped fields., _host_value(), _canonical_metadata_value(), _frozen_mapping(), _frozen_metadata(), Any, Return array rank without importing a host-array implementation., Create and validate one immutable provider frame. (+5 more)

### Community 74 - "Run Optional Author Validation With"
Cohesion: 0.07
Nodes (37): test_prefill_normalizes_all_store_scalars_and_dtypes_before_update(), test_prefill_rejects_fields_absent_from_exchange_contract(), test_prefill_rejects_non_grid_runtime_store_shapes(), test_prefill_rejects_nonnumeric_store_values_with_component_error(), _copy_owned_pytree(), _normalize_prefill_contract_store(), Any, _ComponentStepReturn (+29 more)

### Community 75 - "Public Payload Passed To Component"
Cohesion: 0.12
Nodes (25): test_output_mask_names_remain_unique_after_route_token_sanitizing(), test_grid_field_dims_is_the_single_output_layout_rule(), grid_field_dims(), Return stable generic dimensions for one optional grid-shaped field., Public payload passed to component snapshot writers.      ``time`` is the model, SnapshotContext, _component_output_filenames(), output_masks_for_component() (+17 more)

### Community 76 - "Make Example Grid"
Cohesion: 0.09
Nodes (22): make_example_grid(), Return a small grid for custom component wrapper examples., Structural host component implemented outside VerCOR., StructuralHostComponent, test_grid_constructors_live_on_rectilinear_grid_class(), DummyComponentA, DummyComponentB, DummyGridComponent (+14 more)

### Community 77 - "Class Body Source"
Cohesion: 0.09
Nodes (27): class_body_source(), package_import_cycles(), Return the source segment for one top-level class., Return top-level import cycles within one package directory., Return repository source text for architecture-boundary assertions., source_for(), test_external_package_has_no_top_level_import_cycles(), test_output_package_has_no_top_level_import_cycles() (+19 more)

### Community 78 - "Apply Wind Filter To Tensor"
Cohesion: 0.15
Nodes (22): _anisotropic_gaussian_kernel(), apply_wind_filter_to_tensor(), build_wind_filter_artifacts(), filter_field(), _isotropic_gaussian_kernel(), _odd_kernel_size(), device, dtype (+14 more)

### Community 79 - "Canonical Data Layout Description"
Cohesion: 0.13
Nodes (22): FieldNames, test_canonical_grid_field_shape_error_is_shared(), test_canonical_grid_field_shape_normalizes_array_shape(), test_validate_canonical_grid_field_shape_raises_consistent_error(), declared_runtime_field_names(), Private component field normalization and declaration helpers., Return all input/output names declared by a component spec., canonical_data_layout_description() (+14 more)

### Community 80 - "Boundary Tests For Lazy Bundled"
Cohesion: 0.22
Nodes (14): MonkeyPatch, Path, Boundary tests for lazy bundled setup imports and configuration., Return the package root selected for fresh-process boundary probes., _run_missing_dependency_probe(), _run_setup_probe(), test_camulator_enabled_spinup_fails_before_runtime_configuration(), test_lazy_factory_attribute_access_loads_only_lightweight_factory_module() (+6 more)

### Community 81 - "Atmosphere Components"
Cohesion: 0.22
Nodes (14): test_atmosphere_kernels_support_jit_and_gradients(), test_land_kernel_supports_jit_and_clipping(), test_ocean_kernel_supports_jit_and_matches_closed_form(), test_seaice_kernel_supports_jit_and_gradient(), _bulk_flux_step(), _default_sea_surface_temperature(), Array, _surface_wind_10m() (+6 more)

### Community 82 - "First Development"
Cohesion: 0.06
Nodes (30): Test-First Development, VerCOR Development Guide, VerCOR 0.4.0a1, VerCOR Module Dependency Order, 10. Testing and release evidence, 1. Goals and constraints, 2. Public boundary, 3. Configuration ownership (+22 more)

### Community 83 - "Is Period End"
Cohesion: 0.13
Nodes (22): test_is_period_end_stays_false_within_same_day(), test_time_coordinate_variable_preserves_calendar_attrs(), test_used_dimension_names_preserves_first_use_order_and_excludes_time(), datetime, ModelDateTime, Shared dataset assembly helpers for NetCDF output adapters., Return the one-step NetCDF time coordinate for an output dataset., Return non-excluded dimensions in first-use order across variables. (+14 more)

### Community 84 - "V0 4 Physics"
Cohesion: 0.24
Nodes (15): _physical_constants_type(), Any, test_dtype_helpers_reject_settings_as_a_precision_owner(), test_flux_consumes_canonical_constants_and_differentiates_through_them(), test_physical_constants_are_a_frozen_registered_pytree(), test_physical_constants_are_keyword_only(), test_physical_constants_document_every_field_and_ambiguous_units(), test_physical_constants_have_canonical_names_and_preserved_defaults() (+7 more)

### Community 85 - "Iterator Over Simulation Time Steps"
Cohesion: 0.20
Nodes (8): CalendarType, datetime, ModelDateTime, timedelta, Iterator over simulation time steps in synthetic model calendars., Iterator over simulation steps using the configured stepping strategy., Create a calendar-aware model clock., Iterator over Gregorian datetimes anchored at `start`.

### Community 86 - "Make Veros Gcm"
Cohesion: 0.07
Nodes (30): 2026-04-28, Data Driver Runtime-State Plotting Fix, Incremental Runtime Bridge Simplification, Internal Runtime Responsibility Cleanup, Notes / Failed Approaches, Notes / Failed Approaches, Notes / Failed Approaches, Notes / Failed Approaches (+22 more)

### Community 87 - "External Tools Coverage"
Cohesion: 0.12
Nodes (14): _FakeVariableStore, _FakeVerosState, Any, MonkeyPatch, ndarray, Path, test_load_jcm_coords_terrain_forcing_uses_expected_paths(), test_veros_compute_fluxes_preserves_sign_conventions() (+6 more)

### Community 88 - "Tests For The Injectable Jcm"
Cohesion: 0.21
Nodes (8): _FakeCoupler, MonkeyPatch, Tests for the injectable JCM/ERA5 example entry point., _RecordingRunCoupler, test_build_coupler_default_workflow_keeps_historic_clock(), test_build_coupler_uses_injected_ocean_inputs_and_clock(), test_cli_modes_use_requested_step_count_and_state_path(), test_example_module_import_is_safe_without_jcm()

### Community 89 - "Camulator Stepper"
Cohesion: 0.21
Nodes (12): CAMulator state transformation and model stepping helpers., apply_wind_artifact_filter_to_tensor(), load_wind_filter_config(), post_process_wind_artifacts(), Any, Tensor, Public CAMulator wind artifact filtering facade., Apply CAMulator wind artifact filtering to selected tensor channels. (+4 more)

### Community 90 - "Actual Averaging Window Start"
Cohesion: 0.18
Nodes (10): Actual Averaging-Window Start, Architecture and data flow, Error handling and invariants, Independent Schema Window Starts, Period-Average Window-Start Timestamp Design, Purpose, Root cause, Scope boundaries (+2 more)

### Community 91 - "Package Import Cycles"
Cohesion: 0.18
Nodes (21): test_default_logger_uses_vercor_logger_name(), test_setup_logger_formats_traced_values_under_scan(), Set up and return the callback-backed VerCOR logger., setup_logger(), configure_python_logger(), get_default_logger(), _install_canonical_handler(), normalize_log_level() (+13 more)

### Community 92 - "Structural Differentiable Component Implemented Outside"
Cohesion: 0.09
Nodes (22): 0) Spherical Coordinates vs. Geographical Spherical Coordinates, 10) Properties & remarks, 1) Grids, indexing, and notation, 2) Periodic longitude wrapping, 3.1) Forward (wrapped) longitudinal difference, 3.2) Latitudinal fraction, 3) Cell search and local bilinear coordinates, 4) Bilinear shape functions (weights) (+14 more)

### Community 93 - "Apply Scalar"
Cohesion: 0.32
Nodes (4): BilinearRectilinearInterpolator, Any, Array, Bilinear interpolator for rectilinear lat/lon grids with:      - periodic longit

### Community 94 - "Return Typed Indexing Metadata For"
Cohesion: 0.28
Nodes (5): Tensor, Return typed indexing metadata for a configured CAMulator variable., Extract a named variable view from a CAMulator tensor., Set a named variable in a CAMulator tensor in place., Raise a user-facing error if the variable is absent from the tensor.

### Community 95 - "Load Optional Setup Factories Only"
Cohesion: 0.20
Nodes (11): __getattr__(), Any, Load optional setup factories only when requested., lazy_export_names(), LazyExport, Any, Lazy import helpers for setup package export surfaces., Describe one lazily resolved package export. (+3 more)

### Community 96 - "Run Every Readme Python Block"
Cohesion: 0.27
Nodes (10): _assert_public_imports_only(), MonkeyPatch, Path, _python_fences(), Run every README Python block together, outside the repository directory., Execute the supported 0.4 migration result and verify its observable state., Return Python snippets from Markdown in source order., Reject imports from underscored VerCOR modules in a documentation snippet. (+2 more)

### Community 97 - "Set Up And Return The"
Cohesion: 0.12
Nodes (24): test_create_surface_exchange_masks_rejects_missing_ocean_binary_mask(), test_create_surface_exchange_masks_rejects_non_identical_land_and_atmosphere_grids(), test_validate_land_mask_consistency_rejects_shape_and_value_mismatches(), test_topology_module_owns_public_topology_contracts(), test_surface_mask_policy_is_public_core_configuration(), build_surface_mask_topology_patch(), create_surface_exchange_masks(), Protocol (+16 more)

### Community 98 - "Initialize Camulator Forcing Cursor"
Cohesion: 0.09
Nodes (25): CAMulatorForcingCursor, CamulatorRuntimeCursor, initialize_camulator_forcing_cursor(), load_camulator_forcing_context(), parse_datetime_from_config(), Any, datetime, CAMulator config loading and forcing-time cursor helpers. (+17 more)

### Community 99 - "Camulator Init"
Cohesion: 0.33
Nodes (5): prepare_static_forcing_tensor(), Any, Dataset, Prepare static CAMulator forcing through an explicit xarray/Torch boundary., Initialize a variable accessor for the requested tensor type.

### Community 100 - "Make Differentiable Model"
Cohesion: 0.15
Nodes (18): _frame(), Focused tests for the sole immutable output accumulator and layout helper., test_output_accumulator_canonicalizes_array_metadata_for_jit_reuse(), test_output_accumulator_is_an_immutable_jax_pytree(), test_output_accumulator_preserves_nanmean_counts_without_mutation(), test_output_accumulator_reduces_named_sample_dimension(), test_output_accumulator_rejects_changed_variables_dimensions_and_shape(), test_output_accumulator_replace_preserves_pytree_structure() (+10 more)

### Community 101 - "Emit A Debug Message"
Cohesion: 0.11
Nodes (14): Return the configured runtime logger., LoggerLike, Any, Protocol, Logger interface used across Python and JAX callback runtimes., Emit a debug message., Emit an informational message., Emit a warning message. (+6 more)

### Community 102 - "Build Input With Forcing"
Cohesion: 0.12
Nodes (14): Module, initialize_camulator(), Any, Initialize CAMulator model state and supporting runtime objects., CAMulatorStepper, Any, device, Tensor (+6 more)

### Community 103 - "Controlled Pytest Parallelization Implementation Plan"
Cohesion: 0.50
Nodes (4): Worker Cache Isolation, Rejected Artifact Reuse Experiment, Measured pytest-xdist Default, Timing-Gate Rejection

### Community 104 - "Plot Component Scalar Vector Comparison"
Cohesion: 0.12
Nodes (13): _can_install_signal_handlers(), _close_fd(), default_runtime_interrupt_signals(), Signals, Clear any pending cancellation request., Return terminal signals that request graceful runtime cancellation., Small nonblocking pipe used by ``signal.set_wakeup_fd``., Install this pipe as the process wakeup fd. (+5 more)

### Community 105 - "Validate The Exact Structural Component"
Cohesion: 0.25
Nodes (5): Validate the exact structural component contract immediately., validate_component_contract(), Return the component's rectilinear grid., Return the immutable component declaration., Return the unique component name.

### Community 106 - "Initial State"
Cohesion: 0.25
Nodes (4): Return the one lazily prepared runtime boundary., Prepare components, topology, contracts, and runtime dispatch once., Create and validate the coupled runtime state., Run the configured workflow and optionally write selected outputs.          ``ou

### Community 107 - "Initialize Model State Helpers And"
Cohesion: 0.16
Nodes (18): test_erainterim_helpers_prepare_jax_backed_grid_and_masked_fields(), MonkeyPatch, Path, test_get_forcing_data_valid_and_invalid_file_type(), get_forcing_data(), Path, Resolve setup forcing data to cached assets in $HOME/.vercor/assets., _assemble_erainterim_field() (+10 more)

### Community 108 - "Camulator Contracts"
Cohesion: 0.33
Nodes (4): test_camulator_runtime_field_names_have_lightweight_contract_owner(), camulator_runtime_field_defaults(), Lightweight CAMulator runtime field contract ownership., Return scalar defaults for all CAMulator runtime exchange fields.

### Community 109 - "Bundled Native Output Paths Use"
Cohesion: 0.48
Nodes (6): Bundled native output paths use ordinary providers and core coordination., _source(), test_bundled_factories_install_native_output_providers(), test_camulator_native_period_output_uses_run_level_paths(), test_core_output_session_owns_native_output_boundaries(), test_native_output_modules_return_output_frames()

### Community 110 - "Veros Runtime Settings"
Cohesion: 0.16
Nodes (13): test_model_setup_factories_use_the_public_setup_owner(), _load_veros_implementation(), make_veros_gcm(), Any, Veros ocean component factory., Load Veros implementation owners after runtime configuration., Return a host-backed Veros GCM component., configure_veros_runtime() (+5 more)

### Community 111 - "Camulator Imports"
Cohesion: 0.12
Nodes (17): 2026-05-15, Conservative Bilinear Helper Cleanup, Conservative Compatibility Cleanup, Maintainability Audit Follow-Up Consolidation, Private Compatibility Shim Removal, Private Runtime Helper Consolidation, Runtime FieldStore Compatibility Audit, Runtime/Setup Helper Boundary Cleanup (+9 more)

### Community 112 - "Return A 4D Anisotropic Gaussian"
Cohesion: 0.21
Nodes (11): _component(), _factory(), Any, Array, _ScalingRegridder, test_dispatch_component_exchanges_handles_scalar_masks_and_gradients(), test_dispatch_component_exchanges_preserves_vector_regridding_behavior(), test_runtime_dispatch_context_groups_exchanges_by_destination() (+3 more)

### Community 113 - "Load Jcm Coords Terrain Forcing"
Cohesion: 0.33
Nodes (6): load_jcm_coords_terrain_forcing(), CoordinateSystem, ForcingData, Path, TerrainData, Generate JCM coordinates, forcing and topography files at the specified resoluti

### Community 114 - "Set Diagnostics"
Cohesion: 0.13
Nodes (15): 2026-05-07, Additive Component Authoring API Polish, Component API Internal Split, Component Authoring API Polish and Adapter Rewrite, Component Authoring API Refinement, Component Authoring Facade Refinement, Helper-First Component Wrapping API, User-Friendly Component Wrapping API (+7 more)

### Community 115 - "Use Site"
Cohesion: 0.40
Nodes (4): exercise_plugin(), Path, Mypy use site for the installed public plugin fixture., Return typed smoke evidence from the public plugin.

### Community 116 - "Return Normalized Declarations For Private"
Cohesion: 0.13
Nodes (15): 2026-05-08, Component Author API Cleanup, Component Constructor Boilerplate Tightening, Component Runtime Boilerplate Refactor, Component Runtime Field Adapter Extraction, Coupler Runtime Adapter Refactor, Runtime Module Ownership Refactor, Time-Dependent Data Field Runtime Validation Fix (+7 more)

### Community 117 - "Create A Public Component State"
Cohesion: 0.13
Nodes (15): 2026-05-12, Callable Field Seeding API Removal, Configured Regridder Factory Forwarding, Hypsometric Altitude Corrections, Precision Policy Consistency Audit, Redundant `required_fields` Component API Removal, Runtime Profiling and Core Dispatch Optimization, Shared PyTree Mixin Refactor (+7 more)

### Community 118 - "Positive Current Contract Testing"
Cohesion: 0.07
Nodes (28): File Structure, Global Constraints, Positive Current-Contract Testing, Task 1: Remove Historical 0.3 Distribution Evidence, Task 2: Replace Mutable Component Test Compatibility Helpers, Task 3: Replace the Mutable Output Test Adapter, Task 4: Remove Obsolete Negative Guards and Legacy-Looking Test Names, Task 5: Update Active Documentation and Repository Memory (+20 more)

### Community 119 - "Smoke Components"
Cohesion: 0.26
Nodes (12): Path, test_read_forcing_flips_requested_latitude_axis(), test_read_forcing_reports_missing_mapping_key(), test_read_forcing_reports_missing_netcdf_variable(), test_read_forcing_transposes_file_layout_to_runtime_layout(), test_read_forcing_wraps_broken_netcdf_files(), ndarray, _RuntimeArray (+4 more)

### Community 120 - "Production Numpy Boundaries"
Cohesion: 0.67
Nodes (3): _imports_numpy(), Path, test_numpy_imports_match_explicit_host_boundaries()

### Community 121 - "Dummy Grid Component"
Cohesion: 0.14
Nodes (11): Advance the next exact chunk plan after strict state validation., Return the next plan or reject forged, repeated, and reordered plans., datetime, ModelDateTime, RuntimeArray, Advance one component through dispatch, receive, step, and send phases., step_runtime_component(), Execute ``plan.components`` and return the resulting runtime state.          Arg (+3 more)

### Community 122 - "Reject Invalid Nested Callbacks At"
Cohesion: 0.50
Nodes (3): Reject invalid nested callbacks at configuration time., Validate one optional lifecycle callback immediately., _validate_callback()

### Community 123 - "Return The Registered Component Name"
Cohesion: 0.14
Nodes (13): 2026-05-01, 2026-05-04, 2026-05-13, Canonical Component Data Dimension Order, Canonical VerCOR Logging Format, Centralized VerCOR Dtype Policy, ERA5 Atmosphere Pure Data Component, Explicit Component Author Contracts (+5 more)

### Community 124 - "Unified Opt In Output"
Cohesion: 1.00
Nodes (3): Unified Opt-In Output, Veros Duplicate-Dimension Output Fix, Veros Output Universe Fix

### Community 125 - "Calendar Year Ownership Design"
Cohesion: 0.20
Nodes (9): Calendar API, Calendar Year Ownership Design, Error Handling and Compatibility, Goal, Runtime Data Flow, Scope, Single Calendar Owner, Testing (+1 more)

### Community 126 - "Return Selected Default Jcm Parameter"
Cohesion: 0.67
Nodes (3): _default_jcm_parameter_values(), Parameters, Return selected default JCM parameter values for the example script.

### Community 131 - "Private Runtime Modules Should Import"
Cohesion: 0.14
Nodes (13): Final Acceptance, Global Constraints, Task 10: Finish architecture review, migration docs, release metadata, and CI, Task 1: Freeze the 0.3.2 baseline and record the approved specification, Task 2: Introduce typed physical constants and single precision ownership, Task 3: Replace component authoring with the protocol-first contract, Task 4: Make assembly constructor-only and close public module boundaries, Task 5: Add route IDs, regridder capabilities, route topology, and strict state (+5 more)

### Community 132 - "Path Components"
Cohesion: 0.14
Nodes (13): Baseline and acceptance metrics, Benchmark protocol, Controlled pytest parallelization design, Coverage equivalence, Default command decision, Determinism and state-leak checks, Expected retained deliverables, Objective (+5 more)

### Community 134 - "Validate Cadence And Freeze A"
Cohesion: 0.09
Nodes (18): _BoundaryCall, _clock(), _component(), Path, Regression tests for final VerCOR 0.4 public-boundary review findings., _run_state_components(), _setup_context(), test_prepared_binding_does_not_delegate_private_markers_and_uses_spec_output() (+10 more)

### Community 137 - "Copy Mutable Caller Provided Mappings"
Cohesion: 0.14
Nodes (13): 1. Remove dead state and unreachable code, 2. Consolidate regridding behavior, 3. Simplify output internals, 4. Remove narrow runtime indirection, 5. Reduce bilinear-interpolator PyTree state, 6. Copy Veros state once per forcing update, Constraints, Design (+5 more)

### Community 143 - "2026-04-29"
Cohesion: 0.15
Nodes (13): 2026-04-29, Compatibility Boundary Simplification, Internal Runtime Compatibility Seam Cleanup, Residual Compatibility Marker Cleanup, Source Boundary Simplification, Source Simplification Audit, Unified Coupler Runtime Entrypoint, Validation (Compatibility Boundary Simplification, 2026-04-29) (+5 more)

### Community 144 - "2026-05-14"
Cohesion: 0.15
Nodes (13): 2026-05-14, Conservative Architectural Redundancy Cleanup, Factory-Based Setup Components Refactor, JAXGCM Test-Only Compatibility Surface Removal, Lazy Optional Setup Adapter Imports, Maintainability Audit Refactor Implementation, Setup Lifecycle Helper Consolidation, Validation (Conservative Architectural Redundancy Cleanup, 2026-05-14) (+5 more)

### Community 145 - "File Structure"
Cohesion: 0.15
Nodes (12): File Structure, Global Constraints, Task 1: Collapse the private grid hierarchy and shared scalar regridding, Task 2: Remove unread component, data, store, and flux state, Task 3: Centralize output dimensions and simplify immutable reconstruction, Task 4: Return runtime topology maps directly and inline role lookup, Task 5: Remove one-use runtime execution wrappers, Task 6: Remove unreachable CAMulator modes and inert spinup state (+4 more)

### Community 146 - "run_camulator_prediction_block"
Cohesion: 0.18
Nodes (13): coerce_camulator_datetime(), Any, datetime, RuntimeArray, Tensor, Advance the private host-backed CAMulator atmosphere boundary., Return a Python datetime from CAMulator/xarray time coordinates., Run one CAMulator forcing block and return its predictions and final TS. (+5 more)

### Community 147 - "normalize_component_step_callable"
Cohesion: 0.17
Nodes (10): _AuthorStepCallable, _ComponentStepCallable, Any, RuntimeArray, Validate configuration and normalize the callable signature., Delegate one model step through the normalized callable signature., _component_step_signature_error(), normalize_component_step_callable() (+2 more)

### Community 148 - "Milestone Timeline"
Cohesion: 0.18
Nodes (11): 2026-04-27 to 2026-04-23: JAX Translation and Unified Runtime Foundation, 2026-04-30 to 2026-04-28: Runtime Package and Boundary Refactors, 2026-05-04 to 2026-05-01: Data Layout and Data Components, 2026-05-05: Runtime Interrupt Handling, 2026-05-06: Settings and Lifecycle Logging, 2026-05-07: Component Authoring API, 2026-05-08: Runtime Ownership and Component Boilerplate, 2026-05-12: Precision, Performance, and API Simplification (+3 more)

### Community 149 - "Test-suite performance optimization design"
Cohesion: 0.18
Nodes (10): Baseline, Behavioral equivalence, Failure handling and isolation, Focused timing record, Historical objective, Outcome, Planned implementation and validation sequence, Rejected experimental architecture (+2 more)

### Community 150 - "Veros Duplicate-Dimension Output Bug-Fix Design"
Cohesion: 0.20
Nodes (9): Design, Error Handling, Intended Behavior, Non-Goals, Problem, Representable Veros Output Universe, Testing, Veros Duplicate-Dimension Output Bug-Fix Design (+1 more)

### Community 151 - "canonicalize_external_typing_aliases"
Cohesion: 0.20
Nodes (9): canonicalize_external_typing_aliases(), Stable rendering support for source and installed public-signature tests., Replace evidenced dependency-sensitive aliases with public tokens., Keep equivalent NumPy aliases stable across dependency renderings., Keep equivalent JAX aliases stable without freezing private names., Avoid canonicalizing strings that are not the evidenced aliases., test_external_jax_arraylike_renderings_have_one_public_token(), test_external_numpy_ndarray_renderings_have_one_public_token() (+1 more)

### Community 153 - "2026-04-30"
Cohesion: 0.22
Nodes (9): 2026-04-30, JAX Callback Runtime Logging, Public/Runtime API Boundary Clarification, Runtime Context Boundary Cleanup, Runtime Package Refactor, Validation (JAX Callback Runtime Logging, 2026-04-30), Validation (Public/Runtime API Boundary Clarification, 2026-04-30), Validation (Runtime Context Boundary Cleanup, 2026-04-30) (+1 more)

### Community 154 - "2026-05-05"
Cohesion: 0.22
Nodes (9): 2026-05-05, Compiled Runtime Wakeup-Fd Interrupt Handling, JAXGCM Forcing Payload Scan Shape Stability, Scanned Runtime Progress Logging, Unified Runtime Interrupt Handling, Validation (Compiled Runtime Wakeup-Fd Interrupt Handling, 2026-05-05), Validation (JAXGCM Forcing Payload Scan Shape Stability, 2026-05-05), Validation (Scanned Runtime Progress Logging, 2026-05-05) (+1 more)

### Community 155 - "VerCOR 0.4.0a1 Protocol-First API Design"
Cohesion: 0.22
Nodes (8): Compatibility and release, Coupling contracts, Execution and output, Goal, Historical implementation status, Public architecture, Structural Component Contract, VerCOR 0.4.0a1 Protocol-First API Design

### Community 156 - "VerCOR Pre-1.0 Versioning Correction Design"
Cohesion: 0.22
Nodes (8): Artifact evidence, Boundaries, Corrected release sequence, Pre-1.0 Release Policy, Purpose, Scope, Testing strategy, VerCOR Pre-1.0 Versioning Correction Design

### Community 157 - ".request"
Cohesion: 0.22
Nodes (5): FrameType, Raise ``RuntimeInterrupted`` when a terminal signal is pending., Promote pending wakeup-fd bytes into a runtime interruption request., Record that ``signum`` requested runtime cancellation., Signal-handler entrypoint that records a cancellation request.

### Community 158 - ".__init__"
Cohesion: 0.22
Nodes (7): Predictions, CoordinateSystem, ForcingData, TerrainData, timedelta, Return the model step function, optionally JIT compiled., Build JAXGCM model resources and the VerCOR grid.

### Community 159 - "Veros Output Universe Bug-Fix Design"
Cohesion: 0.25
Nodes (7): Unified Output Coordinator, Design, Intended Behavior, Non-Goals, Problem, Testing, Veros Output Universe Bug-Fix Design

### Community 160 - "2026-05-06"
Cohesion: 0.29
Nodes (7): 2026-05-06, Coupler Lifecycle Logging, Dynamic Settings Attribute Refactor, Unified Metadata-Backed Settings Container, Validation (Coupler Lifecycle Logging, 2026-05-06), Validation (Dynamic Settings Attribute Refactor, 2026-05-06), Validation (Unified Metadata-Backed Settings Container, 2026-05-06)

### Community 161 - "VerCOR Progress Archive: 2026-05-16 to 2026-07-14"
Cohesion: 0.29
Nodes (6): Current Status, Follow-Up Candidates, Known Failed Approaches / Corrections, Next Session Checklist, Validation Policy, VerCOR Progress Archive: 2026-05-16 to 2026-07-14

### Community 162 - "File Map"
Cohesion: 0.29
Nodes (6): Calendar Year Ownership Implementation Plan, File Map, Global Constraints, Task 1: Establish the calendar-owned year API, Task 2: Remove runtime ownership and derive time metadata from each timestamp, Task 3: Independent review and final verification

### Community 163 - "Controlled Pytest Parallelization Implementation Plan"
Cohesion: 0.29
Nodes (6): Controlled Pytest Parallelization Implementation Plan, File structure, Global Constraints, Task 1: Isolate VerCOR-owned test caches per worker, Task 2: Benchmark worker counts and configure only a proven winner, Task 3: Prove coverage, determinism, quality, and final performance

### Community 164 - "Test-suite Artifact Reuse Implementation Plan"
Cohesion: 0.29
Nodes (6): Execution outcome, Historical file scope and restored structure, Original constraints, Task 1: Share one immutable artifact bundle across serial test modules, Task 2: Verify the complete gate and record measured evidence, Test-suite Artifact Reuse Implementation Plan

### Community 165 - "VerCOR Pre-1.0 Versioning Correction Implementation Record"
Cohesion: 0.33
Nodes (5): Completed migration unit, Constraints, Goal, VerCOR Pre-1.0 Versioning Correction Implementation Record, Verification contract

### Community 166 - "Period-Average Window-Start Timestamp Implementation Plan"
Cohesion: 0.40
Nodes (4): File Map, Global Constraints, Period-Average Window-Start Timestamp Implementation Plan, Task 1: Correct period identity at the output coordinator

### Community 167 - "Global Constraints"
Cohesion: 0.50
Nodes (3): Global Constraints, Task 1: Exclude Unrepresentable Veros Variables, Veros Duplicate-Dimension Output Fix Implementation Plan

### Community 168 - "Global Constraints"
Cohesion: 0.50
Nodes (3): Global Constraints, Task 1: Align Veros provider enumeration with extraction, Veros Output Universe Fix Implementation Plan

### Community 169 - "Considered approaches"
Cohesion: 0.50
Nodes (4): 1. Controlled pytest-xdist execution (selected), 2. Shared JAX-compiled test fixtures, 3. Cached source/AST indexes, Considered approaches

### Community 170 - "Considered approaches"
Cohesion: 0.50
Nodes (4): 1. Reuse immutable built artifacts across test modules (rejected experiment), 2. Batch fresh-process import probes, 3. Share JAX-compiled callables or reduce numerical inputs, Considered approaches

## Knowledge Gaps
- **638 isolated node(s):** `vercor`, `vercor-public-plugin`, `Filesystem`, `What is this?`, `Quick reference` (+633 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RectilinearGrid` connect `Make Example Grid` to `Representative Near Surface State Over`, `Single Private Normalization Bridge For`, `Make Grid`, `Custom Component Wrapping`, `Validate Cadence And Freeze A`, `Regrid Vector`, `With Fields`, `Configuration For The Bundled Camulator`, `Field Names`, `Compute Sigma Pressure Levels`, `Has Identical Grids`, `Profile Runtime`, `normalize_component_step_callable`, `Create A Complete Immutable Coupling`, `Return Setup Owned Field And`, `Api Boundaries`, `Field Transfer`, `Grid Geometry`, `Run A Coupler Through The`, `.__init__`, `Run Camulator With Veros`, `Fields Components`, `Apply Conservative Scalar Regridding`, `Run Jcm With Era5Data`, `Capture Logger Output`, `Init Context`, `Regrid Vector`, `Base Class For Exceptions Related`, `Apply Scalar`, `Iter Fields`, `Stable Public Output Contracts For`, `Run Optional Author Validation With`, `Public Payload Passed To Component`, `Canonical Data Layout Description`, `Atmosphere Components`, `Apply Scalar`, `Set Up And Return The`, `Initialize Camulator Forcing Cursor`, `Validate The Exact Structural Component`, `Initialize Model State Helpers And`, `Smoke Components`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `assert_allclose_compact()` connect `Assertions Components` to `Representative Near Surface State Over`, `Make Grid`, `Cache Coords`, `Regrid Vector`, `Canonicalize Time Last Level Field`, `Workflow Runtime Execution Tests`, `Patch The Configured Route With`, `Configuration For The Bundled Camulator`, `Field Transfer`, `Model Year Seconds`, `Run A Coupler Through The`, `Regrid Vector`, `Fields Components`, `Apply Conservative Scalar Regridding`, `Return The Immutable Runtime Policy`, `Assert Allclose Compact`, `Run Jcm With Era5Data`, `Capture Logger Output`, `Init Context`, `Array To Host`, `Convert Numpy Jax Arrays To`, `Shared H5Netcdf Output Writer For`, `Base Class For Exceptions Related`, `Apply Scalar`, `Class Body Source`, `Atmosphere Components`, `V0 4 Physics`, `External Tools Coverage`, `Make Differentiable Model`, `Initialize Model State Helpers And`, `Return A 4D Anisotropic Gaussian`, `Smoke Components`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `CouplerError` connect `Ver Cor 0 4 Route` to `Single Private Normalization Bridge For`, `Make Grid`, `Cache Coords`, `Custom Component Wrapping`, `Run Every Step Plan In`, `Regrid Vector`, `Workflow Runtime Execution Tests`, `Patch The Configured Route With`, `Field Names`, `Has Identical Grids`, `Api Boundaries`, `Ver Cor 0 4 Unified`, `Focused Tests For The Sole`, `Run A Coupler Through The`, `Regrid Vector`, `Run Camulator With Veros`, `Fields Components`, `Return The Immutable Runtime Policy`, `V0 4 Public Api`, `Capture Logger Output`, `Base Class For Exceptions Inside`, `Assets Components`, `Reject Any Enabled I O`, `Base Class For Exceptions Related`, `Apply Scalar`, `Run Optional Author Validation With`, `Set Up And Return The`, `Dummy Grid Component`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `Coupler` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Coupler` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `Clock` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Clock` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Exchange` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Exchange` has 41 INFERRED edges - model-reasoned connections that need verification._
- **What connects `vercor`, `vercor-public-plugin`, `Filesystem` to the rest of the system?**
  _638 weakly-connected nodes found - possible documentation gaps or missing edges._