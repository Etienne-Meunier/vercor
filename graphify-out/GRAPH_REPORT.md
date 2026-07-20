# Graph Report - VerCOR  (2026-07-20)

## Corpus Check
- 271 files · ~234,728 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4281 nodes · 12501 edges · 166 communities (148 shown, 18 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 1052 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `377df187`
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
- .uniform
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
- .step
- normalize_component_step_callable
- Milestone Timeline
- Test-suite performance optimization design
- _RecordingLogger
- 2026-04-30
- 2026-05-05
- VerCOR Pre-Stable Versioning Correction Design
- .__init__
- VerCOR Progress Archive: 2026-05-16 to 2026-07-14
- File Map
- Controlled Pytest Parallelization Implementation Plan
- Test-suite Artifact Reuse Implementation Plan
- VerCOR Pre-1.0 Versioning Correction Implementation Record
- Global Constraints
- Global Constraints
- Considered approaches
- Considered approaches
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
- `test_public_coupler_annotations_resolve_without_private_types()` --indirect_call--> `Coupler`  [INFERRED]
  tests/test_v0_4_public_api.py → vercor/coupler.py
- `ToyHostModel` --uses--> `Clock`  [INFERRED]
  examples/custom_component_wrapping.py → vercor/clock.py
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

## Communities (166 total, 18 thin omitted)

### Community 0 - "Representative Near Surface State Over"
Cohesion: 0.11
Nodes (28): _ocean_state(), ndarray, Representative near-surface state over ocean for bulk-flux tests., test_cold_air_outbreak_mod_strengthens_flux_magnitudes(), test_compute_ocean_surface_fluxes_matches_reference_state(), test_compute_ocean_surface_fluxes_produces_finite_and_physically_consistent_signs(), test_compute_ocean_surface_fluxes_respects_mask_for_surface_exchange_outputs(), test_flux_kernels_support_jit_and_gradients() (+20 more)

### Community 1 - "Single Private Normalization Bridge For"
Cohesion: 0.08
Nodes (31): make_host_model(), Wrap a Python host-side model while keeping VerCOR runtime fields explicit., Update the custom flux on the host runtime path., FieldNames, Single private normalization bridge for protocol-first components., Callable convenience adapter for the structural component protocol., Private callable-signature normalization for component adapters., Minimal setup context passed to component initialization hooks. (+23 more)

### Community 2 - "Make Grid"
Cohesion: 0.07
Nodes (61): make_test_grid(), NDArray, _coupler(), _prepared_prefill_binding(), Any, Focused coverage for the 0.4 component adapters and private runtime bridge., test_callable_component_adapts_supported_step_arities(), test_callable_component_preparation_retains_only_normalized_step() (+53 more)

### Community 3 - "Cache Coords"
Cohesion: 0.09
Nodes (48): SimpleNamespace, _accumulate_frames(), _ConstructedVerosState, _FakeDynamicsPrediction, _FakeForcing, _FakePhysicsModule, _FakeVariableStore, _make_coupler() (+40 more)

### Community 4 - "Custom Component Wrapping"
Cohesion: 0.05
Nodes (51): test_breaking_api_cleanup_removes_transitional_public_surfaces(), test_component_constructor_hides_raw_setup_internals(), test_coupler_accepts_plain_component_name_sequences(), test_coupler_facade_wraps_runtime_state_and_views(), test_coupler_public_methods_return_stable_state_and_views(), test_coupler_rejects_string_run_order(), test_data_and_callable_factories_return_core_contract_instances(), test_public_api_uses_canonical_breaking_names() (+43 more)

### Community 5 - "Run Every Step Plan In"
Cohesion: 0.03
Nodes (93): Run every step plan in one core-defined chunk., _JaxChunkExecutor, Advance every plan in one core-defined chunk., Build a complete static plan through public workflow contracts., Return the configured entries as an execution plan., Record the chunk and dispatch all of its plans in order., has_period_output(), Return whether the explicit target enables any declared period output. (+85 more)

### Community 6 - "With Fields"
Cohesion: 0.05
Nodes (59): Any, VerCOR 0.4 route identity, topology, regridding, and state contracts., _rebuild_state(), _replace_component_store(), _replace_source_grid(), _state_coupler(), _state_with_duplicate_component_name(), _state_with_nonfinite_mask() (+51 more)

### Community 7 - "Regrid Vector"
Cohesion: 0.08
Nodes (48): StringIO, capture_logger_output(), RuntimeArray, RecordingRegridder, Build runtime state from a Coupler's components for focused tests., runtime_state_from_coupler_components(), _dispatch_runtime_fields(), _HostRunComponent (+40 more)

### Community 8 - "Canonicalize Time Last Level Field"
Cohesion: 0.07
Nodes (59): test_era5_land_layout_uses_shared_jax_helpers(), test_erainterim_helpers_prepare_jax_backed_grid_and_masked_fields(), test_jcm_land_layout_uses_shared_jax_helpers(), test_ocean_mask_helpers_accept_jax_arrays(), test_shared_field_helpers_canonicalize_surface_fields_and_positive_masks(), test_shared_masked_surface_field_helper_supports_jit_and_gradients(), test_unconfigured_real_conversion_preserves_existing_array_dtype(), Path (+51 more)

### Community 9 - "Interrupts Components"
Cohesion: 0.06
Nodes (46): FrameType, JaxRuntimeError, KeyboardInterrupt, NoReturn, _block_until_ready(), _InterruptingHostComponent, _make_pure_coupler(), _NoopRuntimeComponent (+38 more)

### Community 10 - "Workflow Runtime Execution Tests"
Cohesion: 0.10
Nodes (42): Any, MonkeyPatch, Workflow runtime execution tests., test_alternating_jax_workflow_uses_absolute_steps_and_builds_metadata_once(), test_auto_and_forced_host_backend_selection_preserve_behavior(), test_backend_must_execute_every_plan_in_its_chunk(), test_custom_backend_interruption_is_handled_by_core(), test_custom_backend_must_return_run_state_for_each_chunk() (+34 more)

### Community 11 - "Run Data Driver"
Cohesion: 0.10
Nodes (29): build_coupler(), _default_clock(), main(), _parse_args(), Run the bundled JCM atmosphere/land setup with ERA5 ocean forcing., Parse optional short-run and initialization-only CLI controls., Run the example or return its prepared initial state., Return the historic noleap clock, optionally with a shorter run. (+21 more)

### Community 12 - "Patch The Configured Route With"
Cohesion: 0.06
Nodes (46): Patch the configured route with an explicit target-shaped mask., Any, ndarray, _RecordingTopologyPolicy, test_topology_policy_patch_accepts_valid_binary_and_fractional_masks(), test_topology_policy_patch_rejects_invalid_mask_values(), test_topology_policy_patch_rejects_unknown_keys_and_wrong_shapes(), _BoundMethodStepModel (+38 more)

### Community 13 - "Configuration For The Bundled Camulator"
Cohesion: 0.07
Nodes (43): _camulator_output_conf(), _camulator_prediction(), _make_coupler(), Any, datetime, MonkeyPatch, Path, Tensor (+35 more)

### Community 14 - "Return The Configured Runtime Logger"
Cohesion: 0.18
Nodes (12): _is_dynamic_callback_value(), JaxCallbackLogger, _partition_dynamic(), _partition_dynamic_kwargs(), Any, Small logger wrapper that emits messages through ``jax.debug.callback``., Return the wrapped Python logger name., Return whether ``level`` is enabled on the wrapped logger. (+4 more)

### Community 15 - "Field Names"
Cohesion: 0.05
Nodes (55): _NameItem, test_flatten_fields_and_append_unique(), _ComponentBinding, Immutable runtime binding produced once after setup and dtype selection., Return prepared field names in stable insertion order., Any, Raise a clear error when a component skipped base initialization., validate_component_setup() (+47 more)

### Community 16 - "Compute Sigma Pressure Levels"
Cohesion: 0.11
Nodes (30): Any, Normalize floating JAXGCM leaves without changing integer semantics., Cast every JAXGCM PyTree leaf to the configured VerCOR real dtype., Average every JAXGCM PyTree leaf along one or more axes., Flatten the leading dimensions of each JAXGCM PyTree leaf., Stack matching JAXGCM PyTree leaves from a sequence of objects., tree_as_real_dtype(), tree_as_runtime_dtype() (+22 more)

### Community 17 - "Has Identical Grids"
Cohesion: 0.05
Nodes (52): Public-only VerCOR extension fixture., PluginAssembly, PluginConfig, PluginFactory, PluginRegridder, PluginRegridderFactory, PluginWorkflow, Any (+44 more)

### Community 18 - "Profile Runtime"
Cohesion: 0.21
Nodes (13): ArgumentParser, _block_until_ready(), build_parser(), _format_result(), main(), profile_runtime(), Any, Run a small timing profile for the scanned runtime. (+5 more)

### Community 19 - "Regrid Vector"
Cohesion: 0.13
Nodes (23): ScalarPhysicsValue, test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions(), test_compute_hybrid_pressure_levels_matches_hybrid_definition(), test_get_altitudes_hybrid_sigma_levels_handles_zero_top_half_level(), test_get_altitudes_hybrid_sigma_levels_returns_finite_increasing_profile(), test_qsat_august_eqn_behaves_physically_with_t_and_p(), test_qsat_is_positive_and_increases_with_temperature(), compute_hybrid_pressure_levels() (+15 more)

### Community 20 - "Create A Complete Immutable Coupling"
Cohesion: 0.23
Nodes (19): test_cdn_and_stability_functions_are_well_behaved(), test_density_and_potential_temperature_match_closed_form(), test_flux_utility_kernels_support_jit(), Public Earth-system flux and vertical-coordinate utilities., cdn(), compute_air_density(), compute_potential_temperature(), psimhu() (+11 more)

### Community 21 - "Dtypes Components"
Cohesion: 0.06
Nodes (50): ShapeLike, test_remapper_accepts_jax_backed_constructor_inputs(), test_dtype_policy_disable_x64_maps_real_arrays_to_float32(), test_dtype_policy_enable_x64_maps_real_arrays_to_float64(), test_index_dtype_is_int32_for_both_real_precision_modes(), test_numpy_and_jax_helpers_agree_on_dtype_policy(), Private component field normalization and declaration helpers., as_jax_index_array() (+42 more)

### Community 22 - "Jax Gcm Output"
Cohesion: 0.09
Nodes (43): _additional_coordinate_values(), _default_physics_module(), _float0_leaf_to_nan(), _float0s_to_nans(), _infer_shape_to_dims(), _iter_data_items(), jax_gcm_coordinate_variables(), jax_gcm_data_variables_with_unit_metadata() (+35 more)

### Community 23 - "Return Setup Owned Field And"
Cohesion: 0.06
Nodes (44): SpinupResult, test_align_model_timestep_rejects_non_divisible_model_step(), test_align_model_timestep_returns_coupling_timestep_and_substeps(), test_run_logged_spinup_logs_each_step_and_returns_callback_result(), CamulatorRuntimeCursor, load_camulator_forcing_context(), parse_datetime_from_config(), Any (+36 more)

### Community 24 - "Api Boundaries"
Cohesion: 0.07
Nodes (37): Path, test_bilinear_interpolator_removes_unused_cartesian_helper(), test_callable_component_has_one_step_normalization_owner(), test_camulator_adapters_share_runtime_cursor_state_transition_helper(), test_camulator_gcm_factory_passes_runtime_step_directly(), test_common_exchange_recipes_are_centralized_for_examples(), test_component_base_internals_are_private_modules(), test_component_contract_modules_share_field_name_deduplication_owner() (+29 more)

### Community 25 - "Field Transfer"
Cohesion: 0.06
Nodes (46): test_runtime_state_is_separate_from_public_component_objects(), test_field_store_replacements_reject_shape_changes(), test_registered_pytree_classes_inherit_shared_flatten_methods(), _component(), Array, _RuntimeSendComponent, test_runtime_component_and_coupler_state_are_pytrees(), test_runtime_component_state_preserves_optional_payload_under_jit() (+38 more)

### Community 26 - "Ver Cor 0 4 Unified"
Cohesion: 0.19
Nodes (43): _api(), _component(), _coupler(), _period_spec(), Any, MonkeyPatch, ndarray, Path (+35 more)

### Community 27 - "Focused Tests For The Sole"
Cohesion: 0.07
Nodes (39): _ClockStep, Array, _sample_sum_and_counts(), build_output_plan(), _coordinate_dtypes(), _coordinate_parts(), _coordinate_shapes(), initial_output_session() (+31 more)

### Community 28 - "Grid Geometry"
Cohesion: 0.10
Nodes (13): Any, RuntimeArray, Flatten fields and payload while preserving mapping order., Restore a setup result from JAX PyTree leaves., Any, RuntimeArray, Return declared field updates for one component step., Resolve the public protocol annotation once declarations finish loading. (+5 more)

### Community 29 - "Model Year Seconds"
Cohesion: 0.10
Nodes (39): StrEnum, DailyForcingIndexCase, datetime, MonkeyPatch, Path, test_calendar_owns_canonical_year_types_and_durations(), test_calendar_resolves_year_type_from_existing_clock_policy(), test_calendar_year_helpers_reject_foreign_policy_values() (+31 more)

### Community 30 - "Run A Coupler Through The"
Cohesion: 0.07
Nodes (81): build_slab_coupler(), Build and initialize a small pure-JAX slab coupler for profiling., create_runtime_state_from_coupler(), prepared_coupling(), Any, RuntimeArray, Create, prime, and validate state using the coupler's installed topology., Return the Coupler's canonical prepared runtime boundary. (+73 more)

### Community 31 - "Regrid Vector"
Cohesion: 0.14
Nodes (32): _block_until_ready(), _component_state(), _identity_factory(), _IdentityRegridder, _make_initial_state(), _make_output_component(), _make_period_output_coupler(), _period_target() (+24 more)

### Community 32 - "Create Runtime State From Coupler"
Cohesion: 0.02
Nodes (132): 2026-04-23, Compile Cache and Safe Donation Runtime Audit, Coupler / Veros / Clock Coverage Expansion, Coverage Outcome, Eighth JAX Translation Slice 8A: CAMulator Boundary, Eighth JAX Translation Slice 8B: JAX-First Example Drivers, Eleventh JAX Translation Slice 11A: ERA5 Land Adapter, Eleventh JAX Translation Slice 11B: JAX-Backed Forcing Read Boundary (+124 more)

### Community 33 - "Run Camulator With Veros"
Cohesion: 0.11
Nodes (23): _default_jcm_parameter_values(), Parameters, Return selected default JCM parameter values for the example script., test_scalar_and_vector_regridder_capabilities_are_independent(), Canonical public constructor-only coupler assembly and run facade., OutputTarget, Path, Enable selected run-level outputs beneath one directory.      Passing no target (+15 more)

### Community 34 - "Opaque Mutable Leaf Used To"
Cohesion: 0.12
Nodes (35): _api(), _MutablePayload, _one_step_coupler(), Any, Opaque mutable leaf used to prove host payload ownership., test_callable_component_uses_declared_outputs_and_setup_payload(), test_component_is_the_structural_protocol(), test_component_mapping_arguments_reject_non_mappings() (+27 more)

### Community 35 - "Build Distributions"
Cohesion: 0.10
Nodes (28): TempPathFactory, build_distributions(), BuiltDistributions, _cached_build_pythonpath(), _existing_distributions(), install_local_target(), Path, Offline distribution build/install helpers for artifact-boundary tests. (+20 more)

### Community 36 - "Select Fast Cases"
Cohesion: 0.13
Nodes (33): Protocol, select_fast_cases(), SelectFastCases, ArithmeticCase, ClockIterationCase, StringCase, test_clock_360_rejects_day_31_start(), test_clock_360_rolls_microseconds_across_year_boundary() (+25 more)

### Community 37 - "Public Post Step Component View"
Cohesion: 0.13
Nodes (34): OutputContext, Public post-step component view supplied to an output provider.      ``step`` is, camulator_average_coordinate_variables(), camulator_average_data_variables(), camulator_output_provider(), camulator_period_output_variables(), _CAMulatorOutputProvider, _configured_level_values() (+26 more)

### Community 38 - "Canonicalize External Typing Aliases"
Cohesion: 0.08
Nodes (34): canonicalize_external_typing_aliases(), Stable rendering support for source and installed public-signature tests., Replace evidenced dependency-sensitive aliases with public tokens., _canonical_public_callable_names(), _canonical_public_method_names(), _documented_public_manifest(), _normalized_signature(), _public_signature_contract() (+26 more)

### Community 39 - "Fields Components"
Cohesion: 0.12
Nodes (28): test_boundary_redesign_removes_remaining_duplicate_public_helpers(), test_runtime_private_state_uses_public_domain_vocabulary(), test_state_constructors_do_not_expose_runtime_stores(), test_total_surface_temperature_diagnostic_uses_runtime_view_fields(), test_component_vector_speed_reads_runtime_component_view(), test_component_vector_speed_uses_jax_arrays(), test_runtime_component_view_reads_fields_without_store_internals(), test_safe_component_nanmean_returns_nan_for_missing_fields() (+20 more)

### Community 40 - "Apply Conservative Scalar Regridding"
Cohesion: 0.16
Nodes (26): _grid(), Any, MonkeyPatch, test_conservative_factory_forwards_remapper_options(), test_conservative_factory_returns_conservative_rectilinear_regridder(), test_conservative_regridder_api_does_not_expose_noop_fill_value(), test_identical_grid_regridder_remains_scalar_only(), test_public_conservative_factory_exposes_radius_km_only() (+18 more)

### Community 41 - "Return The Immutable Runtime Policy"
Cohesion: 0.26
Nodes (18): _fake_jcm_step(), _FakeDynamicsPrediction, _FakeJCMForcing, _FakePhysicsPrediction, _FakePrediction, _FakeShortwaveRad, _FakeSurfaceFlux, _JAXGCMFixture (+10 more)

### Community 42 - "Veros Output"
Cohesion: 0.12
Nodes (34): Public payload passed to component snapshot writers.      ``time`` is the model, SnapshotContext, _active_output_variable_names(), _attrs_for_variable(), _coordinate_dimension_is_extractable(), _current_timestep_index(), _drop_timestep_dim(), _extract_coordinate_variable() (+26 more)

### Community 43 - "Assert Allclose Compact"
Cohesion: 0.11
Nodes (34): float64, assert_allclose_compact(), _format_index(), Any, Assert numerical closeness with concise, greppable diagnostics., _cell_areas(), _make_remapper(), NDArray (+26 more)

### Community 44 - "Assertions Components"
Cohesion: 0.15
Nodes (27): assert_array_equal_compact(), Assert exact equality with the same compact diagnostics., Any, _scalar_interp(), test_descending_latitude_is_constructor_only_and_interpolates_correctly(), test_init_accepts_jax_arrays_and_tracks_source_orientation(), test_init_rejects_non_monotonic_latitude(), test_init_rejects_non_monotonic_longitude() (+19 more)

### Community 45 - "V0 4 Public Api"
Cohesion: 0.10
Nodes (21): _clock(), _component(), Any, Exception, Path, test_caller_cannot_mutate_the_clock_after_coupler_construction(), test_constructor_normalizes_log_level_before_configuring_custom_logger(), test_constructor_rejects_duplicate_component_names() (+13 more)

### Community 46 - "Struct Time"
Cohesion: 0.09
Nodes (9): struct_time, _ModelDateTimeBase, month_day_from_day_of_year(), Self, timedelta, Return a ``datetime.timetuple()``-compatible model-calendar value., Format the datetime using ``datetime.strftime``-style directives., Return ``(month, day)`` for one-based day-of-year. (+1 more)

### Community 47 - "Run Jcm With Era5Data"
Cohesion: 0.07
Nodes (38): JCMLandAtmosphereConfig, Configuration for the bundled paired JCM land/atmosphere setup., Validate paired setup component names., make_jax_gcm(), _CoordinateSystem, _TerrainData, Return a differentiable JAXGCM/JCM atmosphere component., load_jcm_coords_terrain_forcing() (+30 more)

### Community 48 - "Get Loc"
Cohesion: 0.11
Nodes (20): MonkeyPatch, Path, slice, _RecordingLogger, test_camulator_runtime_cursor_initializes_indexes_and_advances(), test_grid_field_defaults_returns_defaults_with_overrides(), test_initialize_camulator_forcing_cursor_accepts_integer_index(), test_initialize_camulator_forcing_cursor_returns_index_and_warns_on_mismatch() (+12 more)

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
Cohesion: 0.06
Nodes (46): test_exchange_accepts_supported_names_only(), test_fields_facade_owns_vector_field_contract(), _factory(), Any, _ScalingRegridder, test_dispatch_component_exchanges_handles_scalar_masks_and_gradients(), test_dispatch_component_exchanges_preserves_vector_regridding_behavior(), test_runtime_dispatch_context_groups_exchanges_by_destination() (+38 more)

### Community 54 - "Base Class For Exceptions Inside"
Cohesion: 0.12
Nodes (34): test_surface_role_lookup_checks_mapping_key_and_component_name(), CouplerError, Base class for exceptions inside Coupler., _array_is_traced(), _array_leaf_metadata(), Any, _raise_if_false(), Validate exact runtime store names, grid shapes, and dtypes. (+26 more)

### Community 55 - "Conftest Components"
Cohesion: 0.18
Nodes (12): CaseT, Config, FixtureRequest, Item, Parser, fast_mode(), pytest_addoption(), pytest_collection_modifyitems() (+4 more)

### Community 56 - "Init Context"
Cohesion: 0.13
Nodes (32): CoverageCouplerStub, datetime, test_camulator_land_declares_radiation_exchange_inputs(), _fake_jcm_land_inputs(), _install_data_driver_factory_fakes(), _prepare_component_for_test(), Any, datetime (+24 more)

### Community 57 - "Regrid Vector"
Cohesion: 0.13
Nodes (25): _make_grid(), Any, MonkeyPatch, test_bilinear_factory_forwards_interpolator_options(), test_bilinear_factory_returns_bilinear_rectilinear_regridder(), test_regridder_constructor_propagates_interpolator_options(), test_regridder_constructor_sets_interpolator_and_grids(), test_regridder_has_identical_grids_false_for_different_coords() (+17 more)

### Community 58 - "Array To Host"
Cohesion: 0.23
Nodes (14): MonkeyPatch, test_runtime_array_to_host_is_canonical_host_transfer(), test_transposed_host_array_uses_canonical_host_transfer(), array_to_host(), host_int64_array(), Any, NDArray, RuntimeArray (+6 more)

### Community 59 - "Advance The Private Host Backed"
Cohesion: 0.14
Nodes (23): Any, Veros host-runtime stepping helpers., Advance the private host-backed Veros ocean boundary., step_veros_runtime(), advance_veros_substeps(), apply_veros_forcing_fields(), copy_state(), extract_surface_temperature() (+15 more)

### Community 60 - "Assets Components"
Cohesion: 0.22
Nodes (23): MonkeyPatch, Path, test_asset_base_url_normalizes_and_handles_empty(), test_download_asset_writes_response_bytes(), test_ensure_registered_asset_downloads_when_cached_md5_invalid(), test_ensure_registered_asset_errors_without_base_url(), test_ensure_registered_asset_raises_and_deletes_on_md5_mismatch(), test_ensure_registered_asset_uses_valid_cached_file() (+15 more)

### Community 61 - "Mark Variables That Are Recognized"
Cohesion: 0.06
Nodes (39): load_credit_modules(), load_postblock_modules(), Lazy optional-dependency loading for CAMulator setup adapters., Load CREDIT core modules at the CAMulator execution boundary., Load optional CREDIT postblock fixers without import-time warnings., initialize_camulator(), Any, CAMulator model, transform, forcing, and state initialization. (+31 more)

### Community 62 - "Convert Numpy Jax Arrays To"
Cohesion: 0.16
Nodes (23): _float_dtype_of(), dtype, ndarray, Convert numpy/jax arrays to a numpy.ndarray (without requiring jax)., With same T and pressure thickness, adding humidity increases Tv and thus thickn, Specific humidity q should not be treated as water-vapor mixing ratio., Return a floating dtype suitable for constructing reference arrays., Consistency check in float32:       ln(p[k-1]/p[k]) ≈ g*Δz / (Rd*Tv_bar) (+15 more)

### Community 63 - "Extrapolate Scalar Field"
Cohesion: 0.23
Nodes (11): all_negative(), all_positive(), great_circle_distance_rad(), Array, r"""Compute local east/north unit vectors on the unit sphere., r"""Return haversine great-circle distance in radians on the unit sphere., Return whether all entries are strictly positive as an eager bool., Return whether all entries are strictly negative as an eager bool. (+3 more)

### Community 64 - "Shared H5Netcdf Output Writer For"
Cohesion: 0.21
Nodes (20): File, Path, test_write_netcdf_dataset_logs_filename_when_logger_is_supplied(), test_write_netcdf_dataset_rejects_conflicting_dimension_sizes(), test_write_netcdf_dataset_writes_scalar_data_variables(), _coordinate_variables(), _mean_variables(), Path (+12 more)

### Community 65 - "Reject Any Enabled I O"
Cohesion: 0.16
Nodes (16): Return repository source text for architecture-boundary assertions., source_for(), test_callable_wrapper_module_does_not_need_request_dataclass(), test_component_runtime_helpers_do_not_keep_annotation_only_protocol_layer(), test_components_package_has_no_top_level_import_cycles(), test_host_runtime_selection_uses_public_component_spec_execution(), test_lifecycle_storage_uses_component_spec_as_single_owner(), test_public_lifecycle_hook_types_are_owned_by_component_contracts() (+8 more)

### Community 66 - "Base Class For Exceptions Related"
Cohesion: 0.17
Nodes (5): BaseException, _FakeSettings, _NullContext, Any, _RecordingLogger

### Community 67 - "Day Of Year"
Cohesion: 0.14
Nodes (18): CalendarDate, day_of_year_from_month_day(), is_leap_year(), Protocol, Return whether ``year`` is a Gregorian leap year., Return one-based day-of-year for a month/day pair., daily_forcing_day_of_year(), daily_forcing_index() (+10 more)

### Community 68 - "Shared Net Cdf Writer Boundary"
Cohesion: 0.04
Nodes (40): [0.4.0a1] - 2026-07-14, Added, Changed, Changelog, Compatibility, Removed, 1. Executive summary, 2. Duplication map (+32 more)

### Community 69 - "Initialize Camulator Runtime Fields"
Cohesion: 0.08
Nodes (33): test_camulator_runtime_field_names_have_lightweight_contract_owner(), camulator_runtime_field_defaults(), Lightweight CAMulator runtime field contract ownership., Return scalar defaults for all CAMulator runtime exchange fields., _camulator_output_array(), initialize_camulator_runtime_fields(), map_camulator_prediction_arrays(), map_camulator_prediction_to_runtime_fields() (+25 more)

### Community 70 - "Apply Scalar"
Cohesion: 0.16
Nodes (10): Any, Array, RuntimeArray, Ensure latitude bounds are monotonically increasing., Return dense destination-by-source overlap lengths., Flatten a dense overlap matrix into sparse triplets., 1D overlap calculation (latitude in sin-space)., 1D longitude overlap with periodicity check. (+2 more)

### Community 71 - "Build Output Plan"
Cohesion: 0.06
Nodes (30): 1. Code Organization, 1. Tests are everything, 2. Code Style, 2. Concise test output (context window hygiene), 3. Fast tests to avoid time blindness, 3. Testing, 4. Keep PROGRESS.md current (agent orientation), 5. Prevent regressions (CI discipline) (+22 more)

### Community 72 - "Iter Fields"
Cohesion: 0.15
Nodes (13): FieldLookupScope, FieldScope, test_prepared_coupling_owns_single_normalized_runtime_boundary(), test_runtime_topology_state_groups_read_only_maps(), _field(), _field_candidates(), _RuntimeArray, Return the fractional mask for an exchange. (+5 more)

### Community 73 - "Stable Public Output Contracts For"
Cohesion: 0.14
Nodes (16): _canonical_metadata_value(), _frozen_mapping(), _frozen_metadata(), OutputProvider, Any, Protocol, Stable public output contracts for components and run-level I/O., Return array rank without importing a host-array implementation. (+8 more)

### Community 74 - "Run Optional Author Validation With"
Cohesion: 0.07
Nodes (44): Update fields and replace the public runtime payload., test_step_result_payload_sentinel_preserves_runtime_payload_by_default(), test_prefill_rejects_fields_absent_from_exchange_contract(), test_prefill_rejects_non_grid_runtime_store_shapes(), test_prefill_rejects_nonnumeric_store_values_with_component_error(), _ComponentDeclaration, _copy_owned_pytree(), _normalize_prefill_contract_store() (+36 more)

### Community 75 - "Public Payload Passed To Component"
Cohesion: 0.15
Nodes (18): test_output_mask_names_remain_unique_after_route_token_sanitizing(), _component_output_filenames(), output_masks_for_component(), datetime, ModelDateTime, Path, RuntimeArray, Write final runtime component views for all configured components. (+10 more)

### Community 76 - "Make Example Grid"
Cohesion: 0.05
Nodes (49): make_custom_coupler(), make_data_forcing(), make_differentiable_model(), make_example_grid(), Any, Small structural component using the public Component protocol., Minimal custom backend that delegates component stepping to RuntimeDriver., Assemble custom-named components without the built-in surface-mask policy. (+41 more)

### Community 77 - "Class Body Source"
Cohesion: 0.14
Nodes (12): class_body_source(), package_import_cycles(), Return the source segment for one top-level class., Return top-level import cycles within one package directory., test_external_package_has_no_top_level_import_cycles(), test_output_package_has_no_top_level_import_cycles(), test_bilinear_module_delegates_private_implementation_owners(), test_interpolators_package_has_no_top_level_import_cycles() (+4 more)

### Community 78 - "Apply Wind Filter To Tensor"
Cohesion: 0.06
Nodes (45): Module, CAMulatorStepper, Any, device, Tensor, CAMulator state transformation and model stepping helpers., Apply wind artifact filtering and conservation fixers., Core CAMulator time-stepper with optional post-processing fixers. (+37 more)

### Community 79 - "Canonical Data Layout Description"
Cohesion: 0.23
Nodes (13): test_canonical_grid_field_shape_error_is_shared(), test_canonical_grid_field_shape_normalizes_array_shape(), test_validate_canonical_grid_field_shape_raises_consistent_error(), canonical_data_layout_description(), canonical_grid_field_shape(), canonical_grid_field_shape_error(), is_canonical_grid_field_shape(), Return the accepted component data layout description. (+5 more)

### Community 80 - "Boundary Tests For Lazy Bundled"
Cohesion: 0.22
Nodes (14): MonkeyPatch, Path, Boundary tests for lazy bundled setup imports and configuration., Return the package root selected for fresh-process boundary probes., _run_missing_dependency_probe(), _run_setup_probe(), test_camulator_enabled_spinup_fails_before_runtime_configuration(), test_lazy_factory_attribute_access_loads_only_lightweight_factory_module() (+6 more)

### Community 81 - "Atmosphere Components"
Cohesion: 0.19
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
Cohesion: 0.15
Nodes (12): _FakeVariableStore, _FakeVerosState, Any, MonkeyPatch, ndarray, Path, test_compute_hybrid_pressure_levels_has_explicit_owner(), test_compute_sigma_pressure_levels_handles_valid_and_invalid_inputs() (+4 more)

### Community 88 - "Tests For The Injectable Jcm"
Cohesion: 0.21
Nodes (8): _FakeCoupler, MonkeyPatch, Tests for the injectable JCM/ERA5 example entry point., _RecordingRunCoupler, test_build_coupler_default_workflow_keeps_historic_clock(), test_build_coupler_uses_injected_ocean_inputs_and_clock(), test_cli_modes_use_requested_step_count_and_state_path(), test_example_module_import_is_safe_without_jcm()

### Community 89 - "Camulator Stepper"
Cohesion: 0.18
Nodes (16): StreamHandler, _canonical_handler(), _format_canonical_record(), Logger, test_coupler_configures_injected_python_logger_with_canonical_boundary(), test_default_logger_uses_vercor_logger_name(), test_setup_logger_formats_traced_values_under_scan(), test_setup_logger_installs_canonical_owned_handler_format() (+8 more)

### Community 90 - "Actual Averaging Window Start"
Cohesion: 0.05
Nodes (38): Actual Averaging-Window Start, File Map, Global Constraints, Period-Average Window-Start Timestamp Implementation Plan, Task 1: Correct period identity at the output coordinator, Compatibility and release, Coupling contracts, Execution and output (+30 more)

### Community 91 - "Package Import Cycles"
Cohesion: 0.24
Nodes (8): Return the configured runtime logger., emit_host_log(), _format_message(), Any, Emit a formatted log record on the host without adding a JAX callback., logger_enabled_for(), Return whether a level is enabled., Return whether ``logger`` should emit ``level`` host-side messages.

### Community 92 - "Structural Differentiable Component Implemented Outside"
Cohesion: 0.09
Nodes (22): 0) Spherical Coordinates vs. Geographical Spherical Coordinates, 10) Properties & remarks, 1) Grids, indexing, and notation, 2) Periodic longitude wrapping, 3.1) Forward (wrapped) longitudinal difference, 3.2) Latitudinal fraction, 3) Cell search and local bilinear coordinates, 4) Bilinear shape functions (weights) (+14 more)

### Community 93 - "Apply Scalar"
Cohesion: 0.28
Nodes (11): configure_test_cache_environment(), Path, Private support for deterministic pytest worker isolation., Set serial or worker-local defaults while preserving explicit values., Path, Contracts for serial and distributed test cache isolation., test_explicit_user_values_are_preserved_in_workers(), test_serial_process_defaults_all_test_cache_values() (+3 more)

### Community 94 - "Return Typed Indexing Metadata For"
Cohesion: 0.27
Nodes (10): test_plot_component_scalar_vector_comparison_accepts_callable_scalar(), test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes(), test_plot_component_scalar_vector_comparison_rejects_empty_rows(), _get_component_plot_data(), plot_component_scalar_vector_comparison(), Any, ComponentMetric, NDArray (+2 more)

### Community 95 - "Load Optional Setup Factories Only"
Cohesion: 0.20
Nodes (7): Return the wrapped Python logger level., Return the effective logging threshold., Set the wrapped Python logger threshold., normalize_log_level(), Return a standard ``logging`` integer level from a string or integer., effective_log_level(), Return the effective level for logger-like objects.

### Community 96 - "Run Every Readme Python Block"
Cohesion: 0.27
Nodes (10): _assert_public_imports_only(), MonkeyPatch, Path, _python_fences(), Run every README Python block together, outside the repository directory., Execute the supported 0.4 migration result and verify its observable state., Return Python snippets from Markdown in source order., Reject imports from underscored VerCOR modules in a documentation snippet. (+2 more)

### Community 97 - "Set Up And Return The"
Cohesion: 0.12
Nodes (24): test_create_surface_exchange_masks_rejects_missing_ocean_binary_mask(), test_create_surface_exchange_masks_rejects_non_identical_land_and_atmosphere_grids(), test_validate_land_mask_consistency_rejects_shape_and_value_mismatches(), test_topology_module_owns_public_topology_contracts(), test_surface_mask_policy_is_public_core_configuration(), build_surface_mask_topology_patch(), create_surface_exchange_masks(), Protocol (+16 more)

### Community 98 - "Initialize Camulator Forcing Cursor"
Cohesion: 0.38
Nodes (7): test_era5_atmosphere_helpers_support_jit_and_gradients(), _compute_monthly_diagnostics(), _decode_surface_pressure(), Array, ArrayLike, Convert log surface pressure to physical pressure in Pascals., Compute ERA5 diagnostics for one monthly slice on the runtime JAX path.

### Community 99 - "Camulator Init"
Cohesion: 0.33
Nodes (6): CaptureFixture, _diagnostic_component_order(), test_print_component_field_means_table_with_callable_metric(), print_component_field_means_table(), ComponentMetric, Print a means table for component fields with configurable column order.

### Community 100 - "Make Differentiable Model"
Cohesion: 0.16
Nodes (19): _frame(), Focused tests for the sole immutable output accumulator and layout helper., test_grid_field_dims_is_the_single_output_layout_rule(), test_output_accumulator_canonicalizes_array_metadata_for_jit_reuse(), test_output_accumulator_is_an_immutable_jax_pytree(), test_output_accumulator_preserves_nanmean_counts_without_mutation(), test_output_accumulator_reduces_named_sample_dimension(), test_output_accumulator_rejects_changed_variables_dimensions_and_shape() (+11 more)

### Community 101 - "Emit A Debug Message"
Cohesion: 0.22
Nodes (5): Any, Emit a debug message., Emit an informational message., Emit a warning message., Emit an error message.

### Community 102 - "Build Input With Forcing"
Cohesion: 0.33
Nodes (3): GlobalFourDegreeSetup, CustomGlobalFourDegree, Veros global 4-degree setup with VerCOR-controlled forcing fields.

### Community 103 - "Controlled Pytest Parallelization Implementation Plan"
Cohesion: 0.50
Nodes (4): Worker Cache Isolation, Rejected Artifact Reuse Experiment, Measured pytest-xdist Default, Timing-Gate Rejection

### Community 104 - "Plot Component Scalar Vector Comparison"
Cohesion: 0.67
Nodes (5): configure_python_logger(), _install_canonical_handler(), Logger, Configure ``logger`` to emit VerCOR records with the canonical format., _remove_noncanonical_handlers()

### Community 105 - "Validate The Exact Structural Component"
Cohesion: 0.25
Nodes (5): Validate the exact structural component contract immediately., validate_component_contract(), Return the component's rectilinear grid., Return the immutable component declaration., Return the unique component name.

### Community 106 - "Initial State"
Cohesion: 0.09
Nodes (40): test_component_step_rejects_non_mapping_non_step_result_returns(), test_component_binding_is_stable_when_setup_mutates_original_owner(), test_constructor_builds_coupler_from_plain_recipe(), test_custom_topology_policy_builds_once_and_patches_route_maps(), test_runtime_options_accept_custom_execution_backend(), test_runtime_options_own_core_runtime_configuration(), test_structural_component_like_runs_without_private_component_internals(), test_structural_component_validation_is_actionable() (+32 more)

### Community 107 - "Initialize Model State Helpers And"
Cohesion: 0.40
Nodes (4): _materialize_configuration(), Any, Return normalized declarations for private runtime preparation., Return one constructor collection as an owned immutable tuple.

### Community 108 - "Camulator Contracts"
Cohesion: 0.50
Nodes (4): _field_store_from_mapping(), Any, Create a public component-state view from plain field mappings., Return a runtime field store from public plain field mappings.

### Community 109 - "Bundled Native Output Paths Use"
Cohesion: 0.48
Nodes (6): Bundled native output paths use ordinary providers and core coordination., _source(), test_bundled_factories_install_native_output_providers(), test_camulator_native_period_output_uses_run_level_paths(), test_core_output_session_owns_native_output_boundaries(), test_native_output_modules_return_output_frames()

### Community 110 - "Veros Runtime Settings"
Cohesion: 0.17
Nodes (12): test_model_setup_factories_use_the_public_setup_owner(), _load_veros_implementation(), make_veros_gcm(), Any, Load Veros implementation owners after runtime configuration., Return a host-backed Veros GCM component., configure_veros_runtime(), Any (+4 more)

### Community 111 - "Camulator Imports"
Cohesion: 0.12
Nodes (17): 2026-05-15, Conservative Bilinear Helper Cleanup, Conservative Compatibility Cleanup, Maintainability Audit Follow-Up Consolidation, Private Compatibility Shim Removal, Private Runtime Helper Consolidation, Runtime FieldStore Compatibility Audit, Runtime/Setup Helper Boundary Cleanup (+9 more)

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

### Community 120 - "Production Numpy Boundaries"
Cohesion: 0.67
Nodes (3): _imports_numpy(), Path, test_numpy_imports_match_explicit_host_boundaries()

### Community 122 - "Reject Invalid Nested Callbacks At"
Cohesion: 0.50
Nodes (3): Reject invalid nested callbacks at configuration time., Validate one optional lifecycle callback immediately., _validate_callback()

### Community 123 - "Return The Registered Component Name"
Cohesion: 0.10
Nodes (20): 2026-05-01, 2026-05-04, 2026-05-06, 2026-05-13, Canonical Component Data Dimension Order, Canonical VerCOR Logging Format, Centralized VerCOR Dtype Policy, Coupler Lifecycle Logging (+12 more)

### Community 124 - "Unified Opt In Output"
Cohesion: 1.00
Nodes (3): Unified Opt-In Output, Veros Duplicate-Dimension Output Fix, Veros Output Universe Fix

### Community 125 - "Calendar Year Ownership Design"
Cohesion: 0.20
Nodes (9): Calendar API, Calendar Year Ownership Design, Error Handling and Compatibility, Goal, Runtime Data Flow, Scope, Single Calendar Owner, Testing (+1 more)

### Community 126 - ".uniform"
Cohesion: 0.16
Nodes (11): test_grid_constructors_live_on_rectilinear_grid_class(), _is_strictly_increasing(), Any, Array, _PrecisionPolicy, _RuntimeArray, Self, Return this grid with real arrays converted to ``policy`` precision. (+3 more)

### Community 131 - "Private Runtime Modules Should Import"
Cohesion: 0.14
Nodes (13): Final Acceptance, Global Constraints, Task 10: Finish architecture review, migration docs, release metadata, and CI, Task 1: Freeze the 0.3.2 baseline and record the approved specification, Task 2: Introduce typed physical constants and single precision ownership, Task 3: Replace component authoring with the protocol-first contract, Task 4: Make assembly constructor-only and close public module boundaries, Task 5: Add route IDs, regridder capabilities, route topology, and strict state (+5 more)

### Community 132 - "Path Components"
Cohesion: 0.14
Nodes (13): Baseline and acceptance metrics, Benchmark protocol, Controlled pytest parallelization design, Coverage equivalence, Default command decision, Determinism and state-leak checks, Expected retained deliverables, Objective (+5 more)

### Community 134 - "Validate Cadence And Freeze A"
Cohesion: 0.29
Nodes (9): _BoundaryCall, _clock(), _component(), Path, Regression tests for final VerCOR 0.4 public-boundary review findings., _run_state_components(), _setup_context(), test_prepared_binding_does_not_delegate_private_markers_and_uses_spec_output() (+1 more)

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

### Community 146 - ".step"
Cohesion: 0.50
Nodes (3): Any, RuntimeArray, Return no updates because data components have no active model step.

### Community 147 - "normalize_component_step_callable"
Cohesion: 0.17
Nodes (10): _AuthorStepCallable, _ComponentStepCallable, Any, RuntimeArray, Validate configuration and normalize the callable signature., Delegate one model step through the normalized callable signature., _component_step_signature_error(), normalize_component_step_callable() (+2 more)

### Community 148 - "Milestone Timeline"
Cohesion: 0.18
Nodes (11): 2026-04-27 to 2026-04-23: JAX Translation and Unified Runtime Foundation, 2026-04-30 to 2026-04-28: Runtime Package and Boundary Refactors, 2026-05-04 to 2026-05-01: Data Layout and Data Components, 2026-05-05: Runtime Interrupt Handling, 2026-05-06: Settings and Lifecycle Logging, 2026-05-07: Component Authoring API, 2026-05-08: Runtime Ownership and Component Boilerplate, 2026-05-12: Precision, Performance, and API Simplification (+3 more)

### Community 149 - "Test-suite performance optimization design"
Cohesion: 0.18
Nodes (10): Baseline, Behavioral equivalence, Failure handling and isolation, Focused timing record, Historical objective, Outcome, Planned implementation and validation sequence, Rejected experimental architecture (+2 more)

### Community 153 - "2026-04-30"
Cohesion: 0.22
Nodes (9): 2026-04-30, JAX Callback Runtime Logging, Public/Runtime API Boundary Clarification, Runtime Context Boundary Cleanup, Runtime Package Refactor, Validation (JAX Callback Runtime Logging, 2026-04-30), Validation (Public/Runtime API Boundary Clarification, 2026-04-30), Validation (Runtime Context Boundary Cleanup, 2026-04-30) (+1 more)

### Community 154 - "2026-05-05"
Cohesion: 0.22
Nodes (9): 2026-05-05, Compiled Runtime Wakeup-Fd Interrupt Handling, JAXGCM Forcing Payload Scan Shape Stability, Scanned Runtime Progress Logging, Unified Runtime Interrupt Handling, Validation (Compiled Runtime Wakeup-Fd Interrupt Handling, 2026-05-05), Validation (JAXGCM Forcing Payload Scan Shape Stability, 2026-05-05), Validation (Scanned Runtime Progress Logging, 2026-05-05) (+1 more)

### Community 156 - "VerCOR Pre-Stable Versioning Correction Design"
Cohesion: 0.25
Nodes (7): Artifact evidence, Boundaries, Corrected release sequence, Purpose, Scope, Testing strategy, VerCOR Pre-Stable Versioning Correction Design

### Community 158 - ".__init__"
Cohesion: 0.13
Nodes (13): Predictions, Return a policy matching the active JAX global precision setting., JAXGCMSetupState, CoordinateSystem, ForcingData, TerrainData, timedelta, Return the model step function, optionally JIT compiled. (+5 more)

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
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RectilinearGrid` connect `Make Example Grid` to `Single Private Normalization Bridge For`, `Make Grid`, `Cache Coords`, `Custom Component Wrapping`, `With Fields`, `Regrid Vector`, `Canonicalize Time Last Level Field`, `Configuration For The Bundled Camulator`, `Field Names`, `Has Identical Grids`, `Profile Runtime`, `normalize_component_step_callable`, `Dtypes Components`, `Return Setup Owned Field And`, `Api Boundaries`, `Field Transfer`, `Run A Coupler Through The`, `.__init__`, `Run Camulator With Veros`, `Fields Components`, `Apply Conservative Scalar Regridding`, `Return The Immutable Runtime Policy`, `Assert Allclose Compact`, `Run Jcm With Era5Data`, `Capture Logger Output`, `Init Context`, `Regrid Vector`, `Array To Host`, `Run Optional Author Validation With`, `Atmosphere Components`, `Return Typed Indexing Metadata For`, `Set Up And Return The`, `Camulator Init`, `Validate The Exact Structural Component`, `Camulator Contracts`, `Return A 4D Anisotropic Gaussian`, `.uniform`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `assert_allclose_compact()` connect `Assert Allclose Compact` to `Representative Near Surface State Over`, `Make Grid`, `Cache Coords`, `Regrid Vector`, `Canonicalize Time Last Level Field`, `Workflow Runtime Execution Tests`, `Patch The Configured Route With`, `Configuration For The Bundled Camulator`, `Field Names`, `Regrid Vector`, `Create A Complete Immutable Coupling`, `Field Transfer`, `Run A Coupler Through The`, `Regrid Vector`, `Fields Components`, `Apply Conservative Scalar Regridding`, `Return The Immutable Runtime Policy`, `Assertions Components`, `Capture Logger Output`, `Ver Cor 0 4 Route`, `Init Context`, `Array To Host`, `Convert Numpy Jax Arrays To`, `Shared H5Netcdf Output Writer For`, `Make Example Grid`, `Class Body Source`, `Atmosphere Components`, `V0 4 Physics`, `External Tools Coverage`, `Return Typed Indexing Metadata For`, `Initialize Camulator Forcing Cursor`, `Make Differentiable Model`, `Initial State`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `CouplerError` connect `Base Class For Exceptions Inside` to `Make Grid`, `Custom Component Wrapping`, `Run Every Step Plan In`, `With Fields`, `Regrid Vector`, `Workflow Runtime Execution Tests`, `Patch The Configured Route With`, `Field Names`, `Compute Sigma Pressure Levels`, `Api Boundaries`, `Ver Cor 0 4 Unified`, `Focused Tests For The Sole`, `Run A Coupler Through The`, `Regrid Vector`, `Run Camulator With Veros`, `Return The Immutable Runtime Policy`, `V0 4 Public Api`, `Capture Logger Output`, `Ver Cor 0 4 Route`, `Assets Components`, `Run Optional Author Validation With`, `Make Example Grid`, `Set Up And Return The`, `Initial State`, `Initialize Model State Helpers And`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `Coupler` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Coupler` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `Clock` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Clock` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Exchange` (e.g. with `SequentialBackend` and `StructuralFluxModel`) actually correct?**
  _`Exchange` has 41 INFERRED edges - model-reasoned connections that need verification._
- **What connects `vercor`, `vercor-public-plugin`, `Filesystem` to the rest of the system?**
  _638 weakly-connected nodes found - possible documentation gaps or missing edges._