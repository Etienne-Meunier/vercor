# 2026-04-23

## Coupler / Veros / Clock Coverage Expansion

- Extended `tests/test_coupler_coverage.py` to cover the remaining in-scope coupler control flow:
  - `initialize()` happy path with `ATM`, `OCN`, `LND`, and `ICE`
  - duplicate regridder creation warning path
  - `enable_x64_computations` override with patched `jax.config.update()`
  - `_create_exchange_masks()` failure branches for mismatched land/atmosphere grids and missing ocean masks
  - `finalize()`, `__str__`, `__repr__`, and `run()` happy path ordering
- Extended `tests/test_external_components_coverage.py` to cover more unit-testable `vercor/components/external/veros_gcm.py` helpers:
  - `compute_fluxes()` `qnec` zeroing branch for sentinel `dqfldt`
  - `CustomGlobalFourDegree.set_diagnostics()` via the undecorated Veros routine function
  - `copy_state()` jitted deep-copy path

## First JAX Translation Slice: Flux Kernels

- Completed the first incremental NumPy-to-JAX translation slice without changing the public `Coupler` / `Component` API.
- Translated `vercor/fluxes/utilities.py` to JAX-native array math:
  - inputs are coerced with `jnp.asarray`
  - NumPy-only ops were replaced with `jax.numpy`
  - the ECMWF hybrid-level altitude helper now uses JAX-safe padding instead of `np.insert`
- Rewrote `vercor/fluxes/bulk_formula_cesm.py` as JAX-native kernels:
  - `old_flux_atmOcn()`
  - `new_flux_atmOcn()`
  - `shr_flux_atmIce()`
  - shared stability / exchange-coefficient logic was factored into internal helpers to keep the two ocean schemes numerically aligned
- Made `new_flux_atmOcn()` compatible with `jax.jit` and reverse-mode AD by replacing the dynamic `lax.while_loop` attempt with a fixed two-step `lax.fori_loop` using masked carry updates.
- Tightened direct boundary adapters so JAX kernels run internally and NumPy conversion happens only where the external runtimes need it:
  - `vercor/components/external/veros_gcm.py`
  - `vercor/components/external/camulator.py`
  - `vercor/components/external/jax_gcm.py`

## Tests Added / Updated

- Extended `tests/test_fluxes_utilities.py` with:
  - `jax.jit` coverage for the translated utility kernels
  - `jax.jit` coverage for `new_flux_atmOcn()` and `shr_flux_atmIce()`
  - a finite-difference gradient smoke test for `new_flux_atmOcn()` sensible heat with respect to sea-surface temperature
- Extended `tests/test_external_tools_coverage.py` so `vercor/components/external/jax_gcm_tools.compute_pressure_levels()` is exercised under `jax.jit`.

## Validation (Flux Translation Slice, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_external_tools_coverage.py tests/test_fluxes_utilities.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Failed Approaches / Notes

- An initial `lax.while_loop` implementation for `new_flux_atmOcn()` was `jax.jit`-compatible but failed reverse-mode AD with:
  - `ValueError: Reverse-mode differentiation does not work for lax.while_loop ...`
- The final implementation uses a static two-iteration `lax.fori_loop`, which matches the current convergence limit and keeps gradients available.

## Next Translation Targets

- Second slice still pending:
  - `vercor/grid.py`
  - `vercor/regridders/helpers.py`
  - bilinear / conservative interpolation math
- Third slice still pending:
  - slab component pure kernels in `vercor/components/slab/`
  - `pure()` copy-before-mutate behavior
  - `set_variable()` interior update path
- Extended `tests/test_clock.py` for uncovered calendar helpers:
  - `isoformat()` and `timetuple()`
  - missing `day_of_year` validation in `timetuple()`
  - invalid `day_of_year` handling in `_month_day_from_day_of_year()`
  - negative ordinal overflow in `_from_ordinal_microseconds()`
  - `Clock.days_per_year` and `Clock.fixed_30_day_months` properties
- No production code changes were required; the work stayed in tests only.

## Coverage Outcome

- Overall `vercor` coverage increased from `73%` to `76%` via `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term-missing -q`.
- Module-level gains from the coverage run:
  - `vercor/coupler.py`: `67%` -> `95%`
  - `vercor/components/external/veros_gcm.py`: `57%` -> `73%`
  - `vercor/clock.py`: `81%` -> `86%`
  - `vercor/components/base.py`: `85%` -> `86%`
- The main remaining misses in `vercor/components/external/veros_gcm.py` are the heavy Veros kernel/setup regions (`set_forcing_kernel()` and `__init__()`), which are intentionally not exercised as real integrations in these unit tests.

## Validation (2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_clock.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed
- `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term-missing -q`
  - passed

## Notes / Failed Approaches

- `CustomGlobalFourDegree.set_diagnostics()` is wrapped by Veros runtime validation, so calling it directly in a unit test raises a `TypeError` unless the argument is a real `VerosState`. The test now calls the underlying routine function instead of going through the runtime wrapper.
- Per task instructions, no coverage work was added for:
  - `vercor/components/data/camulator_land.py`
  - `vercor/components/external/camulator.py`
  - `vercor/components/external/camulator_state.py`
  - `vercor/components/external/windpp.py`

## Second JAX Translation Slice 2A: Grid and Bilinear Regridding

- Completed the bilinear-first second translation slice without changing the public construction patterns used by components, examples, or tests.
- Translated `vercor/grid.py` to JAX-friendly grid holders:
  - `RectilinearGrid` now stores JAX arrays internally
  - eager validation for mask dimensionality and strict coordinate monotonicity is preserved
  - the legacy compact `__repr__` / `__str__` behavior is preserved
  - `RectilinearGrid` is now registered as a JAX PyTree
- Translated `vercor/regridders/helpers.py` to JAX-native helper kernels:
  - `make_rectilinear_grid()`
  - `centers_to_edges()`
  - `compute_land_mask()`
  - longitude clamping vs periodic-overhang behavior is preserved under `jax.jit`
- Rewrote `vercor/interpolators/bilinear_rectilinear.py` as a JAX-native interpolator:
  - all geometry helpers now use `jax.numpy`
  - scalar and vector apply paths are `jax.jit`-safe
  - extrapolation now uses JAX array operations instead of Python loops
  - periodic longitude, descending latitude, NaN renormalization, nearest / IDW fallback, and vector rotation behavior remain covered by the existing tests
  - the interpolator is registered as a JAX PyTree
- Kept the public regridder API stable:
  - `vercor/regridders/base.py` still dispatches scalar vs vector calls the same way
  - `vercor/regridders/bilinear.py` required no behavioral change
  - the conservative SciPy-backed remapper remains pending for the next slice

## Tests Added / Updated (Slice 2A)

- Extended `tests/test_helpers_coverage.py` with:
  - `RectilinearGrid` PyTree round-trip coverage
  - `jax.jit` coverage for `centers_to_edges()` and `compute_land_mask()`
- Extended `tests/test_bilinear_rectilinear_interpolator.py` with:
  - interpolator PyTree round-trip coverage
  - `jax.jit` coverage for scalar and vector interpolation
  - a gradient smoke test for scalar interpolation with respect to source field values
- Extended `tests/test_bilinear_rectilinear_regridder.py` so the bilinear regridder is exercised with JAX array input

## Validation (Slice 2A, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_helpers_coverage.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Failed Approaches / Notes (Slice 2A)

- Narrowing public signatures all the way to `jax.Array` broke `mypy` across the existing mixed NumPy/JAX call sites. The final version keeps the public interfaces NumPy-compatible while normalizing to JAX arrays internally.
- The conservative remapper rewrite remains intentionally deferred because its SciPy sparse representation needs a separate JAX-native design pass.

## Second JAX Translation Slice 2B: Conservative Remapping

- Replaced the conservative remapper runtime path in `vercor/interpolators/conservative_remap_rectilinear.py`:
  - removed the SciPy sparse dependency from runtime application
  - precompute now builds eager overlap triplets `(dst_index, src_index, weight)` in Python
  - scalar application now uses only `jax.numpy` gathers and indexed reductions
  - `src_lon_b`, `src_lat_b`, `dst_lon_b`, `dst_lat_b`, `dst_areas`, normalization semantics, periodic longitude handling, descending-latitude handling, source masking, and NaN behavior were preserved
  - `ConservativeRectilinearRemapper` is now registered as a JAX PyTree
- Kept the public conservative wrapper API stable in `vercor/regridders/conservative.py`; no constructor or call signatures changed.
- Extended conservative tests:
  - `tests/test_conservative_rectilinear_remapper.py`
    - PyTree round-trip coverage
    - `jax.jit` execution coverage for `apply_scalar()`
    - linearity + reverse-mode gradient smoke test with respect to the source field
  - `tests/test_conservative_rectilinear_regridder.py`
    - JAX-array input coverage through the public regridder call path

## Validation (Slice 2B Conservative Remapping, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_conservative_rectilinear_remapper.py -q`
  - passed
- `conda run -n scipy pytest tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Third JAX Translation Slice 3A: Slab Pure Kernels

- Translated the slab component compute paths to JAX while keeping the current component wrapper API unchanged:
  - `vercor/components/slab/atmosphere.py`
    - extracted pure JAX helpers for default SST, bulk-flux update, and 10 m wind construction
    - `initialize()` now seeds fields with `jnp.full` / `jnp.zeros`
    - `step()` now computes through JAX kernels and writes the results back to `self.data`
  - `vercor/components/slab/ocean.py`
    - extracted a pure JAX SST update kernel from sensible + latent heat fluxes, restoring, and `dt_seconds`
    - `initialize()` now seeds SST with `jnp.full`
  - `vercor/components/slab/land.py`
    - extracted a pure JAX soil-moisture update kernel using `jnp.clip`
    - `initialize()` now seeds soil moisture and land temperature with `jnp.full`
  - `vercor/components/slab/seaice.py`
    - extracted a pure JAX logistic ice-fraction diagnostic using `jnp.exp`
    - `initialize()` now seeds ice fraction with `jnp.zeros`
- Added dedicated slab-kernel tests in `tests/test_slab_kernels.py`:
  - `jax.jit` coverage for every new pure kernel
  - gradient smoke tests for atmosphere, ocean, land, and sea-ice kernels
  - edge cases for default SST, clipping, and cold-versus-warm sea-ice response
- Trimmed the slab portion of `tests/test_component_models_coverage.py` so it remains focused on wrapper-level initialization and dispatch behavior rather than duplicating all kernel math checks.

## Validation (Slice 3A Slab Kernels, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_slab_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_slab_kernels.py tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Third JAX Translation Slice 3B: Veros Helper Boundary

- Refactored the remaining Veros boundary helper math in `vercor/components/external/veros_gcm.py` without changing public component APIs:
  - added `_update_veros_interior()` as a private JAX helper for fixed `2:-2, 2:-2, ...` halo-preserving interior replacement
  - added `_prepare_surface_forcing_fields()` as a private JAX helper for transpose, singleton-axis expansion, `NaN` cleanup, and `qnec` gating by `restore_to_climatology`
  - kept `pure()` as the copy-before-mutate boundary helper for Veros runtime objects and clarified that scope in the docstring
  - narrowed `set_variable()` into a thin state adapter that copies the state, calls the JAX interior-update helper, and writes NumPy arrays back to the Veros state object
- Audited `compute_fluxes()` so the boundary math now stays JAX-native through masking, velocity interpolation, temperature assembly, and `qnet` / `qnec` construction until the final NumPy conversion required by the Veros adapter boundary.
- Extended `tests/test_external_components_coverage.py` with direct helper coverage:
  - `jax.jit` coverage and a gradient smoke test for `_update_veros_interior()`
  - helper coverage for `_prepare_surface_forcing_fields()` shape/orientation, `NaN` cleanup, and `restore_to_climatology=False` `qnec` zeroing
  - wrapper-level `VerosGCM.step()` coverage that confirms cleaned forcing payloads are what reach `set_variable()`
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 3B Veros Helper Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 3B)

- The numeric helper layer is now explicitly `jax.jit`-safe, but the full Veros runtime object model is still intentionally kept outside `jax.jit`; forcing the entire `pure()` / model-step boundary into JIT would require a broader redesign of how Veros state objects are represented.

## Fourth JAX Translation Slice 4A: JAXGCM Adapter Boundary

- Refactored `vercor/components/external/jax_gcm.py` so the JCM adapter now keeps its internal preprocessing and output mapping in JAX-native helpers while preserving the public wrapper API:
  - added `_cleanup_surface_temperature_fields()` as a private `jax.jit` helper for `NaN` cleanup, total surface temperature assembly, and cold-cell diagnostics
  - added `_prepare_surface_temperature_forcing()` as a private `jax.jit` helper for land/ocean masking and `288.15 K` zero-cell fallback before the JCM forcing boundary
  - added `_map_jcm_output_fields()` as a private `jax.jit` helper for transpose conventions, humidity conversion, flux sign handling, pressure assembly, density / potential-temperature diagnostics, and sigma-level height mapping
  - kept NumPy conversion only at the external forcing boundary passed into `self.forcing.copy(...)`
- Updated `JAXGCM.initialize()` to seed translated runtime fields with `jnp.zeros` / `jnp.full` instead of NumPy arrays.
- Kept `JAXGCM.step()` thin:
  - incoming land / sea surface temperatures are normalized once through the new helper
  - forcing fields are prepared through the new helper and converted to NumPy only when handed back to JCM
  - mapped JCM outputs are written back to `self.data` directly from the jitted helper output
- Extended `tests/test_external_components_coverage.py` with:
  - direct `jax.jit` coverage for all three new private helpers
  - a gradient smoke test for `_cleanup_surface_temperature_fields()`
  - wrapper-level regression assertions for total surface temperature assembly, forcing masking / fallback, transpose conventions, flux signs, humidity scaling, pressure/density/potential-temperature wiring, and output gating
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 4A JAXGCM Adapter Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_jax_gcm_output_frequency.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 4A)

- `Component.data` is still typed as `dict[str, NDArray]`, so directly assigning `jax.Array` values in `JAXGCM.initialize()` triggered `mypy` assignment errors. The final implementation keeps the JAX runtime values and uses narrow `cast(Any, ...)` annotations at initialization to satisfy the current shared component typing without widening that interface in this slice.

## Fifth JAX Translation Slice 5A: Shared Core Array Boundary

- Refactored the shared component storage layer so JAX arrays can move through the runtime core without being eagerly coerced to NumPy:
  - added `RuntimeArray` in `vercor/types.py` for mixed NumPy/JAX in-memory field storage
  - widened `TimedNamedArray.data`, `Shared.fields()`, `Component.data`, and `Component.get()` in `vercor/components/base.py`
  - removed the eager `np.asarray(...)` coercion in `Shared._assign_field()` so JAX-backed fields stay JAX-backed in runtime storage
  - kept explicit NumPy conversion at file/output boundaries in `TimedNamedArray.__array__()`, `ComponentForcingData._read_forcing()`, and `write_shared_to_netcdf()`
- Refactored the coupler dispatch boundary in `vercor/coupler.py`:
  - widened in-memory mask annotations to `RuntimeArray`
  - removed the unconditional `np.asarray(...)` cast around scalar regridder outputs in `interpolate_and_dispatch_fields()`
  - kept land/ocean mask creation and validation NumPy-backed where the existing helper logic depends on NumPy comparison semantics
- Cleaned up translated JAX component slices that no longer needed shared-storage casts:
  - removed now-unnecessary `cast(Any, ...)` assignments in `vercor/components/slab/atmosphere.py`, `vercor/components/slab/ocean.py`, `vercor/components/slab/land.py`, `vercor/components/slab/seaice.py`, and the JAX-backed field seeding path in `vercor/components/external/jax_gcm.py`
- Widened the time-slice helper signatures in `vercor/tools.py` to accept mixed runtime arrays so `Component.send_fields()` remains type-clean after the shared-core change.
- Extended coverage to lock the new runtime guarantees:
  - `tests/test_component_base_coverage.py` now verifies `Shared`, `receive_fields()`, `send_fields()`, and `get()` preserve JAX-backed arrays end to end while the netCDF writer still succeeds at the NumPy/xarray boundary
  - `tests/test_coupler_coverage.py` now verifies scalar exchange dispatch preserves JAX regridder outputs after masking and accepts mixed NumPy/JAX field flow
  - `tests/_coverage_support.py` now accepts mixed runtime arrays in the recording regridder scaffolding
- `DEPENDENCIES.md` did not require changes for this slice because the new runtime-array alias did not introduce a new module-level dependency edge.

## Validation (Slice 5A Shared Core Array Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 5A)

- No production redesign of mask-generation helpers was needed in this slice. Keeping those helpers NumPy-backed avoided broad churn in conservation/mask validation code while still removing the unnecessary runtime coercions from shared storage and scalar dispatch.

## Sixth JAX Translation Slice 6A: JAX-First Rectilinear Regridding

- Refactored the reusable rectilinear regridding core to remove the remaining NumPy-only validation and mask-plumbing paths while preserving public APIs and numerical behavior:
  - `vercor/grid.py`
    - replaced eager monotonicity validation with JAX-backed checks while keeping strict ascending-coordinate requirements and existing error text
  - `vercor/regridders/base.py`
    - replaced NumPy identical-grid detection with JAX-backed coordinate equality collapsed to a Python `bool`
  - `vercor/interpolators/bilinear_rectilinear.py`
    - removed NumPy from constructor monotonicity/orientation checks
    - switched the default `fill_value` to a Python `NaN` literal instead of `np.nan`
    - kept the existing JAX runtime interpolation/extrapolation path unchanged
  - `vercor/interpolators/conservative_remap_rectilinear.py`
    - cleaned up `apply_scalar()` shape validation and the source/destination mass helpers to use JAX-backed arrays end to end
    - intentionally left the eager overlap/precompute assembly in `__init__` host-side for now
  - `vercor/tools.py`
    - moved ocean/land mask construction, land/ocean mask-sum checks, conservation checks, and land-mask creation to JAX-first internal array handling
    - widened those helper signatures to accept `RuntimeArray` so NumPy and JAX callers remain type-clean
- Extended tests around the translated regridding core:
  - `tests/test_helpers_coverage.py`
    - JAX-backed `RectilinearGrid` construction and mask preservation
  - `tests/test_bilinear_rectilinear_interpolator.py`
    - JAX-array constructor inputs and longitude/latitude orientation flags
  - `tests/test_bilinear_rectilinear_regridder.py`
    - identical-grid scalar short-circuit with JAX-backed coordinates and JAX field input
  - `tests/test_conservative_rectilinear_regridder.py`
    - identical-grid scalar short-circuit with JAX-backed coordinates and JAX field input
  - `tests/test_conservative_rectilinear_remapper.py`
    - mass-helper coverage with JAX-array inputs
  - `tests/test_tools_assets_and_regridding.py`
    - JAX-backed inputs/outputs for mask clipping, mask-sum validation, and `create_lnd_mask_from_ocn()`
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 6A JAX-First Rectilinear Regridding, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_helpers_coverage.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_remapper.py tests/test_conservative_rectilinear_regridder.py tests/test_tools_assets_and_regridding.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 6A)

- A first pass kept NumPy-only type annotations on the regridding mask helpers after switching their internal logic to JAX arrays; `mypy` rejected the new JAX-backed call sites. Widening those helper signatures to `RuntimeArray` resolved the issue without changing runtime behavior.
- Full conservative remapper overlap assembly is still intentionally deferred. The current slice only translated validation/runtime application and mask plumbing because the overlap preprocessing is static setup code rather than the differentiable hot path.

## Sixth JAX Translation Slice 6B: Conservative Overlap Assembly

- Finished the remaining conservative-core translation work in `vercor/interpolators/conservative_remap_rectilinear.py` while keeping the public remapper and regridder call patterns unchanged.
- Replaced the NumPy-based overlap/precompute assembly in `ConservativeRectilinearRemapper.__init__()` with JAX-first eager helpers:
  - latitude standardization now uses JAX arrays and preserves the existing flipped-latitude behavior
  - interval-overlap assembly now builds dense destination-by-source overlap matrices with broadcasted `jax.numpy` min/max arithmetic
  - longitude periodic overlap now sums three shifted dense overlap matrices (`0`, `+360`, `-360`) before flattening, so duplicate shift contributions are merged without `np.unique` / `np.add.at`
  - source-mask filtering now drops invalid triplets through JAX boolean/index operations before storing `dst_indices`, `src_indices`, and `overlap_weights`
  - destination-area preparation is now JAX-native and still preserves the `np.inf`-equivalent zero-area sentinel behavior via `jnp.inf`
- Kept the constructor eager and host-side on purpose; the slice removes NumPy from the precompute math without trying to JIT the constructor itself.
- Cleaned the supporting wrapper in `vercor/regridders/conservative.py`:
  - removed the unused runtime NumPy import
  - switched the `fill_value` default literal from `np.nan` to `float("nan")`
  - widened `source_mask` typing to `RuntimeArray` so mixed NumPy/JAX callers remain type-clean
- Extended conservative tests to lock the new guarantees:
  - `tests/test_conservative_rectilinear_remapper.py`
    - JAX-backed constructor inputs for bounds and masks
    - periodic duplicate-shift overlap merging
    - eager masked-triplet dropping
  - `tests/test_conservative_rectilinear_regridder.py`
    - mixed NumPy/JAX edge arrays and JAX-backed `source_mask` through the public wrapper
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 6B Conservative Overlap Assembly, 2026-04-23)

- `conda run -n scipy pytest tests/test_conservative_rectilinear_remapper.py tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 6B)

- A first implementation used Python `sum(...)` to accumulate the three longitude-shift overlap matrices. `mypy` inferred that expression as `Array | Literal[0]`, so the final version uses explicit staged JAX-array accumulation instead.
- The first test pass also kept NumPy-only `NDArray` annotations on the conservative remapper/regridder constructor surfaces. `mypy` rejected the new JAX-backed test inputs until those annotations were widened to the shared `RuntimeArray` alias.

## Next Remaining Migration Targets

- Remaining NumPy-heavy production paths are now mostly outside the conservative-core hot path:
  - data adapters and forcing preparation in `vercor/components/data/`
  - explicit runtime-boundary cleanup still present in `vercor/coupler.py`
  - non-core utility and plotting helpers in `vercor/tools.py`
  - file/output boundaries in `vercor/components/base.py`, which should stay NumPy/xarray-backed unless there is a concrete reason to redesign them

## Seventh JAX Translation Slice 7A: JAX-First Data Adapters

- Translated the remaining in-scope data adapters to keep runtime arrays JAX-backed while preserving the existing public component APIs and NumPy/xarray file boundaries:
  - `vercor/components/data/era5_atmosphere.py`
    - forcing reads are normalized to `jnp.asarray(...)` at the component boundary
    - added private pure helpers for surface-pressure decoding, one-month diagnostic assembly, and total surface-temperature combination
    - `initialize()` now stacks per-month diagnostic outputs from the JAX helper back into runtime storage
  - `vercor/components/data/era5_ocean.py`
    - added private JAX helpers for ocean-mask derivation from land fraction and masked SST application
    - longitude/latitude, binary mask, and stored SST now stay JAX-backed in memory
  - `vercor/components/data/erainterim_ocean.py`
    - added private JAX helpers for global latitude assembly, full-grid field staging, binary-mask derivation, and masked SST application
    - the existing 1 degree vs 4 degree padding, longitude shift, and Celsius-to-Kelvin behavior were preserved
  - `vercor/components/data/jcm_land.py`
    - added a private JAX coordinate-conversion helper using `jnp.rad2deg`
    - stored land temperature and soil moisture now remain JAX-backed in memory
- Added a dedicated helper test module:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for ERA5 atmosphere pressure/diagnostic helpers and JCM coordinate conversion
    - reverse-mode gradient smoke test for the ERA5 atmosphere diagnostic helper
    - JAX-array input/output checks for ERA5 ocean and ERA-Interim ocean helper paths
- Updated `tests/test_component_models_coverage.py` only at the wrapper level:
  - constructors, masks, and shapes remain unchanged
  - translated components now explicitly preserve JAX-backed runtime arrays
- Updated `DEPENDENCIES.md` to include the translated data-adapter layer.

## Validation (Slice 7A JAX-First Data Adapters, 2026-04-23)

- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 7A)

- A first pass vectorized the ERA5 atmosphere monthly diagnostics with `jax.vmap`. That helper was numerically fine, but it broke the existing wrapper tests because they monkeypatch host-side physics helpers that call Python `float(...)` internally. The final implementation keeps the extracted one-month helper JAX-pure and `jax.jit`/`grad`-safe, while `initialize()` loops over months on the host and stacks the results.

## Seventh JAX Translation Slice 7B: Core Runtime NumPy Cleanup

- Cleaned the remaining in-scope runtime NumPy usage in the shared core while preserving the existing public APIs and explicit plotting / file-I/O NumPy boundaries:
  - `vercor/tools.py`
    - `safe_component_nanmean()` and `_safe_component_metric_mean()` now normalize through `jnp.asarray(...)` and compute NaN-aware reductions with `jax.numpy`
    - `grids_identical()` now uses JAX-backed coordinate comparisons collapsed to Python `bool`
    - `get_periodic_interval()` now computes host integer indices with pure scalar arithmetic instead of `np.array(..., dtype="int")`
    - plotting helpers intentionally remain NumPy / Matplotlib boundaries
  - `vercor/coupler.py`
    - removed the top-level NumPy dependency
    - default `_binary_masks` and `_fractional_masks` are now created as JAX arrays
    - `_create_exchange_masks()` now passes runtime arrays directly into `check_remap_conservation()`
    - `_validate_land_mask_consistency()` now compares masks with JAX-backed equality and mismatch counting
  - `vercor/regridders/bilinear.py`
    - removed the NumPy import used only for `np.nan`
    - switched the default `fill_value` to `float("nan")`
- Updated targeted tests:
  - `tests/test_tools_components_and_plotting.py`
    - JAX-backed coordinates for `grids_identical()`
    - JAX-backed component field input for `safe_component_nanmean()`
    - plotting path now consumes JAX-backed runtime fields while preserving the NumPy conversion boundary
  - `tests/test_tools_time_and_forcing.py`
    - added JAX-backed forcing-cube coverage for `get_field_at_specific_time()`
    - locked `get_periodic_interval()` indices to host `int` values
  - `tests/test_coupler_coverage.py`
    - added assertions that untouched default mask pools remain JAX-backed after `initialize()`
  - `tests/_tools_support.py`
    - widened `DummyGridComponent` test storage from NumPy-only arrays to the shared `RuntimeArray` alias so JAX-backed test inputs stay type-clean
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 7B Core Runtime NumPy Cleanup, 2026-04-23)

- `conda run -n scipy pytest tests/test_tools_components_and_plotting.py tests/test_tools_time_and_forcing.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_tools_components_and_plotting.py tests/test_tools_time_and_forcing.py tests/test_coupler_coverage.py -q`
  - passed
  - rerun after `black` reformatted `vercor/tools.py` and `tests/test_tools_components_and_plotting.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 7B)

- The first test-only pass fed JAX arrays into `DummyGridComponent`, but that helper was still typed as `dict[str, np.ndarray]`, so `mypy` rejected the updated coverage. The final version widens that test helper to the existing `RuntimeArray` alias instead of changing any production interface.

## Eighth JAX Translation Slice 8A: CAMulator Boundary

- Translated the in-scope CAMulator adapter boundary while preserving the explicit Torch, xarray, CREDIT, and file-output boundaries:
  - `vercor/components/external/camulator.py`
    - added a JAX-backed runtime-field initializer for exchange storage
    - added `_prepare_camulator_surface_forcing()` for NaN cleanup, land-mask fallback, and rescaling through `jax.numpy`
    - added `_map_camulator_prediction_arrays()` to map host-transferred CAMulator tensor outputs into JAX-backed VerCOR runtime fields
    - kept all Torch tensor creation, xarray output, and NetCDF writes host-side
  - `vercor/components/data/camulator_land.py`
    - initialized and stepped land surface temperature storage with JAX arrays
- Added `tests/test_camulator_component_kernels.py`:
  - `jax.jit` coverage for the CAMulator surface-forcing and prediction-mapping helpers
  - reverse-mode gradient smoke test for surface-forcing preparation
  - flux sign, pressure/height, shape, and JAX runtime-storage checks
  - lightweight patched CAMulatorLand coverage without real CAMulator model files
- Updated `DEPENDENCIES.md` to describe the CAMulator adapter and land forcing layer.

## Validation (Slice 8A CAMulator Boundary, 2026-04-24)

- `conda run -n scipy pytest tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 8A)

- The first flake8 pass surfaced an unrelated stale `numpy` import in `tests/_tools_support.py`; removing it restored the project lint count to `0`.

## Eighth JAX Translation Slice 8B: JAX-First Example Drivers

- Translated the remaining NumPy-heavy example drivers to use JAX-first runtime array handling while keeping plotting and external runtime boundaries explicit:
  - `examples/run_data_driver.py`
    - replaced the NumPy speed metric with a shared JAX helper
    - removed stale `NDArray` typing from the metric path
  - `examples/run_slab_driver.py`
    - replaced example mask construction and ice-fraction diagnostics with `jax.numpy`
  - `examples/run_jcm_with_slab.py`
    - replaced mask construction, coordinate conversion, and mask summaries with `jax.numpy`
  - `examples/run_jcm_with_era5data.py`, `examples/run_jcm_with_veros.py`, and `examples/run_jcm_with_verosdata.py`
    - replaced direct `np.array(...).T` terrain-mask mutation with an explicit JAX-to-host transfer helper
- Added `examples/jax_array_helpers.py` for example-local JAX diagnostics and explicit host transfer at third-party model boundaries.
- Added `examples/__init__.py` so `mypy` resolves the helper under a single module name.
- Added `tests/test_example_jax_helpers.py` to cover:
  - host transfer from JAX runtime arrays
  - transposed host transfer for mutable third-party masks
  - JAX-backed component vector-speed diagnostics
- No core coupler, component, or regridder APIs changed.
- `DEPENDENCIES.md` did not require changes because this slice only updated example-driver code and test coverage.

## Validation (Slice 8B Example Drivers, 2026-04-24)

- `conda run -n scipy pytest tests/test_example_jax_helpers.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 8B)

- The first `mypy` pass after adding the helper reported `examples/jax_array_helpers.py` twice, as both `jax_array_helpers` and `examples.jax_array_helpers`. Adding `examples/__init__.py` made `examples` an explicit package and resolved the duplicate-module error.

## Ninth JAX Translation Slice 9A: External Adapter Boundary Cleanup

- Tightened the remaining in-scope external adapter construction and initialization boundaries while preserving explicit third-party runtime transfers:
  - `vercor/components/external/jax_gcm.py`
    - JCM grid longitude/latitude and interpolation mask construction now use `jax.numpy`
    - `sigma_levels` storage now uses the shared mixed `RuntimeArray` alias instead of a NumPy-only annotation
    - NumPy conversion remains only at the JCM forcing/output boundary
  - `vercor/components/external/veros_gcm.py`
    - Veros grid mask derivation now uses JAX-backed array logic
    - initialized and refreshed sea-surface temperature storage is now explicitly JAX-backed
    - NumPy conversion remains at the Veros state mutation boundary
  - `vercor/components/external/camulator.py`
    - CAMulator static component mask construction now enters VerCOR as a JAX array
    - Torch, xarray, and NetCDF host boundaries remain unchanged
  - `vercor/components/external/jax_gcm_tools.py`
    - public helper annotations were widened from NumPy-only arrays to the shared `RuntimeArray` alias
- Extended targeted coverage:
  - `tests/test_external_components_coverage.py`
    - lightweight JAXGCM constructor coverage for JAX-backed grid and sigma-level storage
    - Veros constructor and runtime SST storage coverage for JAX-backed arrays
  - `tests/test_camulator_component_kernels.py`
    - lightweight CAMulatorGCM constructor coverage for JAX-backed static masks
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9A External Adapter Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_external_tools_coverage.py -q`
  - passed
- `conda run -n scipy mypy vercor/components/external/jax_gcm.py vercor/components/external/veros_gcm.py vercor/components/external/camulator.py vercor/components/external/jax_gcm_tools.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Ninth JAX Translation Slice 9B: Veros Boundary Cleanup

- Tightened the Veros adapter boundary in `vercor/components/external/veros_gcm.py` without changing the public component API:
  - `compute_fluxes()` now returns JAX arrays and keeps VerCOR-side flux math JAX-backed until the Veros state mutation boundary
  - added `_extract_surface_temperature()` as a private jitted helper for the repeated Veros SST readout and Celsius-to-Kelvin conversion
  - `VerosGCM.initialize()` and `VerosGCM.step()` now use the shared helper for JAX-backed SST storage
  - NumPy conversion remains explicit at `set_variable()` / Veros mutable-state handoff
- Extended `tests/test_external_components_coverage.py` with:
  - assertions that `compute_fluxes()` returns JAX-backed arrays while preserving existing sign and `qnec` masking behavior
  - `jax.jit` and reverse-mode gradient coverage for `_extract_surface_temperature()`
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9B Veros Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 9B)

- No failed implementation approaches. The slice kept Veros object mutation host-side and only moved the VerCOR-side flux/SST handoff later in the boundary.

## Ninth JAX Translation Slice 9C: CAMulator Forcing Boundary Cleanup

- Tightened the remaining CAMulator forcing-input boundary while preserving Torch, xarray, CREDIT, and NetCDF as explicit external runtime boundaries:
  - added JAX helpers in `vercor/components/external/camulator.py` for dynamic forcing layout conversion and CAMulator SST input expansion
  - added a single explicit JAX-to-host-to-Torch transfer helper for CAMulator step inputs
  - replaced inline dynamic forcing `np.stack(...)` and inline SST `torch.tensor(np.asarray(...))` staging in `CAMulatorGCM.step()`
  - replaced static forcing `np.stack(...)` in `vercor/components/external/camulator_state.py` with xarray `to_array(...)` staging before the Torch boundary
- Extended `tests/test_camulator_component_kernels.py` with:
  - `jax.jit` coverage for dynamic forcing layout conversion and SST input expansion
  - static forcing order/shape coverage through the xarray/Torch helper
  - lightweight patched `CAMulatorGCM.step()` coverage confirming dynamic forcing shape, SST tensor shape, and JAX-backed total surface temperature storage
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9C CAMulator Forcing Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 9C)

- The first helper implementation used `torch.as_tensor()` directly on a JAX host transfer, which produced a read-only NumPy-array warning from Torch. The final helper copies the host array before constructing the Torch tensor.
- The first `mypy` pass rejected fake test accessors assigned to `CAMulatorGCM` attributes; the final test casts the manually constructed component to `Any` because this is intentionally patched wrapper coverage rather than normal construction.

## Tenth JAX Translation Slice 10A: Shared Runtime Array Boundaries

- Tightened the shared component/tooling runtime-array boundaries without changing public component, coupler, or exchange APIs:
  - added `vercor.tools._runtime_array_to_host()` as the explicit JAX device-to-host transfer helper for NumPy-only consumers
  - normalized `get_field_time_slice()` and `get_field_at_specific_time()` through `jax.numpy` so sliced/interpolated fields return JAX-backed arrays for both NumPy and JAX input data
  - moved plotting data extraction in `vercor/tools.py` to use the explicit host-transfer helper at the Matplotlib boundary
  - moved `TimedNamedArray.__array__()` and `write_shared_to_netcdf()` in `vercor/components/base.py` to the same explicit host-transfer boundary
- Extended focused coverage:
  - `tests/test_tools_time_and_forcing.py`
    - asserts time slicing and monthly interpolation return `jax.Array` from NumPy-backed data
    - adds direct JAX-backed time-slice coverage
  - `tests/test_component_base_coverage.py`
    - asserts `TimedNamedArray.__array__()` works for JAX-backed data
    - writes JAX-backed shared fields and JAX-backed grid coordinates through NetCDF output
  - `tests/test_tools_components_and_plotting.py`
    - keeps mixed NumPy/JAX component plotting coverage and now uses JAX-backed grid coordinates in one plotted component
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 10A Shared Runtime Array Boundaries, 2026-04-24)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_tools_time_and_forcing.py tests/test_tools_components_and_plotting.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 10A)

- No failed implementation approaches. The slice only moves NumPy conversion to explicit host-only boundaries and keeps VerCOR runtime data JAX-backed.

## Tenth JAX Translation Slice 10B: External Host Transfer Centralization

- Centralized the remaining external-adapter host transfers on `vercor.tools._runtime_array_to_host()` while preserving explicit third-party runtime boundaries:
  - `vercor/components/external/jax_gcm.py`
    - replaced direct `np.asarray(...).transpose()` forcing handoffs with shared host transfers and `.T`
    - removed the now-unused NumPy import from the adapter
  - `vercor/components/external/veros_gcm.py`
    - widened `set_variable()` to accept mixed `RuntimeArray` inputs
    - moved the JAX-to-host conversion inside the Veros mutable-state boundary
    - stopped converting prepared forcing fields to NumPy before calling `set_variable()`
  - `vercor/components/external/camulator.py`
    - removed the adapter-local JAX-to-host helper
    - reused the shared host-transfer helper for Torch tensor staging and CAMulator output mapping inputs
    - removed the now-unused NumPy import from the adapter
- Extended focused boundary tests:
  - `tests/test_external_components_coverage.py`
    - asserts JAXGCM forcing copy receives host NumPy arrays with the existing transpose convention
    - asserts Veros `set_variable()` accepts JAX-backed inputs and stores host arrays at the mutation boundary
    - asserts `VerosGCM.step()` passes JAX-backed prepared forcing fields into `set_variable()`
  - `tests/test_camulator_component_kernels.py`
    - asserts CAMulator Torch staging copies host data so mutating the tensor does not mutate the JAX source
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 10B External Host Transfer Centralization, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed after Black reformatted `vercor/components/external/camulator.py`
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 10B)

- No failed implementation approaches. The slice keeps NumPy, xarray, Torch, Matplotlib, and Veros object mutation as explicit host-only boundaries.

## Eleventh JAX Translation Slice 11A: ERA5 Land Adapter

- Translated the ERA5 land forcing adapter runtime path while preserving the public component API and explicit h5netcdf/NumPy file-read boundary:
  - `vercor/components/data/era5_land.py`
    - added `_prepare_era5_land_runtime_fields()` for JAX-backed longitude, latitude, transposed land mask, and land surface temperature storage
    - `ERA5Land.__init__()` now normalizes forcing arrays through that helper before constructing the grid and storing `land_surface_temperature`
    - `initialize()` and `step()` remain no-op dataset-adapter hooks
- Extended focused coverage:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for the ERA5 land runtime-field helper
    - reverse-mode gradient smoke coverage for land surface temperature passthrough
  - `tests/test_component_models_coverage.py`
    - constructor coverage now asserts ERA5 land grid coordinates, binary mask, and runtime temperature storage are JAX-backed
- Updated `DEPENDENCIES.md` to include the ERA5 land forcing adapter layer.

## Validation (Slice 11A ERA5 Land Adapter, 2026-04-24)

- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py -q`
  - passed after Black reformatted `tests/test_data_component_kernels.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 11A)

- No failed implementation approaches. The slice keeps forcing file reads host-side and only normalizes the in-memory VerCOR runtime fields to JAX arrays.

## Eleventh JAX Translation Slice 11B: JAX-Backed Forcing Read Boundary

- Moved the shared forcing-read boundary to JAX-backed runtime storage while preserving the explicit h5netcdf/NumPy file-read boundary:
  - `vercor/components/base.py`
    - `_read_forcing()` now returns `RuntimeArray`
    - file loading remains host-side through h5netcdf and NumPy
    - transposed forcing arrays are normalized with `jnp.asarray(...)`
    - `flip_y=True` now uses `jnp.flip(..., axis=1)`
  - `vercor/components/data/era5_atmosphere.py`, `vercor/components/data/era5_ocean.py`, and `vercor/components/data/erainterim_ocean.py`
    - removed redundant `jnp.asarray(self._read_forcing(...))` wrappers now that `_read_forcing()` is the normalization point
- Extended focused coverage:
  - `tests/test_component_base_coverage.py`
    - asserts normal and flipped `_read_forcing()` calls return `jax.Array`
  - `tests/test_component_models_coverage.py`
    - widened patched `_read_forcing()` helper annotations to `RuntimeArray`
- No public component, coupler, exchange, or regridder APIs changed.

## Validation (Slice 11B JAX-Backed Forcing Read Boundary, 2026-04-24)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed
- `conda run -n scipy mypy vercor/components/base.py vercor/components/data/era5_atmosphere.py vercor/components/data/era5_ocean.py vercor/components/data/erainterim_ocean.py tests/test_component_base_coverage.py tests/test_component_models_coverage.py`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed after Black reformatted the ERA5 atmosphere and ERA5 ocean adapters
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 11B)

- No failed implementation approaches. The slice keeps h5netcdf/NumPy as the file-read boundary and makes the returned in-memory forcing arrays JAX-backed.

## Twelfth JAX Translation Slice 12A: Land Adapter JAX Boundary Cleanup

- Tightened the remaining land-adapter runtime boundary helpers without changing public component, coupler, exchange, or regridder APIs:
  - `vercor/components/data/jcm_land.py`
    - added `_prepare_jcm_land_runtime_fields()` for JAX-backed coordinate conversion plus transposed land temperature and soil-moisture storage
    - kept `_coordinates_in_degrees()` as the existing public test helper and routed it through the shared coordinate logic
    - `JCMLand.__init__()` now stores both land-surface temperature and soil moisture from the helper output
  - `vercor/components/data/camulator_land.py`
    - added `_prepare_camulator_land_surface_temperature()` for JAX-backed CAMulator land-temperature storage
    - `CAMulatorLand.step()` now uses the helper at the xarray-to-runtime boundary
- Extended focused coverage:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for the new JCM land runtime helper
    - reverse-mode gradient smoke coverage for JCM land temperature and soil-moisture passthrough
    - `jax.jit` coverage for the CAMulator land temperature helper
  - `tests/test_component_models_coverage.py`
    - asserts JCM land `soil_moisture` is JAX-backed, matching the existing temperature assertion
  - `tests/test_production_numpy_boundaries.py`
    - adds an AST-based production audit that limits NumPy imports to explicit host/file/plotting/type-boundary modules
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 12A Land Adapter JAX Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py tests/test_camulator_component_kernels.py tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 12A)

- No failed implementation approaches. The slice only adds named JAX helper boundaries around existing land-adapter behavior and leaves host-only NumPy boundaries explicit.

## Twelfth JAX Translation Slice 12B: Migration Completion Audit

- Tightened the production NumPy-boundary audit now that the NumPy-to-JAX migration phase is reduced to explicit host-only boundaries:
  - `tests/test_production_numpy_boundaries.py`
    - removed the stale CAMulator-state allowance from the direct NumPy boundary set
    - changed the assertion from subset matching to exact matching so new direct production NumPy imports fail immediately
    - preserved `veros.core.operators.numpy as npx` as a Veros backend boundary rather than a direct NumPy dependency
- Confirmed the remaining direct NumPy imports are intentionally limited to:
  - `vercor/components/base.py`
  - `vercor/tools.py`
  - `vercor/types.py`
- No public component, coupler, exchange, regridder, or runtime-array APIs changed.
- `DEPENDENCIES.md` did not require changes because this slice only tightens migration audit coverage.

## Validation (Slice 12B Migration Completion Audit, 2026-04-24)

- `conda run -n scipy pytest tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 12B)

- No failed implementation approaches. The slice intentionally keeps NumPy, xarray, Matplotlib, file output, and Veros backend integration as explicit host-only boundaries.

## Thirteenth JAX Translation Slice 13A: Differentiable Public Runtime

- Added the first pure differentiable runtime path while keeping the existing public component, coupler, exchange, regridder, and runtime-array APIs compatible:
  - `vercor/runtime.py`
    - added immutable PyTree containers for runtime field stores, component state, and coupler state
    - added pure exchange dispatch for scalar and vector exchanges
    - added receive/send helpers that update runtime field stores without mutating component objects
    - added pure slab-component stepping over the existing JAX kernels
  - `vercor/coupler.py`
    - routed `interpolate_and_dispatch_fields()` through the pure exchange dispatcher while preserving `Shared` / `TimedNamedArray` wrapper behavior
    - added `run_differentiable()` using `jax.lax.scan` over static run-sequence and exchange metadata
- The differentiable runtime currently supports VerCOR-owned slab components end to end. File I/O, plotting, Veros mutable state, Torch/CAMulator, xarray, and NetCDF remain explicit host-only boundaries.
- Updated `DEPENDENCIES.md` with the new runtime layer.

## Tests Added (Slice 13A)

- Added `tests/test_runtime_state.py` for PyTree round trips, immutable store updates, mapping conversion, and `jax.jit` coverage.
- Added `tests/test_runtime_exchange.py` for scalar mask dispatch, vector exchange dispatch, `jax.jit`, and gradients with respect to source fields and fractional masks.
- Added `tests/test_differentiable_coupler_runtime.py` for one-step and multi-step slab coupler runs under `jax.jit`, `jax.grad`, and `jax.jvp`.

## Validation (Slice 13A, 2026-04-24)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_slab_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_slab_kernels.py tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13A)

- The first scalar dispatch test expected `13.0` for a masked scaled field, but the correct sum is `12.0`; the test was corrected before implementation validation continued.
- The first slab-ocean closed-form test omitted the existing restoring term in `_advance_sea_surface_temperature()`; the test now includes that term.

## Thirteenth JAX Translation Slice 13B: Harden Differentiable Slab Runtime

- Hardened the public differentiable slab runtime path without changing the existing `run_differentiable(initial_state=None)` signature:
  - added `Coupler.create_differentiable_state(prefill_missing=True)` as the public immutable runtime-state builder
  - added preflight validation for configured run sequence, slab-only components, runtime-state component coverage, initialized regridders, and initialized fractional masks
  - `run_differentiable()` now validates both internally created states and caller-provided initial states before entering `jax.lax.scan`
- Made real VerCOR regridders safe to use inside the traced differentiable runtime by caching identical-grid status at regridder construction time instead of recomputing a Python `bool(...)` from JAX arrays inside the scan body.
- Extended `tests/test_differentiable_coupler_runtime.py`:
  - normal four-slab `Coupler` construction through `register()`, `add_exchange()`, `set_components_run_sequence()`, and `initialize()`
  - real bilinear regridder coverage under `jax.jit`
  - gradient coverage for final ocean SST with respect to initialized runtime-state SST
  - clear validation errors for missing run sequence, unsupported non-slab components, missing regridders, and missing fractional masks
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 13B, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13B)

- The first initialized-coupler JIT test failed because real regridders recomputed `has_identical_grids` inside `jax.lax.scan`, triggering `TracerBoolConversionError` from `bool(jnp.all(...))`. Caching the identical-grid result when the regridder is constructed keeps that branch static and preserves the existing identical-grid short-circuit behavior.

## Thirteenth JAX Translation Slice 13C: Mixed-Grid Differentiable Runtime Hardening

- Hardened the pure slab differentiable runtime for non-identical component grids without changing the public `run_differentiable(initial_state=None)` or `create_differentiable_state(prefill_missing=True)` APIs:
  - added mixed-grid four-slab coverage with ATM/LND on a 2x2 grid and OCN/ICE on a 3x3 grid
  - exercised real conservative OCN -> ATM remapping and real bilinear ATM -> OCN / OCN -> ICE remapping inside `jax.lax.scan`
  - kept external adapters, file I/O, plotting, Torch/CAMulator, xarray, NetCDF, and Veros object mutation outside the differentiable runtime path
- Strengthened differentiable-runtime preflight validation in `vercor/coupler.py`:
  - exported source fields must exist before entering the scan
  - source/data/incoming runtime fields must match their owning component grid shape
  - fractional masks must exist and match destination-grid shape
  - invalid caller-provided runtime states now fail with `CouplerError` before traced execution
- Extended `tests/test_differentiable_coupler_runtime.py` with:
  - mixed-grid `jax.jit`, `jax.grad`, and `jax.jvp` coverage
  - destination-shape assertions for conservative and bilinear exchange results
  - explicit validation coverage for missing source fields and fractional-mask shape mismatches
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 13C, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13C)

- The first mixed-grid test used ocean and atmosphere center coordinates whose inferred latitude bounds did not match, so initialization correctly failed the land-mask consistency check. The final test uses explicit matching cell bounds with different grid resolutions.

## Fourteenth JAX Translation Slice 14A: Data-Forcing Differentiable Runtime

- Extended the pure differentiable runtime beyond slab-only couplers while preserving the existing `run_differentiable(initial_state=None)` and `create_differentiable_state(prefill_missing=True)` APIs:
  - added `RuntimeStepInfo` as a JAX PyTree containing precomputed monthly interpolation indices/weights and daily time-slice indices
  - `Coupler.run_differentiable()` now scans over precomputed step metadata instead of deriving forcing times inside the traced body
  - runtime field sending now supports direct 2D fields, monthly interpolated forcing cubes, and daily time-sliced forcing arrays
  - differentiable component validation now accepts VerCOR slab components plus pure data-forcing adapters (`ERA5Atmosphere`, `ERA5Ocean`, `ERA5Land`, `ERAInterimOcean`, and `JCMLand`)
  - external runtime boundaries, including CAMulator-backed land forcing, were rejected at this stage; Slice 16A replaced this with unified runtime-state acceptance
- Added a pure data-component runtime step for `ERA5Atmosphere`:
  - combines imported land and sea surface temperatures into `total_surface_temperature`
  - keeps other supported data-forcing components as no-op steps whose runtime behavior is forcing replay through `send_component_fields()`
- Relaxed differentiable-runtime data-store validation so time cubes and auxiliary arrays can live in component data, while incoming/outgoing exchange fields are still required to match their component grid shape.
- Updated `DEPENDENCIES.md` to record that `vercor/runtime.py` now also depends on the pure data-forcing adapter layer.

## Tests Added / Updated (Slice 14A)

- Extended `tests/test_runtime_state.py` with:
  - `jax.jit` and reverse-mode gradient coverage for monthly runtime forcing interpolation
  - `jax.jit` and reverse-mode gradient coverage for daily runtime time slicing
- Extended `tests/test_differentiable_coupler_runtime.py` with:
  - a lightweight real data-component coupler using manually constructed `ERA5Ocean`, `ERA5Land`, and `ERA5Atmosphere` instances without asset downloads
  - gradient coverage through data-forcing replay into the atmosphere diagnostic
  - a data-to-slab runtime path using a real bilinear regridder
  - unsupported CAMulator land-boundary validation coverage

## Validation (Slice 14A, 2026-04-24)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14A)

- The first step-metadata implementation reused `get_field_time_slice()` with a JAX marker array during `run_differentiable()`. When called inside a jitted closure, converting that JAX scalar to `int` triggered `ConcretizationTypeError`; the final implementation computes daily indices with host scalar calendar logic before `jax.lax.scan`.
- The first `ERA5Atmosphere` data-runtime step added `total_surface_temperature` inside the scan body, changing the carry PyTree structure. The final implementation pre-seeds that diagnostic field in the runtime state and validates caller-provided states before traced execution.

## Fourteenth JAX Translation Slice 14B: Broaden Data-Forcing Runtime Coverage

- Hardened the pure differentiable data-forcing runtime with coverage for the remaining supported adapters without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - `ERAInterimOcean` monthly sea-surface-temperature forcing now has `run_differentiable()` coverage through real bilinear regridding into a slab atmosphere.
  - `JCMLand` daily land-surface-temperature forcing now has `run_differentiable()` coverage with `get_field_time_slice=True` into an ERA5-style data atmosphere.
  - Both paths are covered under `jax.jit` and reverse-mode gradients through selected forcing records.
- No production runtime changes were required; the existing `RuntimeStepInfo`, send-field selection, and supported-data-component dispatch paths already handled both adapters.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 14B, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed after Black reformatted `tests/test_differentiable_coupler_runtime.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14B)

- No failed implementation approaches. This slice was test-first runtime hardening, and the existing differentiable data-forcing path passed without production changes.

## Fourteenth JAX Translation Slice 14C: Calendar-Aware Differentiable Forcing Runtime

- Hardened the pure differentiable data-forcing runtime calendar coverage without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - daily `get_field_time_slice=True` forcing now has `run_differentiable()` coverage under a no-leap model calendar that skips Gregorian February 29.
  - daily 360-day forcing now verifies the runtime step metadata selects the same no-leap Gregorian day index as the host `get_field_time_slice()` helper.
  - monthly `apply_time_interpolation=True` forcing now has year-boundary wrap coverage under `jax.jit` and reverse-mode gradients.
- No production runtime changes were required; the existing host-precomputed `RuntimeStepInfo` path already matched the host forcing calendar helpers.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 14C, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14C)

- No failed implementation approaches. This slice was test-first calendar hardening, and the existing differentiable forcing metadata path passed unchanged.

## Fifteenth JAX Translation Slice 15A: JAXGCM Pure Runtime Integration

- Extended the immutable differentiable runtime so initialized JAXGCM components can participate in `run_differentiable()` without mutating the public wrapper object:
  - added optional runtime payload support to `RuntimeComponentState`
  - added `JAXGCMRuntimePayload` for immutable JCM state and forcing carry data
  - added JAXGCM support detection and preflight payload validation
  - added a pure JAXGCM runtime step that prepares surface-temperature forcing, calls the existing JCM step function, maps JCM outputs back into runtime fields, and skips prediction history / file output
- Preserved explicit host/runtime boundaries at this stage; this was superseded by Slice 16A:
  - CAMulator and Veros were still rejected by `run_differentiable()`
  - JAXGCM imperative `step()` still owns host transfers and output writing outside the pure runtime path
- Pre-seeded JAXGCM runtime output fields, including 3D pressure from sigma-level count, so `jax.lax.scan` carries a stable PyTree structure.
- Updated `DEPENDENCIES.md` with the JAXGCM runtime payload dependency edge.

## Tests Added / Updated (Slice 15A)

- Extended `tests/test_runtime_state.py` with optional payload PyTree and `jax.jit` coverage.
- Extended `tests/test_differentiable_coupler_runtime.py` with lightweight patched JAXGCM runtime coverage:
  - `jax.jit` execution through `run_differentiable()`
  - reverse-mode gradient flow from sea-surface temperature into JAXGCM output fields
  - wrapper state/forcing immutability assertions
  - missing-initialization and missing-payload validation
  - explicit Veros boundary rejection coverage
- Hardened `tests/test_external_components_coverage.py` by clearing the `_map_jcm_output_fields` JIT cache after monkeypatching its helper globals.

## Validation (Slice 15A, 2026-04-24)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 15A)

- The first JAXGCM runtime test failed because `component.settings` is the component time-selection settings, not the coupler physical constants used by JAXGCM output mapping. The runtime step now receives `Coupler.settings` explicitly.
- The first JAXGCM scan attempt added output fields inside the scan body, changing the carry PyTree structure. The coupler now pre-seeds all JAXGCM runtime output fields before scanning.
- Running JAXGCM runtime tests before external coverage exposed a cached `jax.jit` monkeypatch hazard in `_map_jcm_output_fields`; the affected test now clears the JIT cache after monkeypatching.

## Fifteenth JAX Translation Slice 15B: Differentiable Integration Hardening

- Hardened the pure differentiable runtime preflight checks without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - caller-provided runtime states now fail before `jax.lax.scan` if slab components are missing required data fields needed to preserve a stable carry PyTree
  - imported fields must be present in both incoming and data stores before traced receive/update logic can run
  - exported fields must be present in component data before traced send logic can run
  - ERA5Atmosphere data-runtime diagnostics now require land, sea, and total surface temperature fields up front
  - JAXGCM runtime states now validate all pre-seeded 2D output fields plus the sigma-level pressure field before traced execution
- Clarified unsupported external boundary errors at this stage; this was superseded by Slice 16A:
  - CAMulator components were reported as explicit host/runtime boundaries
  - VerosGCM was reported as an explicit host/runtime boundary
- Extended differentiable integration coverage:
  - data-forcing ERA5Ocean now replays into a JAXGCM runtime component under `jax.jit`, `jax.grad`, and `jax.jvp`
  - missing slab required data, missing import/export data, and missing JAXGCM preseeded pressure now raise `CouplerError` before traced execution
  - CAMulatorLand and VerosGCM rejection tests also asserted their VerCOR boundary data remained JAX-backed; Slice 16A replaced these with runtime-acceptance tests
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 15B, 2026-04-27)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 15B)

- No failed implementation approaches. The slice tightened preflight validation and expanded integration coverage while preserving the existing differentiable runtime APIs.

## Sixteenth JAX Translation Slice 16A: Unified Runtime Component Path

- Replaced the divergent imperative/differentiable coupler execution split with a shared runtime-state sequence:
  - `Coupler.run()` now builds `RuntimeCouplerState` and advances components through exchange dispatch, runtime receive, runtime step, runtime send, and wrapper commit.
  - `Coupler.run_differentiable()` remains as a compatibility entrypoint but delegates each scanned component step to the same runtime component helper used by `run()`.
  - legacy `Component.receive_fields()` and `Component.send_fields()` now delegate to runtime receive/send helpers.
- Added a component-owned runtime interface on `Component`:
  - `create_runtime_payload()`
  - `prefill_runtime_state_fields()`
  - `validate_runtime_state()`
  - `step_runtime_state()`
  - `commit_runtime_state()`
- Moved component-specific runtime stepping out of `vercor/runtime.py`:
  - slab atmosphere, ocean, land, and sea-ice runtime steps now live in their component files
  - ERA5 atmosphere surface-temperature diagnostics now live in `vercor/components/data/era5_atmosphere.py`
  - JAXGCM runtime payload, validation, and immutable step logic now live in `vercor/components/external/jax_gcm.py`
  - CAMulatorGCM and VerosGCM expose host-backed runtime-step overrides in their own component files
- Removed CAMulator and Veros runtime-validation rejection paths. They now create and validate runtime state through the same component interface, with host internals remaining explicit component-owned boundaries.
- Updated runtime/coupler tests so `Coupler.run()` is verified against runtime dispatch/receive/step/send order, and CAMulatorLand / VerosGCM are accepted by runtime state creation instead of rejected.

## Validation (Slice 16A, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed

## Notes / Failed Approaches (Slice 16A)

- The first runtime-interface type pass broadened `Component.step()` to `CustomDateTime`, which made external component overrides appear too narrow to `mypy`. The final interface uses `datetime | ModelDateTime`, while component methods that accept `CustomDateTime` remain valid broader overrides.

## Sixteenth JAX Translation Slice 16B: Runtime Compatibility API Cleanup

- Completed the follow-up unified runtime cleanup:
  - added canonical `Coupler.create_runtime_state(prefill_missing=True)`
  - kept `create_differentiable_state()` and `run_differentiable()` as compatibility aliases over the unified runtime path
  - made `Coupler.run()` return the final `RuntimeCouplerState` while preserving wrapper commits on the host path
  - replaced `interpolate_and_dispatch_fields()` internals with runtime exchange dispatch plus a wrapper-field commit
- Removed remaining duplicated compatibility helpers:
  - deleted stale slab/data runtime validators from `vercor/coupler.py`
  - deleted `is_supported_differentiable_component()` and `step_slab_component_state()` from `vercor/runtime.py`
  - removed no-op CAMulatorGCM and VerosGCM `step_runtime_state()` overrides so both use the shared base host-boundary implementation
  - removed the old non-runtime `Component.send_fields()` interpolation/time-slice fallback
- Updated tests so compatibility methods are checked as runtime delegates, while `create_runtime_state()` is covered as the canonical state factory.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 16B, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16B)

- An intermediate lint run caught a stale unused `Any` import in `vercor/components/external/camulator.py` after removing its no-op runtime override. The import was removed and flake8 then reported `0`.

## Sixteenth JAX Translation Slice 16C: Unified Runtime Cleanup Completion

- Completed the unified-runtime cleanup requested after Slice 16B:
  - canonicalized runtime naming in private coupler helpers and runtime-state docstrings
  - kept `create_differentiable_state()` and `run_differentiable()` only as compatibility delegates
  - moved generic import/export/incoming runtime validation into `Component.validate_runtime_state()`
  - updated slab, ERA5 atmosphere, and JAXGCM runtime validators to layer component-specific checks on top of the shared base validation
  - removed remaining component-category validation branching from `Coupler`
  - made `Coupler` create component runtime state through `Component.to_runtime_component_state(prefill_missing=...)`
  - thinned `Component.receive_fields()` to the runtime receive delegate
- Added regression coverage that:
  - `run()` and `run_differentiable()` both use `_step_runtime_component()`
  - `vercor/runtime.py` does not own component-specific step helpers or external-component payload classes
- Updated `DEPENDENCIES.md` to describe `run()` / `create_runtime_state()` as the canonical runtime path with compatibility aliases.

## Validation (Slice 16C, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16C)

- No failed implementation approaches. The slice kept Veros, CAMulator, Torch, xarray, NetCDF, and file output as explicit component-owned host boundaries while unifying the VerCOR runtime interface around immutable runtime state.

## Sixteenth JAX Translation Slice 16D: Remove Legacy Differentiable API

- Completed the final unified-runtime API cleanup:
  - removed `Coupler.create_differentiable_state()`, `Coupler.run_differentiable()`, and `Coupler.interpolate_and_dispatch_fields()`
  - made `Coupler.run(initial_state=None, commit_wrappers=True)` the only execution entrypoint, with `commit_wrappers=False` using the scanned JAX runtime path
  - moved receive/send runtime field handling from `vercor/runtime.py` into `Component.receive_runtime_fields()` and `Component.send_runtime_fields()`
  - removed `receive_component_fields()`, `send_component_fields()`, and `step_component_state()` from `vercor/runtime.py`
  - thinned pure slab, ERA5 atmosphere, and JAXGCM `step()` wrappers so component math goes through `step_runtime_state()`
- Renamed the large runtime integration test module from `tests/test_differentiable_coupler_runtime.py` to `tests/test_coupler_runtime.py`.
- Updated regression coverage so removed legacy API names are absent from `Coupler` and generic runtime.py no longer owns component receive/send/step dispatch.
- Updated `DEPENDENCIES.md` to describe `vercor/runtime.py` as immutable state plus generic exchange dispatch only.

## Validation (Slice 16D, 2026-04-27)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16D)

- The first thinned slab-ocean wrapper broke a direct unit test that calls `Ocean.step()` before initialization; the runtime step now preserves the old no-op behavior when SST is absent, while validated coupler runtime states still reject missing required SST before scans.
- The first JAXGCM external coverage update still monkeypatched the old `do_jcm_steps()` host path. The test now seeds the runtime payload and `_step_function` directly, matching the canonical component runtime path.

## Sixteenth JAX Translation Slice 16E: Unified Runtime Audit Completion

- Completed the unified-runtime audit / hardening pass requested after Slice 16D:
  - confirmed the legacy divergent public APIs remain absent (`run_differentiable`, `create_differentiable_state`, and `interpolate_and_dispatch_fields`)
  - confirmed generic runtime.py remains limited to immutable runtime state plus exchange dispatch, with component-specific runtime stepping kept in component modules
  - added direct `Coupler.run(..., commit_wrappers=False)` coverage for `CAMulatorGCM` so CAMulator atmosphere, CAMulator land, Veros, slab, data, and JAXGCM adapters are all covered through the canonical runtime-state path
- No production code changes were required; the slice only added targeted regression coverage.
- `DEPENDENCIES.md` did not require changes because no new module dependency edge was introduced.

## Validation (Slice 16E, 2026-04-27)

- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_run_accepts_camulator_gcm_runtime_boundary -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16E)

- No failed implementation approaches. The audit found only the missing direct CAMulatorGCM runtime-acceptance coverage, which is now locked by a lightweight test that avoids model-file, Torch, and xarray execution boundaries.

## Sixteenth JAX Translation Slice 16F: Runtime-First Component API Cleanup

- Completed the remaining wrapper-era component cleanup:
  - removed `Component.export_fields()` and `Component.import_fields()`
  - made `Component.initialize()` a concrete no-op default
  - made `Component.step()` a thin compatibility adapter over `step_runtime_state()`
  - made the default `Component.step_runtime_state()` a no-op immutable runtime transition
- Replaced the last production `component.import_fields(...)` call in `Coupler._commit_runtime_incoming_fields()` with direct assignment of runtime-built `Shared` incoming fields.
- Removed redundant no-op `initialize()` / `step()` implementations from data-forcing components and redundant `step()` wrappers from slab components and `ERA5Atmosphere`.
- Moved host-backed adapter stepping into component-owned runtime overrides:
  - `CAMulatorLand.step_runtime_state()`
  - `CAMulatorGCM.step_runtime_state()`
  - `VerosGCM.step_runtime_state()`
- Updated `JAXGCM.step_runtime_state()` so host bookkeeping, prediction storage, logging, and optional output happen when `time` and `coupler` are supplied, while scanned runtime execution remains side-effect free.
- Extended regression coverage so the removed component import/export API is absent, base `step()` delegates through runtime state, `Coupler` no longer calls `import_fields`, and external runtime stepping remains in component files.
- `DEPENDENCIES.md` did not require changes because the module dependency order and ownership descriptions stayed valid.

## Validation (Slice 16F, 2026-04-27)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_runtime.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16F)

- The first focused test run exposed old tests that constructed host-backed external components through `__new__()` and called `step()` directly, bypassing the base runtime state expected by the new compatibility wrapper. The tests now call `step_runtime_state()` with explicit `RuntimeComponentState` objects for those patched adapter-boundary cases.
- The first `mypy` pass rejected `CAMulatorLand.step_runtime_state()` because it accepted the broader `CustomDateTime` alias while `commit_runtime_state()` expects `ModelDateTime`; the annotation was narrowed to the base runtime contract.

## Wrapper Runtime Startup Prefill Fix

- Fixed the default host/wrapper `Coupler.run()` startup path so it creates the same prefilled and primed runtime state as the scanned runtime path when no explicit initial state is supplied.
- Preserved strict validation for caller-supplied `initial_state` objects; only the internally-created default state is prefilled.
- Added regression coverage for an initialized slab coupler whose wrapper incoming fields start empty but whose default `run()` still succeeds and commits imported runtime fields.
- Updated the wrapper-run coverage expectation to include startup outgoing-field priming before the first step.

## Validation (Wrapper Runtime Startup Prefill Fix, 2026-04-27)

- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_initialized_slab_coupler_wrapper_run_prefills_missing_imports -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed
- `MPLCONFIGDIR=/tmp/vercor-mplconfig MPLBACKEND=Agg conda run -n scipy python examples/run_slab_driver.py`
  - passed
- `MPLCONFIGDIR=/tmp/vercor-mplconfig MPLBACKEND=Agg conda run -n scipy python examples/run_data_driver.py`
  - passed

## Notes / Failed Approaches (Wrapper Runtime Startup Prefill Fix)

- The first new regression test asserted the full example-driver wind imports against the smaller existing slab test helper, which only imports heat fluxes into `OCN`; the final assertion uses the helper's actual imported fields while preserving the same missing-prefill failure mode.
- The first full-suite run exposed that the wrapper-run coverage test was still expecting no startup priming. The test now records the intentional initial `send_runtime_fields()` priming events before step dispatch.

## Unified Runtime Test Audit and Cleanup

- Audited `tests/` against the current canonical runtime API:
  - `Coupler.run(...)`
  - `Coupler.create_runtime_state(...)`
  - `Component.step_runtime_state(...)`
  - `Component.receive_runtime_fields(...)`
  - `Component.send_runtime_fields(...)`
- Confirmed removed wrapper-era APIs are referenced only by negative regression guards:
  - `run_differentiable`
  - `create_differentiable_state`
  - `interpolate_and_dispatch_fields`
  - `import_fields`
  - `export_fields`
  - `receive_component_fields`
  - `send_component_fields`
  - `step_component_state`
  - `step_slab_component_state`
  - `is_supported_differentiable_component`
- Rechecked patched external-component tests that construct components through `__new__()`:
  - host-backed Veros and CAMulator boundary tests now call `step_runtime_state()` with explicit `RuntimeComponentState` objects where they exercise runtime behavior
  - remaining direct `step()` tests cover the current compatibility wrapper, not removed APIs
- No stale behavior tests were found, so no unit tests were removed.
- Confirmed the production NumPy-boundary audit still limits direct NumPy imports to explicit host/type/output boundary modules:
  - `vercor/components/base.py`
  - `vercor/tools.py`
  - `vercor/types.py`

## Validation (Unified Runtime Test Audit and Cleanup, 2026-04-27)

- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ --collect-only -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully and left 83 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed after lint/type checks
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Unified Runtime Test Audit and Cleanup)

- No failed implementation approaches. The audit found only intentional absence guards for removed APIs, so deleting tests would have weakened regression coverage rather than removing stale behavior coverage.

## Test-Only Coupler Runtime Wrapper Removal

- Removed the private `Coupler._dispatch_runtime_fields()` and `Coupler._commit_runtime_incoming_fields()` compatibility wrappers from production code.
- Moved their remaining coverage-only behavior into local helpers in `tests/test_coupler_coverage.py`:
  - exchange dispatch now calls the canonical `dispatch_component_exchanges()` runtime function directly
  - wrapper incoming-field commit logic is now test-local for compatibility assertions only
- Extended removed-API regression coverage so `Coupler` is asserted not to expose either private test-only wrapper.
- No `DEPENDENCIES.md` update was required because this removed dead compatibility methods without changing module dependency order.

## Validation (Test-Only Coupler Runtime Wrapper Removal, 2026-04-27)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 83 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Test-Only Coupler Runtime Wrapper Removal)

- No failed implementation approaches. The cleanup stayed limited to private test-only compatibility wrappers and kept public/protocol/lifecycle APIs intact.
