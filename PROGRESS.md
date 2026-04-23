# 2026-04-22

## Test Harness Revision

- Baseline before revision:
  - `PROGRESS.md` was empty.
  - `conda run -n scipy pytest tests/ -q` passed with `119 passed` and a warning-heavy summary.
  - `conda run -n scipy pytest tests/ -q --fast` failed because `--fast` was not implemented.
- Implemented shared pytest infrastructure in `tests/conftest.py`:
  - added `--fast`
  - enabled JAX x64 for tests
  - added deterministic hash-based fast subsampling
  - guaranteed at least one test per file in `--fast` mode and marked representative grouped tests with `fast_always`
- Split `tests/test_tools.py` into:
  - `tests/test_tools_time_and_forcing.py`
  - `tests/test_tools_components_and_plotting.py`
  - `tests/test_tools_assets_and_regridding.py`
- Added shared helpers:
  - `tests/assertions.py` for compact numeric mismatch diagnostics
  - `tests/_tools_support.py` for shared dummy coupler/component fixtures
- Refactored numeric-heavy tests to use compact assertions and replaced manual `try/except` checks in `tests/test_hypsometric.py` with `pytest.raises(...)`.
- Added targeted warning handling for the known bilinear and conservative remap warnings.
- Updated pytest config in `pyproject.toml` to use quiet output and suppress the known third-party `dinosaur` deprecation noise.

## Validation

- `conda run -n scipy pytest tests/ -q`
  - passed
  - wall time: `31.50s`
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
  - wall time: `28.97s`
- File-level `-q` checks passed for every revised test module:
  - `tests/test_bilinear_rectilinear_interpolator.py`
  - `tests/test_bilinear_rectilinear_regridder.py`
  - `tests/test_clock.py`
  - `tests/test_conservative_rectilinear_regridder.py`
  - `tests/test_conservative_rectilinear_remapper.py`
  - `tests/test_fluxes_utilities.py`
  - `tests/test_hypsometric.py`
  - `tests/test_jax_gcm_output_frequency.py`
  - `tests/test_tools_assets_and_regridding.py`
  - `tests/test_tools_components_and_plotting.py`
  - `tests/test_tools_time_and_forcing.py`

## Notes

- The new `--fast` mode is deterministic and exercises every test file, but runtime is still import-bound, so the speedup is modest rather than dramatic.

## Time/Clock Typing and Coverage

- Cleared the remaining `mypy` failures in the time/calendar tests by:
  - adding a reusable typed `SelectFastCases` fixture protocol in `tests/conftest.py`
  - annotating the affected test functions in `tests/test_clock.py`, `tests/test_jax_gcm_output_frequency.py`, and `tests/test_tools_time_and_forcing.py`
  - changing `vercor.tools.get_field_at_specific_time()` to accept a structural protocol instead of the concrete `Coupler` type, matching the runtime interface it actually uses
- Increased bottom-up helper coverage for time-related code:
  - `tests/test_clock.py`: invalid clock configuration, invalid `day_of_year`, 360-day start-day validation, and microsecond rollover across a year boundary
  - `tests/test_tools_time_and_forcing.py`: Gregorian `datetime_to_seconds_in_year`, exact periodic interval boundaries, and year-end field interpolation wraparound
  - `tests/test_jax_gcm_output_frequency.py`: non-string output frequency handling, case-insensitive frequencies, and an explicit `_is_period_end()` false case

## Validation (2026-04-22)

- `conda run -n scipy black vercor tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor tests`
  - passed
- `conda run -n scipy pytest tests/test_clock.py tests/test_jax_gcm_output_frequency.py tests/test_tools_time_and_forcing.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Failed Approaches / Environment Notes

- `conda run -n scipy pytest tests/test_clock.py tests/test_jax_gcm_output_frequency.py tests/test_tools_time_and_forcing.py --cov=vercor.clock --cov=vercor.tools --cov=vercor.components.external.jax_gcm --cov-report=term-missing`
  - aborted in this environment with `Abort trap: 6`
  - do not use that command as the acceptance check for this task; rely on the green `black`/`flake8`/`mypy`/`pytest` validations above instead

## Core Unit Coverage Expansion

- Added typed shared unit-test scaffolding in `tests/_coverage_support.py` for dummy components, grids, coupler stubs, and recording regridders.
- Added focused coverage modules:
  - `tests/test_component_base_coverage.py`
    - `TimedNamedArray`, `Shared`, `Component`, `ComponentForcingData`, and `write_shared_to_netcdf`
    - import/export validation, timestamp mismatch handling, send-field mode selection, finalize wiring, forcing reads, and netCDF output
  - `tests/test_component_models_coverage.py`
    - slab `Atmosphere`, `Ocean`, `Land`, `SeaIce`
    - constructor and masking logic for `ERA5Land`, `ERA5Ocean`, and `ERAInterimOcean` with patched forcing reads
  - `tests/test_helpers_coverage.py`
    - `Grid`, `RectilinearGrid`, `centers_to_edges()`, `compute_land_mask()`, `Exchange`, and `RunSequence`
  - `tests/test_coupler_coverage.py`
    - duplicate registration, run-sequence validation, missing exchange endpoints, mask patching, land-mask consistency checks, output-mask appending, scalar/vector dispatch, and empty-outgoing-field rejection in `run()`
- Coverage increased from `55%` to `66%` for `vercor` via `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term -q`.
- Module-level gains from the coverage run:
  - `vercor/components/base.py`: `28%` -> `85%`
  - `vercor/components/data/era5_land.py`: `44%` -> `96%`
  - `vercor/components/data/era5_ocean.py`: `41%` -> `97%`
  - `vercor/components/data/erainterim_ocean.py`: `33%` -> `97%`
  - `vercor/components/slab/atmosphere.py`: `28%` -> `100%`
  - `vercor/components/slab/ocean.py`: `29%` -> `94%`
  - `vercor/components/slab/land.py`: `56%` -> `100%`
  - `vercor/components/slab/seaice.py`: `48%` -> `100%`
  - `vercor/coupler.py`: `19%` -> `67%`
  - `vercor/exchange.py`: `79%` -> `100%`
  - `vercor/grid.py`: `78%` -> `98%`
  - `vercor/regridders/helpers.py`: `72%` -> `87%`
  - `vercor/run_sequence.py`: `86%` -> `100%`

## Validation (Coverage Expansion, 2026-04-22)

- `conda run -n scipy black vercor tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor tests`
  - passed
- `conda run -n scipy pytest tests/test_component_base_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_helpers_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed
- `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term -q`
  - passed

## Notes / Gotchas

- `ComponentForcingData._read_forcing()` currently routes a missing variable name through its `KeyError` handler, so the exception message refers to the `where` key even when the real issue is a missing variable in the file. The new tests document the current behavior without changing production code.

## External/Data Coverage Expansion

- Extended `tests/test_component_models_coverage.py` to cover the remaining in-scope data components:
  - `vercor/components/data/era5_atmosphere.py`
    - constructor defaults and file lookup
    - flipped latitude handling and vertical-level slicing
    - `initialize()` monthly pressure/height/density/potential-temperature wiring with patched physics helpers
    - `step()` total surface temperature aggregation with `NaN` handling
  - `vercor/components/data/jcm_land.py`
    - radian-to-degree coordinate conversion
    - land-mask wiring via patched `create_lnd_mask_from_ocn()`
    - transposed forcing fields
    - `get_field_time_slice` enabled
    - no-op `initialize()` / `step()` execution
- Added `tests/test_external_tools_coverage.py` for helper-heavy external code:
  - `vercor/components/external/jax_gcm_tools.py`
    - parameter mutation/default lookup helpers
    - `compute_pressure_levels()` normal and validation branches
    - both `generate_jcm_coords_forcing_topography_files()` path branches
    - `mean_leaf()`, `unwrap_leading_dims()`, `stack_objects()`, and `concat_objects()`
  - `vercor/components/external/veros_gcm.py`
    - `compute_fluxes()` sign conventions and `dqfldt` masking
    - non-jitted `copy_state()`, `pure()`, and `set_variable()`
- Added `tests/test_external_components_coverage.py` for wrapper behavior without full model integration runs:
  - `vercor/components/external/jax_gcm.py`
    - `asfloat()`
    - non-jitted `_generate_step_function()`
    - `do_jcm_steps()`
    - `initialize()` forcing selection, timestep guard, and optional spinup
    - `step()` field mapping/output gating
    - `_write_output()` NetCDF write and prediction-list reset
  - `vercor/components/external/veros_gcm.py`
    - `initialize()` timestep guard, spinup path, and SST extraction
    - `step()` forcing-field dispatch for `restore_to_climatology=True/False`
- No production code changes were required; the work stayed entirely in the test suite.
- Coverage increased from `66%` to `73%` for `vercor` via `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term-missing -q`.
- Module-level gains from the coverage run:
  - `vercor/components/data/era5_atmosphere.py`: `23%` -> `96%`
  - `vercor/components/data/jcm_land.py`: `57%` -> `100%`
  - `vercor/components/external/jax_gcm.py`: `39%` -> `88%`
  - `vercor/components/external/jax_gcm_tools.py`: `53%` -> `100%`
  - `vercor/components/external/veros_gcm.py`: `21%` -> `57%`

## Validation (External/Data Coverage Expansion, 2026-04-22)

- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_external_tools_coverage.py tests/test_external_components_coverage.py -q`
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

## Scope Notes

- Per task instructions, no coverage was added for:
  - `vercor/components/data/camulator_land.py`
  - `vercor/components/external/camulator.py`
  - `vercor/components/external/camulator_state.py`
  - `vercor/components/external/windpp.py`

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
