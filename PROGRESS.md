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
