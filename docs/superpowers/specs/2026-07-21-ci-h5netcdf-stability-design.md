# CI h5netcdf Stability Design

## Goal

Make the GitHub `quality` job deterministic in the pip-installed optional-model
environment while preserving VerCOR's existing `h5netcdf` I/O boundary and
offline distribution tests.

## Root causes

The failure has two independent sources:

1. The `quality` job runs distribution tests without downloading the artifact
   bundle produced by `build-artifacts`. The tests consequently attempt a fresh
   offline build. GitHub's hosted Python has `build` but not `flit_core`, and it
   has no Conda package cache fallback.
2. Installing JCM pulls in xarray's I/O extra and `netCDF4`. Current xarray
   prefers that backend when no engine is specified. The Linux pip HDF5 stack
   fails before VerCOR's forcing assertions run and while JCM reads its packaged
   terrain file. VerCOR's owned production I/O already uses `h5netcdf`.

## Selected design

### Artifact flow

Make `quality` depend on `build-artifacts`, download the existing three-file
artifact bundle, and expose it through `VERCOR_ARTIFACT_DIR` during pytest and
coverage. Distribution tests will validate the same wheel, sdist, and plugin
wheel used by the installed-artifact jobs instead of starting a second build.

### NetCDF fixtures

Every forcing-data test fixture that needs a valid NetCDF file will write it
with `engine="h5netcdf"`. The missing-mapping-key test will not create a file,
because mapping validation occurs before I/O. This keeps those tests focused on
`read_forcing`, whose production reader is also `h5netcdf`-based.

### JCM loading

Wrap only the two JCM packaged-data reads in a scoped xarray option that prefers
`h5netcdf`. Restore xarray's prior global option immediately afterward. On
xarray releases older than the engine-order option, retain the existing loading
behavior so VerCOR's declared older minimum does not break at import or runtime.

The scope belongs in `jax_gcm_tools.py`; VerCOR will not globally reconfigure
xarray and will not patch JCM internals.

## Rejected alternatives

- Uninstall or pin `netCDF4` only in CI: this hides the same failure from
  installed JCM users and makes the optional environment unlike a normal pip
  installation.
- Set xarray's engine order globally in `tests/conftest.py`: this masks the JCM
  integration defect and leaks mutable process-wide configuration.
- Disable HDF5 file locking or serialize pytest: the failures use distinct files
  in separate workers, so these changes treat an unsupported symptom rather
  than deterministic backend selection.
- Add `flit_core` only to the development extra: that permits duplicate builds
  but contradicts the existing build-once artifact architecture.

## Tests

TDD regressions will verify that:

- the `quality` workflow consumes `build-artifacts` and exports its directory;
- forcing fixtures request `h5netcdf` and the missing-key path performs no I/O;
- JCM packaged inputs load while `netCDF4` selection is disabled; and
- xarray's engine-order option is unchanged after loading.

Focused tests run first, followed by Black, strict flake8, mypy, compileall,
fast/full pytest, coverage, distribution probes, and whitespace checks.

## Non-goals

This change does not pin scientific packages, alter VerCOR's public API, change
numerical behavior, or modify JCM itself.
