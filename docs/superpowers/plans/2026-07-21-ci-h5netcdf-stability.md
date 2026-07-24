# CI h5netcdf Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GitHub quality suite reuse built distributions and use VerCOR's supported `h5netcdf` path for forcing fixtures and JCM packaged inputs.

**Architecture:** The quality job consumes the existing `build-artifacts` bundle through `VERCOR_ARTIFACT_DIR`. Tests create valid fixtures with an explicit `h5netcdf` engine, while the optional JCM adapter uses a short-lived xarray engine preference that is restored after both packaged files are loaded.

**Tech Stack:** Python 3.12+, pytest/xdist, xarray, h5netcdf, JCM 1.1.1, GitHub Actions YAML.

## Global Constraints

- Do not pin or uninstall `netCDF4`.
- Do not change global xarray configuration outside the JCM loading scope.
- Preserve xarray compatibility when `netcdf_engine_order` is unavailable.
- Preserve VerCOR's public API and numerical results.
- Run tests through `/Users/romannuterman/miniforge3/envs/scipy/bin/python`.

---

### Task 1: Reuse the built artifact bundle in quality

**Files:**
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `.github/workflows/python-package.yml`

**Interfaces:**
- Consumes: the `vercor-distributions` artifact uploaded by `build-artifacts`.
- Produces: `VERCOR_ARTIFACT_DIR=${{ github.workspace }}/dist` for both quality pytest invocations.

- [x] **Step 1: Extend the quality workflow contract test**

Add these assertions to `test_ci_quality_job_enforces_static_full_and_coverage_gates` after `quality` is loaded:

```python
    assert quality["needs"] == "build-artifacts"
    download = next(
        step
        for step in quality["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["with"] == {
        "name": "vercor-distributions",
        "path": "dist/",
    }
    assert quality["env"]["VERCOR_ARTIFACT_DIR"] == (
        "${{ github.workspace }}/dist"
    )
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_ci_quality_job_enforces_static_full_and_coverage_gates -q -n0 --tb=short
```

Expected: FAIL because `quality` has no `needs`, artifact download, or artifact-directory environment.

- [x] **Step 3: Wire the quality job to the artifact producer**

Change the start of the job and insert the download after Python setup:

```yaml
  quality:
    needs: build-artifacts
    runs-on: ubuntu-latest
    env:
      VERCOR_ARTIFACT_DIR: ${{ github.workspace }}/dist
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/download-artifact@v4
        with:
          name: vercor-distributions
          path: dist/
```

- [x] **Step 4: Run the focused workflow tests and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_ci_validates_installed_artifacts_across_supported_environments tests/test_distribution_boundaries.py::test_ci_quality_job_enforces_static_full_and_coverage_gates tests/test_distribution_boundaries.py::test_distribution_helper_reuses_explicit_artifact_directory_without_building -q -n0 --tb=short
```

Expected: 3 passed.

- [x] **Step 5: Commit the artifact-flow fix**

```bash
git add .github/workflows/python-package.yml tests/test_distribution_boundaries.py
git commit -m "fix: reuse built artifacts in quality CI"
```

### Task 2: Make forcing-data fixtures backend-explicit

**Files:**
- Modify: `tests/test_forcing_data.py`

**Interfaces:**
- Consumes: `xarray.Dataset.to_netcdf(..., engine="h5netcdf")`.
- Produces: deterministic valid NetCDF fixtures for `vercor.forcing_data.read_forcing`.

- [x] **Step 1: Reproduce the existing fixture/backend mismatch**

Run the affected tests with xarray's installed default engine and record that the test source makes four implicit `to_netcdf(path)` calls:

```bash
rg -n 'to_netcdf\(path\)' tests/test_forcing_data.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_forcing_data.py -q -n4 --dist=loadscope --max-worker-restart=0 --tb=short
```

Expected before the edit: four implicit calls are reported; the tests may pass in Conda because its HDF5 libraries are unified, while the approved CI RED evidence fails in `netCDF4` before `read_forcing` executes.

- [x] **Step 2: Make every valid fixture use h5netcdf**

Change the two success fixtures and the missing-variable fixture to:

```python
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(
        path,
        engine="h5netcdf",
    )
```

and:

```python
    xr.Dataset({"foo": (("x",), np.asarray([1.0]))}).to_netcdf(
        path,
        engine="h5netcdf",
    )
```

Replace the missing-mapping-key setup with no file creation:

```python
def test_read_forcing_reports_missing_mapping_key() -> None:
    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        read_forcing({"sample": "unused.nc"}, "foo", "missing")
```

- [x] **Step 3: Verify the forcing focus is GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_forcing_data.py -q -n4 --dist=loadscope --max-worker-restart=0 --tb=short
```

Expected: 5 passed without the xarray `netCDF4` write warning.

- [x] **Step 4: Commit the fixture fix**

```bash
git add tests/test_forcing_data.py
git commit -m "test: use h5netcdf forcing fixtures"
```

### Task 3: Scope h5netcdf preference around JCM packaged inputs

**Files:**
- Modify: `tests/test_external_tools_coverage.py`
- Modify: `vercor/setups/_external/jax_gcm_tools.py`

**Interfaces:**
- Consumes: `xarray.get_options()` and, when available, `xarray.set_options(netcdf_engine_order=...)`.
- Produces: `_prefer_h5netcdf() -> AbstractContextManager[object]`; JCM terrain and forcing reads execute inside it and prior xarray state is restored.

- [x] **Step 1: Add the scoped-engine regression**

Import xarray in `tests/test_external_tools_coverage.py`, then add:

```python
def test_load_jcm_inputs_scopes_h5netcdf_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = SimpleNamespace(name="coords")
    observed_orders: list[tuple[str, ...]] = []
    original_order = tuple(xr.get_options()["netcdf_engine_order"])

    monkeypatch.setattr(
        jax_gcm_tools_module,
        "get_speedy_coords",
        lambda spectral_truncation: coords,
    )

    def record_order(path: Path, coords: Any) -> str:
        _ = path, coords
        observed_orders.append(
            tuple(xr.get_options()["netcdf_engine_order"])
        )
        return "loaded"

    monkeypatch.setattr(
        jax_gcm_tools_module.TerrainData,
        "from_file",
        staticmethod(record_order),
    )
    monkeypatch.setattr(
        jax_gcm_tools_module.ForcingData,
        "from_file",
        staticmethod(record_order),
    )

    _, terrain, forcing = (
        jax_gcm_tools_module.load_jcm_coords_terrain_forcing(
            input_data_directory=Path("/tmp/jcm-inputs"),
        )
    )

    assert terrain == "loaded"
    assert forcing == "loaded"
    assert len(observed_orders) == 2
    assert all(order[0] == "h5netcdf" for order in observed_orders)
    assert tuple(xr.get_options()["netcdf_engine_order"]) == original_order
```

- [x] **Step 2: Run the new test and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_tools_coverage.py::test_load_jcm_inputs_scopes_h5netcdf_preference -q -n0 --tb=short
```

Expected: FAIL because the observed default order begins with `netcdf4`.

- [x] **Step 3: Add the compatibility-aware context manager**

Add imports and the private helper to `jax_gcm_tools.py`:

```python
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext

import xarray as xr


def _prefer_h5netcdf() -> AbstractContextManager[object]:
    """Temporarily prefer h5netcdf when xarray supports engine ordering."""

    options = xr.get_options()
    if "netcdf_engine_order" not in options:
        return nullcontext()
    configured = tuple(
        cast(Sequence[str], options["netcdf_engine_order"])
    )
    preferred = (
        "h5netcdf",
        *(engine for engine in configured if engine != "h5netcdf"),
    )
    return xr.set_options(netcdf_engine_order=preferred)
```

Add `cast` to the existing `typing` import:

```python
from typing import Optional, Tuple, cast
```

Wrap the external reads:

```python
    with _prefer_h5netcdf():
        terrain = TerrainData.from_file(terrain_file, coords=coords)
        forcing = ForcingData.from_file(forcing_file, coords=coords)
```

- [x] **Step 4: Add and verify the older-xarray fallback**

Add:

```python
def test_prefer_h5netcdf_is_noop_without_engine_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jax_gcm_tools_module.xr, "get_options", lambda: {})

    with jax_gcm_tools_module._prefer_h5netcdf():
        pass
```

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_tools_coverage.py::test_load_jcm_inputs_scopes_h5netcdf_preference tests/test_external_tools_coverage.py::test_prefer_h5netcdf_is_noop_without_engine_order tests/test_coupler_runtime.py::test_real_jax_gcm_initial_payload_seeds_speedy_coords -q -n0 --tb=short
```

Expected: 3 passed; the real packaged-data test may emit only its existing JCM/xarray future warning.

- [x] **Step 5: Run the complete JCM/tools focus and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_tools_coverage.py tests/test_setup_lifecycle_helpers.py tests/test_coupler_runtime.py -q -n4 --dist=loadscope --max-worker-restart=0 --tb=short
```

Expected: all selected tests pass without `NetCDF: HDF error`.

- [x] **Step 6: Commit the production integration fix**

```bash
git add vercor/setups/_external/jax_gcm_tools.py tests/test_external_tools_coverage.py
git commit -m "fix: prefer h5netcdf for JCM inputs"
```

### Task 4: Verify release gates and update project memory

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verified CI-stability change and a durable progress entry.

- [x] **Step 1: Run formatting and static gates**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black --check vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor examples tests
git diff --check
```

Expected: all commands exit 0; Black changes no files, flake8 reports 0, mypy reports success, and compileall/whitespace checks are silent.

- [x] **Step 2: Run fast and full pytest**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
```

Expected: both suites pass without NetCDF/HDF failures.

- [x] **Step 3: Run branch coverage**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
```

Expected: suite passes with branch coverage at or above 90%.

- [x] **Step 4: Record the durable result**

Add this dated item at the start of `PROGRESS.md` Current Status and append the actual gate counts from Steps 1-3:

```markdown
- CI artifact and NetCDF backend stability completed locally (2026-07-21): the quality job now reuses the build-once artifact bundle, forcing fixtures explicitly use h5netcdf, and JCM packaged input loading temporarily prefers h5netcdf without leaking xarray configuration. The focused, static, fast, full, and coverage gates passed without NetCDF/HDF failures.
```

- [x] **Step 5: Commit progress documentation**

```bash
git add PROGRESS.md docs/superpowers/plans/2026-07-21-ci-h5netcdf-stability.md
git commit -m "docs: record CI stability verification"
```

- [x] **Step 6: Inspect the final branch**

```bash
git status --short --branch
git log -6 --oneline --decorate
```

Expected: the worktree is clean and the new focused commits follow the pre-existing JAX dtype commit and design commit.
