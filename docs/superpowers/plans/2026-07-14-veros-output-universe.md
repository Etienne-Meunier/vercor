# Veros Output Universe Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the bundled Veros output provider from sampling setup-local fields such as `sss_clim` while preserving supported native output order and validation.

**Architecture:** Keep Veros's global `veros.variables.VARIABLES` mapping as the authoritative output registry. Filter the setup-state metadata against that registry before resolving active-variable dimensions, then let the existing shared coordinator apply the configured period selection unchanged.

**Tech Stack:** Python 3.13, Veros, JAX, NumPy, pytest, Black, flake8, mypy.

## Global Constraints

- Setup-local Veros fields such as `sss_clim` and `sst_clim` are excluded from the provider output universe.
- Explicit requests for excluded setup-local fields remain unknown-variable errors.
- Preserve state-manifest order for supported variables.
- Do not change public provider, coordinator, snapshot-default, or NetCDF-dimension contracts.
- Write the regression test before production code and observe the intended RED failure.
- Use the `scipy` Conda environment; invoke its executables directly because the local Conda wrapper crashes while loading its solver plugin.

---

### Task 1: Align Veros provider enumeration with extraction

**Files:**
- Modify: `tests/test_external_components_coverage.py:1506-1545`
- Modify: `vercor/setups/_external/veros_output.py:220-237`
- Modify: `PROGRESS.md:7`

**Interfaces:**
- Consumes: `_VerosOutputProvider.sample(context: OutputContext) -> OutputFrame` and `veros.variables.VARIABLES`.
- Produces: `_active_output_variable_names(veros_state: Any) -> tuple[str, ...]`, restricted to active, present, globally registered non-coordinate variables in state-manifest order.

- [ ] **Step 1: Extend the provider test with a setup-local field**

Add a dimension-resolution counter and an active `sss_clim` entry before constructing the provider:

```python
    local_dimension_resolutions = 0

    def local_dimensions(settings: Any) -> tuple[str, ...]:
        nonlocal local_dimension_resolutions
        _ = settings
        local_dimension_resolutions += 1
        return ("xt", "yt", "months")

    state.variables.sss_clim = np.ones((6, 7, 12), dtype=float)
    state.var_meta["sss_clim"] = SimpleNamespace(
        active=True,
        dims=local_dimensions,
    )
```

After the existing ordered-variable assertion, add:

```python
    assert "sss_clim" not in frame.variables
    assert local_dimension_resolutions == 0
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: FAIL while sampling with `ValueError: Unknown Veros output variable 'sss_clim'.` This proves the test exercises the reported provider path before the assertions.

- [ ] **Step 3: Apply the minimal registry filter**

Change the `active_metadata` comprehension in `_active_output_variable_names` to exclude unsupported names before resolving any metadata callables:

```python
    active_metadata = {
        name: variable
        for name, variable in metadata.items()
        if name in veros_variables.VARIABLES
        and bool(_resolve_metadata(variable.active, veros_state.settings))
        and hasattr(veros_state.variables, name)
    }
```

Do not change `normalize_veros_output_variables`, `_variable_definition`, provider signatures, or coordinator selection.

- [ ] **Step 4: Run the regression test and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: PASS. The existing ordered tuple remains unchanged, `sss_clim` is absent, and its dimension callable is never invoked.

- [ ] **Step 5: Run focused Veros output coverage**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py -q -k "veros and output"
```

Expected: all selected tests pass with no new warnings or errors.

- [ ] **Step 6: Format and run static checks**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py
/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py
git diff --check
```

Expected: Black reports the files unchanged or reformatted; flake8 reports zero findings; mypy reports success; `git diff --check` exits 0.

- [ ] **Step 7: Run repository regression suites**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/ -q --fast
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/ -q
```

Expected: both suites pass completely with no regressions.

- [ ] **Step 8: Update the project progress log**

Add this dated entry at the top of `PROGRESS.md` under `## Current Status`, retaining the verification commands' observed results:

```markdown
- Veros output-universe regression fixed locally on 2026-07-14. The bundled
  provider now excludes setup-local state metadata such as `sss_clim` before
  resolving dimensions, keeping enumeration aligned with Veros's global output
  registry while preserving supported manifest order and explicit
  unknown-variable validation. The focused provider regression, focused Veros
  output coverage, fast suite, full suite, Black, flake8, mypy, and whitespace
  checks pass.
```

- [ ] **Step 9: Review the final diff and commit the bug fix**

Run:

```bash
git diff -- vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py PROGRESS.md
git status --short
git add vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py PROGRESS.md
git commit -m "fix: exclude setup-local Veros output fields"
```

Expected: the diff contains only the registry filter, its regression coverage, and the progress entry; the commit succeeds after all required suites pass.
