# Veros Duplicate-Dimension Output Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the bundled Veros provider from sampling active variables with repeated dimension names, unavailable effective coordinates, or adapter-coordinate name collisions, so selected output from `run_jcm_with_veros.py` accumulates successfully.

**Architecture:** Keep provider sampling and coordinator-owned variable selection unchanged. Resolve each active Veros candidate's dimensions once during provider-universe enumeration, use that mapping for coordinate discovery, and omit candidates that violate `OutputVariable`'s unique-dimension invariant, require a non-timestep coordinate that the Veros adapter cannot construct, or collide with the adapter-owned time coordinate.

**Tech Stack:** Python 3.13, Veros 1.6.2, JAX, NumPy, pytest, Black, flake8, mypy.

## Global Constraints

- Preserve state-manifest order for supported variables.
- Preserve active-state, value-presence, global-registry, and coordinate-variable filtering.
- Do not change public provider, coordinator, snapshot-default, `OutputVariable`, or NetCDF contracts.
- Do not invent distinct names for `line_psin`'s two `isle` axes.
- Do not invent definitions for the absent `tensor1` and `tensor2` coordinates.
- Preserve variables with a `timesteps` dimension because extraction removes that axis before coordinate construction.
- Reserve `VEROS_TIME_DIM` for the adapter-owned coordinate; do not emit native scalar `time` as a data variable.
- Write the regression test before production code and observe the intended RED failure.
- Use the direct `/Users/romannuterman/miniforge3/envs/scipy/bin/` executables because the local `conda run` path panics while loading Rattler.
- Update `PROGRESS.md` after the tested implementation is complete.

---

### Task 1: Exclude Unrepresentable Veros Variables

**Files:**
- Modify: `tests/test_external_components_coverage.py:1506-1550`
- Modify: `vercor/setups/_external/veros_output.py:220-238`
- Modify: `PROGRESS.md:7`

**Interfaces:**
- Consumes: `_resolved_dims(variable: Any, settings: Any, name: str) -> tuple[str, ...]` and the insertion-ordered `veros_state.var_meta` mapping.
- Produces: `_active_output_variable_names(veros_state: Any) -> tuple[str, ...]`, restricted to active, present, globally registered, non-coordinate variables whose resolved dimensions are unique, whose effective non-timestep coordinates are constructible, and whose names do not collide with adapter-owned coordinates.

- [ ] **Step 1: Add the real repeated-dimension candidate to the provider regression**

In `test_veros_output_provider_exposes_active_native_variable_universe`, add the following immediately after assigning `surface_tauy` and before the setup-local `sss_clim` fixture:

```python
    state.variables.line_psin = np.ones((6, 6), dtype=float)
    state.var_meta["line_psin"] = SimpleNamespace(
        active=True,
        dims=("isle", "isle"),
    )
```

After the existing `assert "sss_clim" not in frame.variables`, add:

```python
    assert "line_psin" not in frame.variables
```

- [ ] **Step 2: Run the focused regression and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: FAIL during `provider.sample(...)` with `ValueError: OutputVariable.dims must be unique`. This proves the test reaches the reported provider path before its final assertions.

- [ ] **Step 3: Resolve dimensions once and filter repeated names**

Replace the coordinate-name and return block in `_active_output_variable_names` with:

```python
    dimensions_by_name = {
        name: _resolved_dims(variable, veros_state.settings, name)
        for name, variable in active_metadata.items()
    }
    coordinate_names = {
        dim for dims in dimensions_by_name.values() for dim in dims
    }
    return tuple(
        name
        for name, dims in dimensions_by_name.items()
        if name not in coordinate_names and len(set(dims)) == len(dims)
    )
```

Do not change `_extract_variable`, `OutputVariable`, provider signatures, coordinator selection, or snapshot defaults.

- [ ] **Step 4: Run the focused regression and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: PASS. The existing supported tuple remains unchanged, `sss_clim` and `line_psin` are absent, and the setup-local dimension resolver remains uncalled.

- [ ] **Step 4a: Add an unavailable-coordinate candidate to the provider regression**

In the same test, add this fixture immediately after `line_psin`:

```python
    state.variables.Ai_ez = np.ones((6, 7, 2, 2, 2), dtype=float)
    state.var_meta["Ai_ez"] = SimpleNamespace(
        active=True,
        dims=("xt", "yt", "zt", "tensor1", "tensor2"),
    )
```

After `assert "line_psin" not in frame.variables`, add:

```python
    assert "Ai_ez" not in frame.variables
```

- [ ] **Step 4b: Run the unavailable-coordinate regression and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: FAIL during coordinate construction with `ValueError: Unknown Veros output variable 'tensor2'.`

- [ ] **Step 4c: Filter candidates whose effective coordinates cannot be constructed**

Add this private helper immediately before `_active_output_variable_names`:

```python
def _coordinate_dimension_is_extractable(veros_state: Any, dim: str) -> bool:
    if dim == _TIMESTEP_DIM:
        return True
    variable = veros_variables.VARIABLES.get(dim)
    return bool(
        variable is not None
        and _resolved_dims(variable, veros_state.settings, dim) == (dim,)
        and hasattr(veros_state.variables, dim)
    )
```

Extend the return condition to:

```python
    return tuple(
        name
        for name, dims in dimensions_by_name.items()
        if name not in coordinate_names
        and len(set(dims)) == len(dims)
        and all(
            _coordinate_dimension_is_extractable(veros_state, dim) for dim in dims
        )
    )
```

- [ ] **Step 4d: Run the complete provider regression and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: PASS. The existing supported tuple remains unchanged, `sss_clim`, `line_psin`, and `Ai_ez` are absent, and the setup-local dimension resolver remains uncalled.

- [ ] **Step 4e: Add the reserved native time collision to the provider regression**

In the same test, add this fixture immediately after `Ai_ez`:

```python
    state.variables.time = np.asarray(259_200.0, dtype=np.float32)
    state.var_meta["time"] = SimpleNamespace(active=True, dims=None)
```

After `assert "Ai_ez" not in frame.variables`, add:

```python
    assert "time" not in frame.variables
    assert "time" in frame.coordinates
```

- [ ] **Step 4f: Run the reserved-name regression and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: FAIL during frame construction with `ValueError: OutputFrame name 'time' is both a variable and coordinate`.

- [ ] **Step 4g: Reserve the adapter-owned time coordinate name**

Change coordinate-name construction to:

```python
    coordinate_names = {VEROS_TIME_DIM} | {
        dim for dims in dimensions_by_name.values() for dim in dims
    }
```

- [ ] **Step 4h: Run the complete provider regression and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
```

Expected: PASS. Supported variables remain ordered; `sss_clim`, `line_psin`, `Ai_ez`, and native `time` are absent from variables; the adapter-owned `time` coordinate remains present.

- [ ] **Step 5: Run focused Veros output coverage**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py -q -k "veros and output"
```

Expected: all selected tests pass without new warnings or errors.

- [ ] **Step 6: Run a bounded reproduction through the actual example**

Run the example source with only its clock-step expression replaced in memory, execute it inside a temporary output directory, and leave the tracked example unchanged:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c 'from pathlib import Path; import os, tempfile; source_path=Path("examples/run_jcm_with_veros.py").resolve(); source=source_path.read_text().replace("steps=365 * 100 - 2", "steps=1"); temporary=tempfile.TemporaryDirectory(); os.chdir(temporary.name); exec(compile(source, str(source_path), "exec"), {"__name__": "__main__"}); print("one-step example passed")'
```

Expected: exit 0 after one coupled step, print `one-step example passed`, and produce no `OutputVariable.dims must be unique` error. This command requires access to the existing Veros asset cache under `~/.veros`.

- [ ] **Step 7: Format and run focused static checks**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py
/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py
git diff --check
```

Expected: Black leaves both files formatted, flake8 reports `0`, mypy reports no issues in two source files, and the whitespace check exits 0.

- [ ] **Step 8: Run repository regression suites**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/ -q
```

Expected: both suites pass completely with only the repository's previously recorded third-party warnings.

- [ ] **Step 9: Update the progress log with observed evidence**

After confirming the expected counts in Steps 5, 6, and 8, add this dated entry at the top of `PROGRESS.md` under `## Current Status`. If repository collection changes independently before execution, update only the numerical counts to the observed values:

```markdown
- Veros native-variable representability regression fixed locally on
  2026-07-14. The
  native provider now resolves each supported active variable's dimensions
  once, excludes `line_psin`, whose repeated `("isle", "isle")` axes cannot
  satisfy the shared `OutputVariable` contract, and excludes four `Ai_*`
  variables whose `tensor1` and `tensor2` coordinates are unavailable, and
  reserves `time` for the adapter-owned coordinate. The
  focused provider regression and Veros output selection pass 6/6; a bounded one-step
  `run_jcm_with_veros.py` execution passes; the fast suite passes 480 tests with
  586 deselected, and the full suite passes all 1066 tests. Black, flake8, mypy,
  and whitespace checks pass.
```

- [ ] **Step 10: Re-run final focused verification after the documentation edit**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/pytest tests/test_external_components_coverage.py::test_veros_output_provider_exposes_active_native_variable_universe -q
git diff --check
```

Expected: one passing test and a clean whitespace check.

- [ ] **Step 11: Review scope and commit the bug fix**

Run:

```bash
git diff -- vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py PROGRESS.md
git status --short
git add vercor/setups/_external/veros_output.py tests/test_external_components_coverage.py PROGRESS.md
git commit -m "fix: filter unrepresentable Veros output"
```

Expected: the implementation commit contains only the provider-universe filter, its regression coverage, and the progress entry. The design and plan remain in their preceding documentation commits.
