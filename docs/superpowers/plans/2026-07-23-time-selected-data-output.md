# Time-Selected Data Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make period output average the exact linear-monthly or indexed-daily forcing slice exported during every coupling step while preserving current-field slab output.

**Architecture:** Extract the existing transfer-policy selection into one pure private helper shared by exchange sending and default output sampling. Pass the already precomputed run-level `RuntimeStepInfo` into the output plan, select by `OutputContext.step`, and leave custom providers and accumulation unchanged.

**Tech Stack:** Python 3.12+, JAX, NumPy, h5netcdf, pytest, Black, flake8, mypy, coverage.py, flit/build.

## Global Constraints

- Follow `AGENTS.md`, `DESIGN.md`, and `PROGRESS.md`.
- Write every behavior regression before production changes.
- Keep physics values JAX-traced and precision policy static.
- Preserve pure JAX-compatible functions, frozen PyTrees, public signatures, and output-free differentiation.
- Do not change forcing interpolation formulas, calendar indexing, cadence boundaries, accumulator arithmetic, or custom provider semantics.
- Do not add fallback sampling of raw forcing arrays when time selection is active.
- Do not add a registry, new public API, serialized-state schema, or dependency.
- Run the full test suite before every implementation commit.
- Do not tag, push, publish, upload, or create a release.

## File Structure

- `vercor/_runtime/field_transfer.py`: sole pure owner of `current`, `linear`, and `daily` runtime field selection; exchange sending delegates to it.
- `vercor/output/_session.py`: default provider selects declared state fields with the shared helper and exact step metadata before accumulation.
- `vercor/_runtime/execution.py`: passes the run's already-built `RuntimeStepInfo` into output planning.
- `tests/test_runtime_state.py`: focused shared-selector and unchanged JIT/gradient exchange-send coverage.
- `tests/test_bundled_period_output.py`: end-to-end linear, daily, and slab-current period-output regressions.
- `tests/test_final_review_boundaries.py`: direct private output-plan construction remains explicit about runtime metadata.
- `DESIGN.md`: records policy-consistent default-provider sampling.
- `DEPENDENCIES.md`: records shared selection ownership and the output-session dependency.
- `PROGRESS.md`: records root cause, corrected coverage, and exact verification evidence.

---

### Task 1: Extract One Shared Runtime Field Selector

**Files:**
- Modify: `tests/test_runtime_state.py:22-28, 910-985`
- Modify: `vercor/_runtime/field_transfer.py:1-82`

**Interfaces:**
- Consumes: `TransferPolicy.time_selection`, `RuntimeStepInfo`, and `RuntimeArray`.
- Produces:

```python
def select_runtime_field(
    field: RuntimeArray,
    transfer: TransferPolicy,
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    """Select one current, linearly interpolated, or daily runtime field."""
```

- [ ] **Step 1: Import and test the not-yet-implemented shared selector**

Change the private import in `tests/test_runtime_state.py` to:

```python
from vercor._runtime.field_transfer import (
    select_runtime_field,
    send_runtime_fields,
)
```

Add immediately before the existing monthly-send test:

```python
def test_shared_runtime_field_selector_applies_every_transfer_policy() -> None:
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [2]),
    )
    forcing = jnp.arange(4 * 2 * 2, dtype=jnp.float64).reshape((4, 2, 2))

    current = select_runtime_field(
        forcing,
        TransferPolicy("current"),
        step_info,
    )
    linear = select_runtime_field(
        forcing,
        TransferPolicy("linear"),
        step_info,
    )
    daily = select_runtime_field(
        forcing,
        TransferPolicy("daily"),
        step_info,
    )
    without_step_metadata = select_runtime_field(
        forcing,
        TransferPolicy("linear"),
        None,
    )

    assert current is forcing
    assert without_step_metadata is forcing
    assert_allclose_compact(
        linear,
        0.75 * np.asarray(forcing[0]) + 0.25 * np.asarray(forcing[1]),
    )
    assert_allclose_compact(daily, np.asarray(forcing[2]))
```

- [ ] **Step 2: Run the new test to prove the interface is absent**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_runtime_state.py::test_shared_runtime_field_selector_applies_every_transfer_policy -q --tb=short
```

Expected: collection fails because `select_runtime_field` cannot be imported.

- [ ] **Step 3: Implement the pure selector and delegate exchange sending to it**

In `vercor/_runtime/field_transfer.py`, expand the type-only import:

```python
if TYPE_CHECKING:
    from vercor.components.contracts import Component, TransferPolicy
```

Replace `_select_runtime_field_for_send` with:

```python
def select_runtime_field(
    field: RuntimeArray,
    transfer: "TransferPolicy",
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    """Select one current, linearly interpolated, or daily runtime field."""

    if step_info is None:
        return field

    time_selection = transfer.time_selection
    if time_selection == "linear":
        array = jnp.asarray(field)
        left = jnp.take(array, step_info.monthly_index_left, axis=0)
        right = jnp.take(array, step_info.monthly_index_right, axis=0)
        return (
            step_info.monthly_weight_left * left
            + step_info.monthly_weight_right * right
        )
    if time_selection == "daily":
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)
    return field
```

Update the `send_runtime_fields` comprehension to:

```python
{
    field_name: select_runtime_field(
        component_state.fields.get(field_name),
        component.spec.transfer,
        step_info,
    )
    for field_name in contract.sends
}
```

- [ ] **Step 4: Run selector, existing send, JIT, and gradient tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_runtime_state.py::test_shared_runtime_field_selector_applies_every_transfer_policy tests/test_runtime_state.py::test_runtime_send_applies_monthly_interpolation_under_jit_and_grad tests/test_runtime_state.py::test_runtime_send_applies_daily_time_slice_under_jit_and_grad -q --tb=short
```

Expected: `3 passed`.

- [ ] **Step 5: Run the fast and full regression suites before committing**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
```

Expected: both commands exit `0` with only documented third-party warnings.

- [ ] **Step 6: Commit the behavior-preserving selector extraction**

```bash
git add vercor/_runtime/field_transfer.py tests/test_runtime_state.py
git commit -m "refactor: share runtime field selection"
```

---

### Task 2: Sample Exact Time-Selected Fields for Period Output

**Files:**
- Modify: `tests/test_bundled_period_output.py:1-380`
- Modify: `tests/test_final_review_boundaries.py:1-205`
- Modify: `vercor/output/_session.py:9-31, 270-345, 415-445`
- Modify: `vercor/_runtime/execution.py:120-145`

**Interfaces:**
- Consumes:

```python
select_runtime_field(
    field: RuntimeArray,
    transfer: TransferPolicy,
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray
```

and the existing `_RuntimeExecutionData.step_infos: RuntimeStepInfo`.

- Produces:

```python
def build_output_plan(
    components: Mapping[str, _ComponentBinding],
    clock: Clock,
    target: OutputTarget,
    *,
    step_infos: RuntimeStepInfo,
    clock_steps: Sequence[_ClockStep] | None = None,
) -> _OutputPlan:
    """Normalize component providers and allocate all period filenames."""
```

The private `_RuntimeFieldProvider` stores `step_infos` and selects every
default-provider output with `context.step`.

- [ ] **Step 1: Add the linear monthly regression before changing output code**

Add `RuntimeOptions` and `build_runtime_step_info` imports to
`tests/test_bundled_period_output.py`:

```python
from vercor.runtime import RuntimeOptions
from vercor._runtime.time import build_runtime_step_info
```

Add after `test_data_period_file_contains_declared_outputs_only`:

```python
@pytest.mark.parametrize("backend", ["host", "jax"])
def test_linear_data_month_output_averages_exact_exported_slices(
    backend: str,
    tmp_path: Path,
) -> None:
    grid = make_test_grid(name=f"linear-output-{backend}")
    forcing = (
        jnp.arange(12.0, dtype=jnp.float64)[:, jnp.newaxis, jnp.newaxis]
        * jnp.ones((12, *grid.shape), dtype=jnp.float64)
    )
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": forcing},
        outputs=("temperature",),
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=("temperature",),
            )
        ),
    )
    clock = Clock(
        datetime(2001, 1, 1),
        dt_seconds=86_400.0,
        steps=59,
        calendar="noleap",
    )
    coupler = Coupler(
        clock,
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(backend=cast(Any, backend)),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    metadata = build_runtime_step_info(clock)
    record_values = np.arange(12.0)
    selected = (
        np.asarray(metadata.monthly_weight_left)
        * record_values[np.asarray(metadata.monthly_index_left)]
        + np.asarray(metadata.monthly_weight_right)
        * record_values[np.asarray(metadata.monthly_index_right)]
    )
    expected = (float(np.mean(selected[:31])), float(np.mean(selected[31:])))
    paths = sorted(tmp_path.glob("data.averages.*.nc"))
    actual = []
    for path in paths:
        with h5netcdf.File(path, "r") as dataset:
            values = np.asarray(dataset.variables["temperature"])
            assert values.shape == (1, *grid.shape)
            actual.append(float(np.mean(values)))

    assert [path.name for path in paths] == [
        "data.averages.2001-01.nc",
        "data.averages.2001-02.nc",
    ]
    np.testing.assert_allclose(actual, expected)
    assert not np.isclose(actual[0], actual[1])
```

- [ ] **Step 2: Add the daily monthly regression**

Add `DataComponent` and `TransferPolicy` to the existing component imports:

```python
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    TransferPolicy,
)
```

Add:

```python
def test_daily_data_month_output_averages_exact_exported_slices(
    tmp_path: Path,
) -> None:
    grid = make_test_grid(name="daily-output")
    forcing = (
        jnp.arange(365.0, dtype=jnp.float64)[:, jnp.newaxis, jnp.newaxis]
        * jnp.ones((365, *grid.shape), dtype=jnp.float64)
    )
    component = DataComponent(
        "DATA",
        grid,
        {"temperature": forcing},
        spec=ComponentSpec(
            outputs=("temperature",),
            transfer=TransferPolicy("daily"),
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=("temperature",),
                )
            ),
        ),
    )
    coupler = Coupler(
        Clock(
            datetime(2001, 1, 1),
            dt_seconds=86_400.0,
            steps=59,
            calendar="noleap",
        ),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(backend="jax"),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = sorted(tmp_path.glob("data.averages.*.nc"))
    actual = []
    for path in paths:
        with h5netcdf.File(path, "r") as dataset:
            values = np.asarray(dataset.variables["temperature"])
            assert values.shape == (1, *grid.shape)
            actual.append(float(np.mean(values)))

    assert [path.name for path in paths] == [
        "data.averages.2001-01.nc",
        "data.averages.2001-02.nc",
    ]
    np.testing.assert_allclose(actual, (15.0, 44.5))
```

In `test_slab_factory_accepts_keyword_only_output_spec`, add:

```python
assert component.spec.transfer == TransferPolicy("current")
```

This checks every bundled slab factory retains current-field sampling; the
existing monthly slab test remains the end-to-end current-mode value check.

- [ ] **Step 3: Run the new regressions and observe the raw forcing-array bug**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_bundled_period_output.py::test_linear_data_month_output_averages_exact_exported_slices tests/test_bundled_period_output.py::test_daily_data_month_output_averages_exact_exported_slices -q --tb=short
```

Expected: all selected cases fail because output variables have shapes
`(1, 12, *grid.shape)` or `(1, 365, *grid.shape)` and repeated raw records.

- [ ] **Step 4: Give the default provider exact run-level step metadata**

In `vercor/output/_session.py`, import `cast`, `select_runtime_field`, and
`RuntimeStepInfo`:

```python
from typing import TYPE_CHECKING, Any, cast

from vercor._runtime.field_transfer import select_runtime_field
from vercor._runtime.time import RuntimeStepInfo
```

Replace `_RuntimeFieldProvider` with:

```python
class _RuntimeFieldProvider:
    """Default provider exposing time-selected declared component outputs."""

    def __init__(
        self,
        component: "_ComponentBinding",
        step_infos: RuntimeStepInfo,
    ) -> None:
        self._component = component
        self._step_infos = step_infos

    def sample(self, context: OutputContext) -> OutputFrame:
        step_info = cast(
            RuntimeStepInfo,
            jax.tree_util.tree_map(
                lambda value: value[context.step],
                self._step_infos,
            ),
        )
        variables = {}
        for name in self._component.spec.outputs:
            values = select_runtime_field(
                context.state.field(name, scope="state"),
                self._component.spec.transfer,
                step_info,
            )
            variables[name] = OutputVariable(
                grid_field_dims(
                    name,
                    tuple(values.shape),
                    self._component.grid.shape,
                ),
                values,
                {"component": self._component.name, "field_name": name},
            )
        return OutputFrame(
            variables,
            coordinates={
                "latitude": OutputVariable(("nlat",), self._component.grid.latitude),
                "longitude": OutputVariable(("nlon",), self._component.grid.longitude),
            },
        )
```

Add the required keyword to `build_output_plan`:

```python
def build_output_plan(
    components: Mapping[str, "_ComponentBinding"],
    clock: Clock,
    target: OutputTarget,
    *,
    step_infos: RuntimeStepInfo,
    clock_steps: Sequence[_ClockStep] | None = None,
) -> _OutputPlan:
```

Construct default providers with:

```python
provider = component.spec.output.provider or _RuntimeFieldProvider(
    component,
    step_infos,
)
```

In `vercor/_runtime/execution.py`, pass the metadata already owned by
`_RuntimeExecutionData`:

```python
output_plan = build_output_plan(
    context.dispatch_context.components,
    context.clock,
    context.output,
    step_infos=execution_data.step_infos,
    clock_steps=execution_data.clock_steps,
)
```

In `tests/test_final_review_boundaries.py`, import:

```python
from vercor._runtime.time import build_runtime_step_info
```

Update its direct private-plan construction to:

```python
plan = build_output_plan(
    prepared.components,
    prepared.clock,
    OutputTarget(tmp_path),
    step_infos=build_runtime_step_info(prepared.clock),
)
```

- [ ] **Step 5: Run time-selected data, slab-current, custom-provider, and backend tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_bundled_period_output.py tests/test_final_review_boundaries.py::test_prepared_binding_does_not_delegate_private_markers_and_uses_spec_output tests/test_runtime_run.py::test_period_output_values_and_cadence_are_backend_consistent tests/test_v0_4_output_providers.py tests/test_runtime_state.py::test_runtime_send_applies_monthly_interpolation_under_jit_and_grad tests/test_runtime_state.py::test_runtime_send_applies_daily_time_slice_under_jit_and_grad -q --tb=short
```

Expected: all selected tests pass; linear output passes for host and JAX,
daily output contains exact January/February means, and slab/custom providers
remain unchanged.

- [ ] **Step 6: Run the fast and full regression suites before committing**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
```

Expected: both commands exit `0` with only documented third-party warnings.

- [ ] **Step 7: Commit the output correction**

```bash
git add vercor/output/_session.py vercor/_runtime/execution.py tests/test_bundled_period_output.py tests/test_final_review_boundaries.py
git commit -m "fix: sample time-selected data output"
```

---

### Task 3: Document Ownership and Run Final Verification

**Files:**
- Modify: `DESIGN.md:145-170`
- Modify: `DEPENDENCIES.md:20-45`
- Modify: `PROGRESS.md:7-25`

**Interfaces:**
- Consumes: the completed shared selector and default-provider behavior.
- Produces: durable architecture and verification evidence; no new code
  interface.

- [ ] **Step 1: Update architecture documentation**

In `DESIGN.md` section 8, add:

```markdown
The default runtime-field provider applies the component's `TransferPolicy`
with the exact precomputed metadata for `OutputContext.step`. Period output
therefore samples the same `current`, linearly interpolated monthly, or indexed
daily field exported during that coupling step; internal forcing-record axes
are never emitted as physical output dimensions.
```

In `DEPENDENCIES.md`, update layer 9 to identify
`vercor/_runtime/field_transfer.py` as the shared transfer-policy selection
owner and update layer 14 to state that `vercor/output/_session.py` consumes it
with run-precomputed `RuntimeStepInfo`. Do not reorder modules because both
imports continue to point from layer 14 to earlier layers.

- [ ] **Step 2: Run formatting, linting, typing, and bytecode gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black exits `0`; flake8 reports `0`; mypy reports success; compileall
and `git diff --check` exit `0`.

- [ ] **Step 3: Run fast, full, gradient, and branch-coverage gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_gradients.py -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90 --tb=short
```

Expected: every command exits `0`; branch coverage remains at least `90%`.

- [ ] **Step 4: Build and run distribution/artifact gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build --outdir dist
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build --wheel --outdir dist tests/fixtures/public_plugin
VERCOR_ARTIFACT_DIR="$(pwd)/dist" /Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_setup_boundaries.py::test_veros_implementation_import_does_not_configure_runtime tests/test_setup_boundaries.py::test_veros_factory_configures_once_before_implementation_import tests/test_external_components_coverage.py::test_veros_initialize_spinup_follows_enabled_only -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_v0_4_workflow_execution.py::test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients tests/test_v0_4_workflow_execution.py::test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad tests/test_v0_4_output_providers.py::test_all_disabled_target_remains_jit_and_gradient_compatible -q --tb=short
```

Expected: builds succeed; executable wheel/sdist/plugin boundaries and optional
model/gradient nodes pass.

- [ ] **Step 5: Record exact evidence in `PROGRESS.md`**

Add a dated top entry containing:

- the raw-climatology sampling root cause;
- the shared-selector correction;
- focused, fast, full, gradient, and coverage test counts;
- branch coverage percentage;
- Black, flake8, mypy, compileall, distribution, artifact, optional-model, and
  whitespace results; and
- any unchanged documented third-party warnings.

Do not claim a gate that did not run or pass.

- [ ] **Step 6: Re-run documentation-sensitive checks after the progress edit**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_api_architecture_review.py tests/test_distribution_boundaries.py tests/test_final_review_boundaries.py -q --tb=short
git diff --check
git status --short
```

Expected: selected documentation/artifact tests pass, whitespace is clean, and
only the intended documentation files remain uncommitted.

- [ ] **Step 7: Commit final documentation**

```bash
git add DESIGN.md DEPENDENCIES.md PROGRESS.md
git commit -m "docs: record time-selected output verification"
```

- [ ] **Step 8: Confirm final repository state**

Run:

```bash
git status --short
git log -5 --oneline
```

Expected: the worktree is clean and the latest commits are the design, plan,
shared-selector refactor, output fix, and verification documentation commits
in execution order.
