# Period-Average Window-Start Timestamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label every period-average file and NetCDF time coordinate with the actual start of the averaging window instead of the first instant after that window.

**Architecture:** Keep cadence, accumulation, filename allocation, and NetCDF assembly in `vercor.output._session`. Precompute one mutable local window start per immutable output schema, copy aligned starts into each immutable output boundary, and advance only schemas written at that boundary. Provider sampling continues to use post-step represented-state time.

**Tech Stack:** Python 3.12+, frozen dataclasses, JAX, h5netcdf, pytest, mypy, Black, and flake8.

## Global Constraints

- Use actual averaging-window starts for `step`, `day`, `month`, and `year` output.
- Preserve Gregorian, no-leap, and 360-day calendar values without converting model datetimes to standard datetimes.
- Preserve cadence detection, accumulator values/counts, provider post-step context, incomplete-period behavior, and collision safety.
- Keep output opt-in, transform-safe when disabled, and owned by the existing private output coordinator.
- Add no public API, dependency, configurable timestamp policy, period bounds, weighted averaging, or final partial-period flush.
- Leave `DEPENDENCIES.md` unchanged because module dependencies do not change.
- Follow red-green-refactor: no production change may precede its failing regression test.
- Use the direct `scipy` environment interpreter at `/Users/romannuterman/miniforge3/envs/scipy/bin/python` if the Conda launcher panics.
- Run the complete unit suite before the implementation commit, as required by `AGENTS.md`.

---

## File Map

- `vercor/output/_session.py`: sole owner of per-schema window starts, immutable due boundaries, filenames, collision suffixes, and NetCDF period timestamps.
- `tests/test_runtime_run.py`: end-to-end partial/full monthly windows, supported calendars, mixed cadences, numerical values, and filename behavior.
- `tests/test_v0_4_output_providers.py`: unified provider contexts, period coordinates, payload providers, errors, and selection contracts under start-labeled files.
- `tests/test_distribution_boundaries.py`: installed public-plugin filename evidence under the corrected convention.
- `DESIGN.md`: durable output architecture statement naming averaging-window-start timestamps.
- `PROGRESS.md`: red-green and final validation evidence for the completed correction.

---

### Task 1: Correct period identity at the output coordinator

**Files:**

- Modify: `tests/test_runtime_run.py`
- Modify: `tests/test_v0_4_output_providers.py`
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `vercor/output/_session.py`
- Modify: `DESIGN.md`
- Modify: `PROGRESS.md`

**Interfaces:**

- Consumes: `Clock.iter() -> Iterator[tuple[int, datetime | ModelDateTime, timedelta]]` and existing per-schema `PeriodOutput.frequency` cadence decisions.
- Produces: private `_OutputBoundary.period_starts: tuple[_Time, ...]`, aligned with `due_schema_indices` and `output_filenames`.
- Preserves: `OutputContext.time == step_time + dt`, `_OutputAccumulator`, `should_write_period_output`, public output types, and all module manifests.

- [ ] **Step 1: Write the failing actual-window-start regression**

Extend `_make_period_output_coupler` in `tests/test_runtime_run.py` so calendar cases use the existing helper without duplicating coupler assembly:

```python
def _make_period_output_coupler(
    *,
    execution: str,
    frequency: str = "day",
    steps: int = 2,
    dt_seconds: float = 86_400.0,
    start: datetime = datetime(2000, 1, 1),
    calendar: Any = "gregorian",
    component: Component | None = None,
) -> Coupler:
    selected_component = component or _make_output_component(frequency=frequency)
    return Coupler(
        clock=Clock(
            start=start,
            dt_seconds=dt_seconds,
            steps=steps,
            calendar=calendar,
        ),
        components=(selected_component,),
        run_order=(selected_component.name,),
        runtime=RuntimeOptions(backend=cast(Any, execution)),
        log_level="WARNING",
    )
```

Add this parametrized end-to-end test beside the existing period cadence tests:

```python
@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("start", "calendar", "steps", "expected_starts"),
    [
        (datetime(2000, 1, 1), "gregorian", 31, ("2000-01-01",)),
        (
            datetime(2001, 1, 3),
            "noleap",
            57,
            ("2001-01-03", "2001-02-01"),
        ),
        (datetime(2001, 2, 5), "noleap", 24, ("2001-02-05",)),
        (datetime(2001, 2, 5), "360_day", 26, ("2001-02-05",)),
    ],
)
def test_monthly_period_identity_uses_actual_window_start(
    start: datetime,
    calendar: Any,
    steps: int,
    expected_starts: tuple[str, ...],
    tmp_path: Path,
) -> None:
    _make_period_output_coupler(
        execution="jax",
        frequency="month",
        start=start,
        calendar=calendar,
        steps=steps,
    ).run(output=_period_target(tmp_path))

    paths = sorted(tmp_path.glob("model.averages.*.nc"))
    assert [path.name for path in paths] == [
        f"model.averages.{window_start}.nc"
        for window_start in expected_starts
    ]
    for path, window_start in zip(paths, expected_starts, strict=True):
        with h5netcdf.File(path, "r") as dataset:
            assert (
                dataset.variables["time"].attrs["isoformat"]
                == f"{window_start}T00:00:00"
            )
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_runtime_run.py::test_monthly_period_identity_uses_actual_window_start \
  -q
```

Expected: four failures because current files and time coordinates use the
closing boundary (`2000-02-01`, `2001-02-01`, or `2001-03-01`) instead of the
stored actual window start.

- [ ] **Step 3: Carry aligned per-schema starts in immutable boundaries**

Replace the private boundary shape in `vercor/output/_session.py` with:

```python
@dataclass(frozen=True)
class _OutputBoundary:
    stop_step: int
    due_schema_indices: tuple[int, ...]
    period_starts: tuple[_Time, ...]
    output_filenames: tuple[str, ...]
```

Update `write_output_boundary` so every file consumes its aligned start:

```python
    for index, period_start, filename in zip(
        boundary.due_schema_indices,
        boundary.period_starts,
        boundary.output_filenames,
        strict=True,
    ):
```

Replace its time-coordinate argument with:

```python
            coordinates[frame.time_dimension] = time_coordinate_variable(
                period_start,
                time_dim=frame.time_dimension,
            )
```

Replace `_output_boundaries` with this precomputation, preserving the existing
safe-token and record-collision loop below it:

```python
def _output_boundaries(
    schemas: tuple[_OutputSchema, ...],
    clock: Clock,
    *,
    clock_steps: Sequence[_ClockStep] | None,
) -> tuple[_OutputBoundary, ...]:
    steps = tuple(clock.iter()) if clock_steps is None else tuple(clock_steps)
    if not steps:
        return ()

    window_starts = [steps[0][1] for _ in schemas]
    raw: list[
        tuple[int, tuple[int, ...], tuple[_Time, ...], tuple[str, ...]]
    ] = []
    for step, time, dt in steps:
        due = tuple(
            index
            for index, schema in enumerate(schemas)
            if should_write_period_output(schema.period, time=time, dt=dt)
        )
        if due:
            period_starts = tuple(window_starts[index] for index in due)
            bases = tuple(
                f"{_safe_token(schemas[index].component.name)}.averages."
                f"{period_start.strftime('%Y-%m-%d')}.nc"
                for index, period_start in zip(due, period_starts, strict=True)
            )
            raw.append((step + 1, due, period_starts, bases))
            next_window_start = time + dt
            for index in due:
                window_starts[index] = next_window_start

    counts = Counter(filename for *_, filenames in raw for filename in filenames)
    used: set[str] = set()
    result: list[_OutputBoundary] = []
    request = 0
    for stop, due, period_starts, filenames in raw:
        allocated: list[str] = []
        for schema_index, period_start, filename in zip(
            due,
            period_starts,
            filenames,
            strict=True,
        ):
            candidate = filename
            if counts[filename] > 1:
                stem = filename[:-3]
                candidate = (
                    f"{stem}T{period_start.hour:02d}{period_start.minute:02d}"
                    f"{period_start.second:02d}.{period_start.microsecond:06d}."
                    f"step{stop - 1:08d}.schema{schema_index:04d}.nc"
                )
            collision = 0
            while candidate in used:
                stem = candidate[:-3]
                candidate = (
                    f"{stem}.record{request:08d}.collision{collision:04d}.nc"
                )
                collision += 1
            used.add(candidate)
            allocated.append(candidate)
            request += 1
        result.append(
            _OutputBoundary(stop, due, period_starts, tuple(allocated))
        )
    return tuple(result)
```

- [ ] **Step 4: Run the regression and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_runtime_run.py::test_monthly_period_identity_uses_actual_window_start \
  -q
```

Expected: `4 passed` with matching filename and NetCDF `time.isoformat`
values for Gregorian, no-leap, and 360-day clocks.

- [ ] **Step 5: Align existing filename contracts with the approved semantics**

In `tests/test_runtime_run.py`, replace the per-step collision expectations
with:

```python
    assert [path.name for path in paths] == [
        "model.averages.2000-01-01T000000.000000.step00000000.schema0000.nc",
        "model.averages.2000-01-01T010000.000000.step00000001.schema0000.nc",
        "model.averages.2000-01-01T020000.000000.step00000002.schema0000.nc",
    ]
```

Replace the same-safe-token collision mapping with:

```python
    expected = {
        "shared-model.averages.2000-01-01T000000.000000.step00000000.schema0000.nc": 11.0,
        "shared-model.averages.2000-01-01T000000.000000.step00000000.schema0001.nc": 21.0,
    }
```

Strengthen `test_mixed_component_period_frequencies_coexist` with the aligned
independent starts:

```python
    assert sorted(path.name for path in tmp_path.glob("daily.averages.*.nc")) == [
        "daily.averages.2000-01-30.nc",
        "daily.averages.2000-01-31.nc",
        "daily.averages.2000-02-01.nc",
    ]
    monthly_paths = tuple(tmp_path.glob("monthly.averages.*.nc"))
    assert [path.name for path in monthly_paths] == [
        "monthly.averages.2000-01-30.nc"
    ]
```

In that file, replace each remaining one-day start `model.averages.2000-01-02.nc`
expectation with `model.averages.2000-01-01.nc`.

In `tests/test_v0_4_output_providers.py`, preserve provider context assertions
at January 2 and January 3, while changing period identity expectations as
follows:

```python
    path = tmp_path / "model.averages.2000-01-01.nc"
```

```python
    with h5netcdf.File(
        case_dir / "model.averages.2000-01-01.nc", "r"
    ) as dataset:
```

```python
        match=r"component 'model'.*model\.averages\.2000-01-01\.nc.*bad dataset",
```

```python
        match=r"component 'model'.*model\.averages\.2000-01-01\.nc",
```

```python
    assert [path.name for path in paths] == [
        "payload-model.averages.2000-01-01.nc",
        "payload-model.averages.2000-01-02.nc",
    ]
```

Replace the sample-dimension file with:

```python
    path = tmp_path / "model.averages.2000-01-01.nc"
```

In `tests/test_distribution_boundaries.py`, replace installed-plugin per-step
filenames with:

```python
        "jax.averages.2000-01-01T000000.000000.step00000000.schema0000.nc",
        "jax.averages.2000-01-01T000100.000000.step00000001.schema0000.nc",
```

- [ ] **Step 6: Run all focused output contracts**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_runtime_run.py \
  tests/test_v0_4_output_providers.py \
  tests/test_distribution_boundaries.py \
  tests/test_native_period_output.py \
  tests/test_output_datasets.py \
  tests/test_output_netcdf.py \
  -q --fast --tb=short
```

Expected: all selected tests pass; provider-context timestamps remain post-step,
numeric means remain unchanged, and filenames/time coordinates use starts.

- [ ] **Step 7: Synchronize durable architecture documentation**

In `DESIGN.md` section 8, replace:

```markdown
- collision-safe filenames and represented-state timestamps;
```

with:

```markdown
- collision-safe filenames and per-schema averaging-window-start timestamps;
```

Do not alter `DEPENDENCIES.md` because no dependency edge changes.

- [ ] **Step 8: Format, lint, type-check, and verify the complete repository**

Run these commands in order:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest --cov=vercor tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
git diff --check
```

Expected: Black completes without edits remaining; flake8 reports `0`; mypy,
compileall, fast pytest, full pytest, coverage pytest, and whitespace checks exit
zero. Third-party warnings already documented in `PROGRESS.md` may remain.

- [ ] **Step 9: Record exact completion evidence in the active progress log**

Add a dated first bullet under `PROGRESS.md` `## Current Status` stating that
period files and NetCDF time coordinates now use each schema's actual window
start; partial first periods, subsequent periods, mixed cadences, and all three
calendar policies are covered; provider times and numeric means are unchanged;
and listing the exact RED, focused, formatting, lint, typing, fast, full, and
coverage results observed in Steps 2, 4, 6, and 8.

Then run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_versioning_policy.py tests/test_api_architecture_review.py \
  -q --fast --tb=short
git diff --check
```

Expected: active documentation policy and whitespace checks pass after the log
update.

- [ ] **Step 10: Review and commit the implementation**

Inspect the exact scope:

```bash
git status --short
git diff --stat
git diff -- vercor/output/_session.py tests/test_runtime_run.py tests/test_v0_4_output_providers.py tests/test_distribution_boundaries.py DESIGN.md PROGRESS.md
```

Verify only the planned output behavior, its tests, and synchronized docs are
present. Stage and commit:

```bash
git add vercor/output/_session.py tests/test_runtime_run.py tests/test_v0_4_output_providers.py tests/test_distribution_boundaries.py DESIGN.md PROGRESS.md
git diff --cached --check
git commit -m "fix: label period averages by window start"
```
