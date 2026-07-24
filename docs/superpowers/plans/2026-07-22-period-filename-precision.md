# Period Filename Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `day`, `month`, and `year` period-average filenames use matching calendar precision while preserving existing `step` filenames and exact NetCDF window-start metadata.

**Architecture:** Keep filename ownership in `vercor/output/_session.py`. Add one private cadence-aware formatter and call it from the existing boundary precomputation; leave cadence detection, accumulation, NetCDF time coordinates, and collision allocation unchanged.

**Tech Stack:** Python 3.13, JAX, h5netcdf, pytest, Black, flake8, mypy

## Global Constraints

- `day` filenames use `YYYY-MM-DD`, `month` filenames use `YYYY-MM`, and `year` filenames use `YYYY`.
- `step` timestamp/step/schema collision filenames remain byte-for-byte compatible.
- NetCDF time metadata retains the exact actual averaging-window start.
- Follow TDD: observe the filename expectations fail before editing production code.
- Do not add public API, dependencies, configurable templates, or new cadences.
- Update `PROGRESS.md` and run the full test suite before the implementation commit.

---

### Task 1: Implement cadence-aware period filename precision

**Files:**

- Modify: `tests/test_runtime_run.py:398-475`
- Modify: `tests/test_runtime_run.py:536-564`
- Modify: `vercor/output/_session.py:507-573`
- Modify: `docs/superpowers/specs/2026-07-22-period-filename-precision-design.md:4`
- Modify: `PROGRESS.md:9`
- Create: `docs/superpowers/plans/2026-07-22-period-filename-precision.md`

**Interfaces:**

- Consumes: `PeriodOutput.frequency` with validated values `"step"`, `"day"`, `"month"`, or `"year"`; `_Time` as `datetime | ModelDateTime`.
- Produces: private `_period_filename_date(period_start: _Time, period: PeriodOutput) -> str`; period basenames with cadence-specific precision.

- [ ] **Step 1: Add failing integration expectations for cadence precision**

In `tests/test_runtime_run.py`, add this test before the monthly calendar-identity parametrization:

```python
@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("frequency", "start", "dt_seconds", "expected_filename"),
    [
        ("day", datetime(2000, 2, 1), 86_400.0, "model.averages.2000-02-01.nc"),
        ("month", datetime(2000, 2, 1), 29 * 86_400.0, "model.averages.2000-02.nc"),
        ("year", datetime(2000, 2, 1), 365 * 86_400.0, "model.averages.2000.nc"),
    ],
)
def test_period_filename_precision_matches_frequency(
    frequency: str,
    start: datetime,
    dt_seconds: float,
    expected_filename: str,
    tmp_path: Path,
) -> None:
    _make_period_output_coupler(
        execution="jax",
        frequency=frequency,
        start=start,
        dt_seconds=dt_seconds,
        steps=1,
    ).run(output=_period_target(tmp_path))

    paths = tuple(tmp_path.glob("model.averages.*.nc"))
    assert [path.name for path in paths] == [expected_filename]
    with h5netcdf.File(paths[0], "r") as dataset:
        assert dataset.variables["time"].attrs["isoformat"] == start.isoformat()
```

Update the existing monthly calendar-identity parameter from `expected_starts`
to `expected_filenames`, using these values while leaving
`expected_isoformats` unchanged:

```python
(
    datetime(2000, 1, 1),
    "gregorian",
    31,
    ("2000-01",),
    ("2000-01-01T00:00:00",),
),
(
    datetime(2001, 1, 3),
    "noleap",
    57,
    ("2001-01", "2001-02"),
    ("2001-01-03T00:00:00.000000", "2001-02-01T00:00:00.000000"),
),
(
    datetime(2001, 2, 5),
    "noleap",
    24,
    ("2001-02",),
    ("2001-02-05T00:00:00.000000",),
),
(
    datetime(2001, 2, 5),
    "360_day",
    26,
    ("2001-02",),
    ("2001-02-05T00:00:00.000000",),
),
```

Use the exact expected filenames directly:

```python
expected_filenames: tuple[str, ...],
```

```python
assert [path.name for path in paths] == [
    f"model.averages.{date_token}.nc" for date_token in expected_filenames
]
```

Finally, change the mixed-cadence monthly expectation while preserving all
three daily expectations:

```python
assert [path.name for path in monthly_paths] == ["monthly.averages.2000-01.nc"]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n scipy pytest tests/test_runtime_run.py::test_period_filename_precision_matches_frequency tests/test_runtime_run.py::test_monthly_period_identity_uses_actual_window_start tests/test_runtime_run.py::test_mixed_component_period_frequencies_coexist tests/test_runtime_run.py::test_subdaily_step_output_keeps_one_file_per_step -q
```

Expected: 7 FAIL and 2 PASS. The monthly/yearly reduced-precision expectations
fail because `_output_boundaries` still always formats `%Y-%m-%d`; the daily
and step cases remain passing evidence.

- [ ] **Step 3: Add the minimal private formatter and use it for basenames**

In `vercor/output/_session.py`, replace the inline `strftime` expression in
`_output_boundaries`:

```python
bases = tuple(
    f"{_safe_token(schemas[index].component.name)}.averages."
    f"{_period_filename_date(period_start, schemas[index].period)}.nc"
    for index, period_start in zip(due, period_starts, strict=True)
)
```

Add this helper immediately before `_safe_token`:

```python
def _period_filename_date(period_start: _Time, period: PeriodOutput) -> str:
    """Return a filename date token at the configured cadence's precision."""

    format_by_frequency = {
        "step": "%Y-%m-%d",
        "day": "%Y-%m-%d",
        "month": "%Y-%m",
        "year": "%Y",
    }
    return period_start.strftime(format_by_frequency[period.frequency])
```

Do not alter the `counts`, timestamp/step/schema discriminator, `used` set, or
record/collision loop below basename creation.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 again.

Expected: all selected tests PASS, including unchanged daily and subdaily-step
expectations and exact NetCDF `isoformat` metadata.

- [ ] **Step 5: Run the complete output regression focus**

Run:

```bash
conda run -n scipy pytest tests/test_runtime_run.py tests/test_v0_4_output_providers.py tests/test_native_period_output.py -q --fast
```

Expected: PASS with no filename, collision, accumulation, or writer
regressions.

- [ ] **Step 6: Format and run static validation**

Run:

```bash
conda run -n scipy black vercor examples tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
conda run -n scipy python -m compileall -q vercor examples tests
```

Expected: Black completes without unformatted files; flake8 reports zero
violations; mypy succeeds; compileall exits zero.

- [ ] **Step 7: Run fast, full, and coverage test gates**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast
conda run -n scipy pytest tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
conda run -n scipy pytest --cov=vercor tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
```

Expected: all three commands PASS. Coverage remains at or above the project's
90% branch release gate.

- [ ] **Step 8: Record verified progress and finalize spec status**

Change the design status to:

```markdown
**Status:** Approved and implemented
```

Add a dated first bullet under `PROGRESS.md`'s `## Current Status`. State that
daily, monthly, and yearly means now use `YYYY-MM-DD`, `YYYY-MM`, and `YYYY`,
respectively; step filenames and exact NetCDF window-start metadata remain
unchanged. Include the exact RED, GREEN, output-focus, fast, full, and coverage
pass counts observed in Steps 2, 4, 5, and 7, plus the successful static and
whitespace gates. Do not estimate counts before their commands complete.

- [ ] **Step 9: Verify the final diff and commit the implementation**

Run:

```bash
git diff --check
git status --short
git diff -- vercor/output/_session.py tests/test_runtime_run.py PROGRESS.md docs/superpowers/specs/2026-07-22-period-filename-precision-design.md
```

Confirm that only the five planned files changed and every approved requirement
is represented. Then stage and inspect only those files:

```bash
git add vercor/output/_session.py tests/test_runtime_run.py PROGRESS.md docs/superpowers/specs/2026-07-22-period-filename-precision-design.md docs/superpowers/plans/2026-07-22-period-filename-precision.md
git diff --cached --check
git diff --cached --stat
git commit -m "feat: reflect averaging cadence in filenames"
```

Expected: one implementation commit containing the helper, regression tests,
approved-plan documentation, final design status, and verified progress entry.
