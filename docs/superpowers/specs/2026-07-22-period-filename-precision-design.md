# Period Filename Precision Design

**Date:** 2026-07-22
**Status:** Approved and implemented

## Purpose

Period-average filenames must communicate the configured averaging cadence by
using matching calendar precision. The filename date token will be:

- `YYYY-MM-DD` for `frequency="day"`;
- `YYYY-MM` for `frequency="month"`; and
- `YYYY` for `frequency="year"`.

For example, a February 2000 atmosphere mean will be named
`atm.averages.2000-02.nc` at monthly cadence and
`atm.averages.2000.nc` at yearly cadence.

Per-step filenames are outside this change. Their existing timestamp, step,
schema, record, and collision discriminators must remain unchanged.

## Considered approaches

1. Add one private cadence-to-date-token helper in the output session
   (recommended). This keeps the policy centralized and independently testable
   without expanding the public API.
2. Put cadence conditionals directly inside `_output_boundaries`. This is a
   smaller textual change, but it mixes filename policy with boundary
   allocation and is less clear to test or reuse.
3. Introduce cadence-specific formatter objects. This would provide an
   extensible strategy boundary, but there are only four closed, validated
   cadence values, so the extra types would violate YAGNI.

## Design

`vercor/output/_session.py` remains the sole owner of period filename
allocation. A private helper will accept the averaging-window start and the
validated `PeriodOutput.frequency`, then return the calendar token at the
required precision. `_output_boundaries` will use that token when constructing
the basename for `day`, `month`, and `year` schemas.

The helper will preserve the existing `YYYY-MM-DD` base token for `step` so
the existing duplicate-basename detection continues to add the exact
`T%H%M%S.%f.step........schema....` suffix. The downstream collision allocator
will not change. If reduced month or year precision ever yields two otherwise
identical basenames, the existing deterministic record/collision mechanism
will still prevent overwrites.

Filename precision does not change represented-time ownership. The NetCDF time
coordinate and its `isoformat` attribute continue to store the exact start of
the actual averaging window, including its day and time for a partial month or
year. Only the filename's display precision changes.

No public API, output cadence, accumulation, component provider, calendar,
writer, module dependency, or file-content behavior changes. Consequently,
`DEPENDENCIES.md` does not change.

## Invariants and error handling

`PeriodOutput` already validates `frequency` as one of `step`, `day`, `month`,
or `year`, so the private formatter requires no new public validation or error
type. Existing component-scoped writer exceptions and collision-safe allocation
remain unchanged.

The implementation must preserve these invariants:

- daily files retain an exact calendar date;
- monthly files omit the day even when a run begins partway through a month;
- yearly files omit the month and day even when a run begins partway through a
  year;
- step filenames remain byte-for-byte compatible with current expectations;
- NetCDF time metadata retains the exact averaging-window start;
- safe component tokens and all uniqueness discriminators remain deterministic;
  and
- file counts, means, cadence boundaries, and incomplete-period behavior do not
  change.

## Testing strategy

Use TDD by changing or adding integration expectations before production code
and observing failures for monthly and yearly precision. Tests will cover:

- `day` producing `model.averages.2000-02-01.nc`;
- `month` producing `model.averages.2000-02.nc`;
- `year` producing `model.averages.2000.nc`;
- partial monthly and yearly windows using reduced filename precision while
  retaining the exact window start in NetCDF metadata;
- no-leap and 360-day calendar compatibility through existing monthly identity
  cases;
- unchanged subdaily `step` timestamp/step/schema filenames; and
- unchanged collision freedom and numerical output.

After the focused red-green cycle, run Black, strict flake8, mypy, compileall,
the fast and full pytest suites, and Git whitespace checks before committing
the implementation. Update `PROGRESS.md` with the date and verification counts.

## Scope boundaries

This task does not add configurable filename templates, new cadences, nominal
calendar-boundary timestamps, end-of-window labels, partial-period flushing,
or changes to output file contents. Those behaviors are not required by the
approved filename contract.
