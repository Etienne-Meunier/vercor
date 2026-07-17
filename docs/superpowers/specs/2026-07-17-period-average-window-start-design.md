# Period-Average Window-Start Timestamp Design

**Date:** 2026-07-17
**Status:** Approved for implementation

## Purpose

Period-average files must identify the averaging window whose values they
contain. A monthly integration beginning in January must therefore write its
first January average with a January timestamp instead of the first instant in
February. If an integration begins partway through a period, the label must use
the actual integration-window start rather than the nominal calendar-period
start.

This rule applies consistently to `step`, `day`, `month`, and `year` output and
to Gregorian, no-leap, and 360-day clocks.

## Root cause

The output coordinator already accumulates the correct sequence of post-step
samples and closes each accumulator at the correct cadence boundary. The bug is
limited to represented-time ownership: `_output_boundaries` currently assigns
`time + dt`, the first instant after the completed window, to both the filename
and NetCDF time coordinate. Consequently, a January monthly mean is written as
`*.averages.2000-02-01.nc`.

Changing cadence detection or component sampling would alter correct numerical
behavior and is outside the fix.

## Timestamp semantics

Each output schema owns an independent current window-start timestamp:

- Before the first step, its window starts at the first clock-step time.
- When that schema's cadence becomes due, the completed file uses the stored
  window start.
- After the file boundary, that schema's next window starts at the completed
  boundary time, `time + dt`.
- Schemas that are not due retain their existing window start.

Examples for monthly output are:

- start `2000-01-01` -> first file `*.averages.2000-01-01.nc`;
- start `2000-01-03` -> first file `*.averages.2000-01-03.nc`, then
  `*.averages.2000-02-01.nc`;
- start `2000-02-05` -> first file `*.averages.2000-02-05.nc`.

The same actual-window-start rule applies to daily, yearly, and per-step
windows. Provider `OutputContext.time` remains the post-step represented-state
time because providers sample post-step state; it is not the period label.

## Architecture and data flow

`vercor/output/_session.py` remains the single owner of precomputed cadence
boundaries, filenames, time coordinates, and accumulator resets. No public API
or component setup changes are needed.

During `_output_boundaries` precomputation, one mutable local start value is
maintained per immutable output schema. Every generated `_OutputBoundary`
carries start timestamps aligned with its due schema indices and allocated
filenames. `write_output_boundary` uses each aligned start timestamp for that
file's NetCDF time coordinate. Filename allocation uses the same start
timestamp, including its existing collision discriminator when multiple
records would otherwise share a basename.

The boundary-closing timestamp remains available only for advancing due
schemas to their next window. Mixed component cadences therefore remain
independent: at a shared boundary a daily file can represent a one-day window
while a monthly file represents the longer window that began earlier.

There are no new modules or dependencies, so `DEPENDENCIES.md` does not change.

## Error handling and invariants

The existing output validation and exception wrapping remain unchanged. The
implementation preserves these invariants:

- due schema indices, start timestamps, and filenames have identical ordering
  and lengths;
- a schema's next start advances only when that schema is written;
- zero-step and incomplete-period runs still write no period file;
- filename safety and collision handling remain deterministic;
- output remains opt-in and outside differentiated or outer-jitted workflows;
- accumulation values, counts, cadence detection, and provider sampling times
  do not change.

## Testing strategy

Regression tests will first demonstrate the current off-by-one-period label,
then cover:

- a full January monthly window labeled `2000-01-01`;
- a partial January window labeled with its actual start date;
- an integration beginning partway through February;
- the next complete window starting at the prior boundary;
- consistency between each filename and its NetCDF time coordinate;
- independent starts for mixed output frequencies;
- Gregorian, no-leap, and 360-day calendar values;
- unchanged numeric means, boundary counts, incomplete-period behavior,
  provider context times, filename safety, and collision freedom.

Existing expectations that encode end-of-window filenames or time coordinates
will be updated only where the approved represented-time contract changes.
Focused output tests will run during the red-green cycle. Final validation will
include Black, strict flake8, mypy, compileall, fast and full pytest, coverage,
and whitespace checks before the implementation commit.

## Scope boundaries

This task does not add configurable timestamp conventions, period-bound
variables, weighted temporal integration, partial-period flushing at the end of
a run, new output frequencies, or changes to model stepping. Those features are
not required to correct the existing period identity and would violate YAGNI.
