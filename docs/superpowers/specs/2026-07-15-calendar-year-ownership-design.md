# Calendar Year Ownership Design

## Goal

Make `vercor.calendar` the single owner of model-year duration while preserving
the existing public clock configuration and runtime execution behavior. Remove
the independently configurable `RuntimeOptions.model_year_seconds` value so a
run cannot combine contradictory clock and forcing calendars.

## Scope

The existing `Clock(calendar="gregorian" | "noleap" | "360_day")` interface
remains the sole run-level calendar selection. Clock iteration, daily forcing
lookup, host and JAX execution, differentiation, and output timestamp behavior
remain otherwise unchanged.

The ownership correction intentionally removes the public
`RuntimeOptions.model_year_seconds` constructor parameter and attribute. No
compatibility alias or deprecation shim will retain the duplicate owner. Tests
that used arbitrary short model years to simplify monthly interpolation will
instead use calendar-derived durations and suitable timestamps.

## Calendar API

`vercor.calendar` will define a string-compatible `YearType` enum with these
members:

- `YearType.GREGORIAN_LEAP`
- `YearType.GREGORIAN_NO_LEAP`
- `YearType.DAY_360`

`DAY_360` is used because Python identifiers cannot begin with a digit. Enum
values will retain the existing forcing-policy spellings (`"leap"`,
`"noleap"`, and `"360"`) so existing string-based forcing calls keep their
meaning.

The calendar module will expose one canonical helper that returns seconds per
year for a `YearType`: 366 days for `GREGORIAN_LEAP`, 365 days for
`GREGORIAN_NO_LEAP`, and 360 days for `DAY_360`. A second calendar helper will
resolve the applicable `YearType` from the existing clock calendar and a
simulated year:

- `calendar="gregorian"` resolves from Gregorian leap-year rules for the
  timestamp's year.
- `calendar="noleap"` always resolves to `GREGORIAN_NO_LEAP`.
- `calendar="360_day"` always resolves to `DAY_360`.

Resolution is per timestamp, not once at run construction, because a run may
cross between leap and non-leap Gregorian years.

## Runtime Data Flow

`RuntimeOptions` will retain only dtype, backend, workflow, and topology
policy. The numeric validation and normalization dedicated to
`model_year_seconds` will be deleted.

Private runtime time-metadata construction will derive `YearType` from each
clock timestamp and request its model-year duration from `vercor.calendar`.
Monthly interpolation cycle length and spacing will therefore have the same
calendar owner as daily forcing selection. Initial-state priming and full-run
metadata will use this same path, avoiding separate duration parameters.

`vercor.forcing_index` will consume the calendar-owned `YearType` rather than
owning a duplicate year-type literal. Its public functions will continue to
accept the existing string values through the string-compatible enum boundary.

## Error Handling and Compatibility

Invalid public `Clock.calendar` values continue to fail at clock construction
with the existing error. Calendar helpers will reject unsupported year types
at their own boundary with a focused `TypeError` or `ValueError`, as appropriate
for the input type.

This is an intentional 0.4 alpha signature correction. Public signature and
module-export contracts, architecture documentation, and migration guidance
will be updated to show that `RuntimeOptions` no longer owns a model-year
policy and that `vercor.calendar` owns `YearType` and year duration.

No unrelated clock, forcing, runtime, component, output, or plugin API will be
redesigned.

## Testing

Development will follow red-green-refactor:

1. Add calendar unit tests for enum values, leap/no-leap/360-day duration, and
   timestamp-sensitive Gregorian resolution.
2. Add public-contract tests proving `RuntimeOptions` no longer accepts or
   exposes `model_year_seconds` and calendar exports own the replacement API.
3. Add runtime tests proving monthly selection uses the calendar-derived
   duration in Gregorian leap, Gregorian non-leap, no-leap, and 360-day runs.
4. Retain daily forcing, JAX, host, gradient, output, plugin, and distribution
   coverage while replacing tests that configured arbitrary model-year
   seconds.

Before the implementation commit, run Black, strict flake8, mypy, compileall,
the fast and full pytest suites, and `git diff --check`. Update `PROGRESS.md`,
`DESIGN.md`, `DEPENDENCIES.md`, API documentation, and canonical public
signature evidence in the same implementation unit.
