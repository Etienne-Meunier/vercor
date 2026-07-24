# Time-Selected Data Output Design

## Goal

Period output for data components must average the exact forcing slices
exported during each coupling step. Linear monthly forcing and indexed daily
forcing must therefore produce time-varying period means with the same
selection semantics used by exchanges. Slab components must continue to sample
their evolving runtime fields unchanged.

## Root Cause

Runtime exchange export applies `TransferPolicy.time_selection` before placing
a data component field in its sent store. The generic output provider instead
samples the component's stored field directly. For monthly and daily data
components, that stored field is the complete forcing record array.

Consequently, every output sample contains the same full climatology. The
period accumulator correctly averages those identical samples, producing
identical monthly files that retain an erroneous forcing-record dimension.
This is a sampling divergence, not an accumulator or forcing-reader defect.

## Selected Architecture

Use one pure internal field-selection helper for both exchange export and
default output sampling.

The helper accepts a field, its static transfer policy, and the runtime step
metadata. It preserves the existing modes:

- `current` returns the stored field unchanged;
- `linear` interpolates the adjacent monthly records with the precomputed
  indices and weights; and
- `daily` selects the precomputed daily record.

The default runtime-field output provider receives the same precomputed
run-level step metadata used by execution. For each `OutputContext.step`, it
selects every declared output through the shared helper before constructing the
`OutputFrame`.

This keeps time selection in one owner, introduces no extra mutable state, and
does not depend on whether an output field participates in an exchange.
Sampling the sent store was rejected because declared outputs that are not
routed have no sent-store entry. Per-factory custom providers were rejected
because they would duplicate selection behavior and omit third-party data
components using the same public transfer policy.

## Data Flow

1. Runtime preparation computes `RuntimeStepInfo` arrays for every clock step.
2. During component execution, exchange export selects outgoing fields using
   the shared helper and the current step's metadata.
3. After that step completes, the core invokes the default output provider with
   `OutputContext.step`.
4. The provider selects declared fields using the same step-indexed metadata
   and helper.
5. The existing immutable sum/count accumulator receives only the selected
   physical slice.
6. At a period boundary, the existing writer emits the arithmetic mean and
   resets the completed window.

Using the zero-based step index is intentional: output represents the forcing
exported and used for that coupling step. The provider does not recompute
selection from the post-step output timestamp.

## Component Coverage

The correction applies generically to every component using the default output
provider:

- all ERA5 and ERA-Interim components created through the shared
  time-interpolated data helper use `linear`;
- direct JCM land uses `daily`;
- all bundled slab components use `current`; and
- custom components using `current`, `linear`, or `daily` receive the same
  policy-consistent behavior.

Custom `OutputProvider` implementations remain authoritative and are not
wrapped or altered.

## Validation and Errors

Existing component layout, runtime-state, output schema, variable-selection,
shape, dtype, and coordinate validation remains unchanged. Invalid forcing
indices and incompatible field ranks continue to fail at the existing JAX
selection boundary. Provider failures remain wrapped in the existing
component-scoped output diagnostic.

No fallback to a raw forcing array is permitted when time selection is active.

## Testing

Implementation follows a red-green cycle:

- reproduce two monthly files from a data component containing distinct
  monthly records;
- assert that the files contain distinct exact means and no forcing-record
  dimension;
- cover daily selection with distinct daily records and exact expected period
  means;
- verify `current` selection preserves slab output values and shapes;
- cover host and JAX execution where applicable;
- retain existing custom-provider, accumulator, cadence, schema, and variable
  subset behavior; and
- run focused tests before the complete repository verification gates.

Final verification includes Black, strict flake8, mypy, compileall, fast and
full pytest, branch coverage, distribution probes, installed artifact checks,
and `git diff --check`.

## Compatibility and Documentation

There is no public API or serialized-state schema change. Output files from
time-selected components intentionally change from repeated full forcing arrays
to the physical slices used by the coupled runtime. Their variable rank loses
the internal forcing-record axis, which is the corrected output contract.

`DESIGN.md`, `DEPENDENCIES.md`, and `PROGRESS.md` will record the shared
selection ownership and final verification evidence.

## Non-Goals

- Changing forcing interpolation formulas, calendar indexing, cadence
  boundaries, or accumulator arithmetic.
- Altering custom provider semantics.
- Adding time weighting within a coupling step.
- Refactoring unrelated output or component APIs.
