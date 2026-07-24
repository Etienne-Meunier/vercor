# Bundled Model Period Output Design

> Revised by
> `docs/superpowers/specs/2026-07-23-bundled-output-configurability-design.md`,
> which adds per-component `OutputSpec` factory parameters while preserving the
> default behavior described here.

## Goal

Enable step-cadence period output for VerCOR's bundled slab and data model
factories when a caller supplies an output-enabled `OutputTarget`. Period files
contain only each component's declared output fields. Existing external/native
model output, third-party component defaults, and output-free execution remain
unchanged.

## Current behavior

The private output session already provides a generic `_RuntimeFieldProvider`.
When a component declares `OutputSpec(period=PeriodOutput(...))`, that provider
samples the component's declared `ComponentSpec.outputs`, and the session owns
accumulation, cadence boundaries, NetCDF dimensions, filenames, host transfer,
and writes.

Bundled external components may install native providers and snapshot writers.
Before this change, bundled slab factories and the shared data-component factory
left `ComponentSpec.output` at `OutputSpec()`, whose `period` is `None`.
Therefore a bare `OutputTarget` wrote their final runtime fields but did not
schedule period-average files.

## Scope

This change applies to bundled VerCOR slab components and components built by
the shared bundled data-component factory. It does not change the default
`ComponentSpec` contract, third-party or plugin components, public factory
signatures, external/native providers, snapshot behavior, or final-field
output.

The default bundled period policy is:

- cadence: `step`;
- provider: `None`, selecting VerCOR's existing runtime-field provider;
- variables: empty selection, which means every declared component output;
- activation: only when `Coupler.run(output=OutputTarget(...))` enables period
  output.

`Coupler.run(output=None)` and an all-disabled output target continue to avoid
provider sampling, host transfer, directory creation, and file I/O.

## Architecture

A private setup helper constructs the standard bundled policy using the public
`OutputSpec` and `PeriodOutput` classes:

```python
def step_period_output() -> OutputSpec:
    return OutputSpec(period=PeriodOutput(frequency="step"))
```

The four slab factories pass this policy to their existing `ComponentSpec`
instances. The shared `time_interpolated_data_component` helper passes the same
policy to its `ComponentSpec`, so ERA5, ERA-Interim, and future bundled data
factories using that helper receive consistent behavior without duplicating
declarations. The direct JCM land data factory reuses the same policy in its
existing `ComponentSpec`.

The component classes, output coordinator, provider implementation, NetCDF
writer, and model kernels do not change. This preserves the existing ownership
boundary: bundled setup factories choose a default output declaration, while
the core output session performs all runtime output work.

## Data flow

1. A bundled slab or data factory constructs its component with the shared
   step-period `OutputSpec`.
2. The caller supplies an enabled `OutputTarget` to `Coupler.run`.
3. The output plan sees the declared `PeriodOutput` and selects the existing
   `_RuntimeFieldProvider` because no custom provider is configured.
4. After each model step, the provider samples only names in
   `ComponentSpec.outputs` from the public component-state view.
5. The existing session writes the step-period NetCDF file at its precomputed
   boundary.

## Error handling

No new error-handling path is required. Existing component preparation rejects
invalid output declarations. The output session continues to wrap provider,
selection, schema, and write failures in component-scoped diagnostics. Unknown
variable selection is not introduced because the default empty selection uses
all provider variables.

## Testing

Behavior-first tests will establish the change through a red/green cycle:

- every bundled slab factory declares step-period output through an
  `OutputSpec` with no custom provider;
- a representative slab component run with `OutputTarget` writes one period
  file per step containing declared outputs and excluding input-only fields;
- the shared data-component factory declares the same step-period policy;
- the direct JCM land data factory declares the same step-period policy;
- a representative data component run writes declared outputs and excludes
  input-only fields;
- existing output tests continue to prove `output=None` performs no I/O and
  third-party components remain opt-in;
- focused tests, Black, strict flake8, mypy, compileall, fast and full pytest,
  coverage, and whitespace checks run before the implementation commit.

## Documentation and compatibility

`DESIGN.md` remains architecturally accurate: output is still run-level opt-in,
and `ComponentSpec` remains the owner of component output policy. `PROGRESS.md`
will record the completed behavior and verification evidence. User-facing
output documentation will state that bundled slab and data factories declare
step-period output, while custom components must continue to configure their
own `OutputSpec`.

This is a backward-compatible bundled-factory default change. Supplying
`output=None` preserves numerical and differentiation behavior, and callers
can disable period files with `OutputTarget(..., write_period=False)`.
