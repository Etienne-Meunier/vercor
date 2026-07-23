# Bundled Output Configurability Design

> Revised by
> `docs/superpowers/specs/2026-07-23-bundled-output-default-alignment-design.md`,
> which changes the omitted slab/data declaration from step-period output to
> `OutputSpec()` while preserving the configurable factory APIs described here.

## Goal

Allow every bundled slab and data component factory to accept a complete
per-component `OutputSpec` while preserving the existing step-period output
default. This includes the four slab factories, ERA5 and ERA-Interim data
factories, direct JCM land construction, and the land component created by the
paired JCM land-atmosphere setup.

## Public API

Each public slab and data factory gains a final keyword-only parameter:

```python
output: OutputSpec | None = None
```

Existing positional parameters and their meanings remain unchanged. Omitting
`output`, or passing `None`, selects the current bundled default:

```python
OutputSpec(period=PeriodOutput(frequency="step"))
```

Passing an `OutputSpec` uses that object as the component's complete output
declaration. For example:

```python
component = make_era5_land(
    output=OutputSpec(
        period=PeriodOutput(
            frequency="month",
            variables=("land_surface_temperature",),
        )
    )
)
```

`OutputSpec()` explicitly disables period and snapshot output for that
component. A supplied provider or snapshot writer is preserved without merging
it with the bundled default.

The affected public factories are:

- `make_slab_atmosphere`;
- `make_slab_land`;
- `make_slab_ocean`;
- `make_slab_seaice`;
- `make_era5_atmosphere`;
- `make_era5_land`;
- `make_era5_ocean`;
- `make_erainterim_ocean`; and
- `make_jcm_land`.

`JCMLandAtmosphereConfig` gains:

```python
land_output: OutputSpec
```

Its default is the same step-period declaration. The paired setup passes
`config.land_output` to `make_jcm_land`, independently of
`config.atmosphere.output`.

## Architecture

The private setup output module remains the single owner of the bundled default.
It provides a resolver that returns a fresh step-period `OutputSpec` for
`None`, and otherwise returns the caller's supplied `OutputSpec` unchanged.
The resolver also validates that a non-`None` value is an `OutputSpec`, so every
factory fails consistently at its boundary.

Each slab factory resolves its `output` argument immediately before constructing
its existing `ComponentSpec`.

Each public time-interpolated data factory forwards `output` to
`time_interpolated_data_component`. That shared helper resolves the policy once
and places it in the existing `ComponentSpec`. Direct JCM land uses the same
resolver because it does not use the time-interpolated helper.

No provider, accumulator, cadence, writer, component kernel, or runtime
execution code changes. The existing generic runtime-field provider continues
to sample declared component outputs at the end of each coupler step. The
existing output session continues to select variables, accumulate samples, and
flush averages at the cadence specified by `PeriodOutput`.

## Data Flow

1. A caller omits `output` or supplies a complete `OutputSpec`.
2. The factory resolves the omitted value to the bundled step-period default or
   retains the supplied declaration.
3. The factory stores that declaration in `ComponentSpec.output`.
4. When `Coupler.run` receives an enabled `OutputTarget`, the output session
   samples the component after each coupler step.
5. `PeriodOutput.variables` selects the declared provider variables and
   `PeriodOutput.frequency` determines when the accumulated mean is written.

`OutputTarget` remains a run-level enablement and destination policy. It does
not override per-component cadence or variable selection.

## Validation and Errors

- Passing a non-`OutputSpec`, non-`None` value raises `TypeError` at factory
  construction with one consistent diagnostic.
- Invalid nested providers, periods, variable selections, and snapshot writers
  continue to fail in the existing `OutputSpec` and `PeriodOutput`
  constructors.
- Unknown selected variables continue to fail in the output session with
  component-scoped diagnostics.
- Component configuration remains immutable after coupler construction;
  callers configure output through the factory rather than assigning a new
  `component.spec`.

## Testing

Tests follow a red-green cycle and cover:

- every slab factory accepting a keyword-only full `OutputSpec`;
- every public ERA5 and ERA-Interim data factory forwarding the supplied
  declaration;
- the shared data helper preserving the exact supplied declaration;
- direct and paired JCM land forwarding;
- omitted output retaining step cadence;
- `OutputSpec()` disabling period output for one bundled component;
- a monthly slab run collecting coupler-step samples and writing the correct
  arithmetic mean at the month boundary;
- variable subsets containing only requested declared outputs;
- consistent invalid-output diagnostics;
- unchanged custom-component defaults and output-free execution; and
- updated public signature and installed-artifact contracts.

Focused tests run first, followed by Black, strict flake8, mypy, compileall,
fast pytest, full pytest, branch coverage, distribution probes, and Git
whitespace checks.

## Compatibility and Documentation

This is backward compatible for existing callers because all prior positional
arguments remain valid and omission retains step-period output. The new
parameter is keyword-only to avoid ambiguous calls and future positional API
constraints.

`README.md`, `DESIGN.md`, `DEPENDENCIES.md`, public signature fixtures, and
`PROGRESS.md` are updated with the final API and verification evidence.

The original bundled-period-output design remains a record of the initial
default behavior; this specification supersedes its statement that public
factory signatures do not change.
