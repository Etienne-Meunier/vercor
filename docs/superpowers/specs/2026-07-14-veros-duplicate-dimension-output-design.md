# Veros Duplicate-Dimension Output Bug-Fix Design

## Problem

The bundled Veros provider constructs an `OutputFrame` from every active,
present, globally registered non-coordinate variable before the shared output
coordinator applies `PeriodOutput.variables` selection. The Global Four Degree
setup activates `line_psin`, whose Veros metadata declares dimensions
`("isle", "isle")`. VerCOR's `OutputVariable` contract requires unique dimension
names, so provider sampling fails even though the example requests only eight
unrelated variables.

Direct inspection of the real setup found that `line_psin` is the only
candidate with repeated dimensions. After excluding it, all 117 remaining
variables extract successfully, but coordinate construction exposes four more
unrepresentable candidates: `Ai_ez`, `Ai_nz`, `Ai_bx`, and `Ai_by` use
`tensor1` and `tensor2`, neither of which is registered as a Veros coordinate.
Excluding those four leaves 113 variables whose extraction and coordinate
construction pass against the real Veros 1.6.2 state. Complete frame
construction then exposes one final collision: Veros's scalar native `time`
data variable has the same name as the adapter-owned `time` coordinate.
Excluding that reserved name leaves 112 native variables and eight coordinates;
the complete frame, selected accumulator, and bounded example all pass.

## Intended Behavior

The Veros provider universe contains only variables representable by the
shared `OutputVariable` and NetCDF pipeline. Active variables with repeated
dimension names or unavailable effective coordinate definitions are excluded
before extraction. Native data names that collide with adapter-owned
coordinates are also excluded. The configured period-variable selection
remains coordinator-owned and unchanged.

The `run_jcm_with_veros.py` example must complete its first output accumulation
without attempting to extract `line_psin`, while its requested `temp`, `salt`,
`u`, `v`, `w`, `surface_taux`, `surface_tauy`, and `psi` variables remain
available in their existing manifest order.

## Design

Extend `_active_output_variable_names` in
`vercor/setups/_external/veros_output.py` to resolve dimensions once for each
active, present, globally registered candidate. Use those resolved dimensions
both to identify coordinate variables and to exclude data variables whose
dimension tuple contains repeated names or requires a coordinate that
`_extract_coordinate_variable` cannot construct.

Coordinate representability is evaluated against the effective sample axes:
`timesteps` is allowed because `_drop_timestep_dim` removes it before frame and
coordinate construction. Every other dimension must have a global Veros
variable definition, resolve to the canonical `(dim,)` coordinate layout, and
exist in `veros_state.variables`. This excludes the four `Ai_*` tensor
variables without removing supported time-dependent fields such as `temp`,
`salt`, `u`, `v`, `w`, and `psi`.

The coordinate-name set always includes `VEROS_TIME_DIM` because
`veros_average_coordinate_variables` creates that adapter-owned coordinate even
though no native variable uses it as a dimension. This keeps the scalar Veros
`time` value out of `OutputFrame.variables` and prevents a data-coordinate name
collision.

This filter belongs at provider-universe enumeration because:

- the provider must return a valid complete frame before coordinator selection;
- repeated dimensions violate the existing shared output contract;
- missing tensor coordinates prevent construction of the complete frame;
- the native `time` value collides with the adapter-owned time coordinate;
- passing selections into providers would change the unified provider API;
- renaming `line_psin` axes would invent semantics not supplied by Veros; and
- weakening `OutputVariable` would affect every provider and downstream writer.

State-metadata insertion order, active-state checks, value-presence checks,
coordinate exclusion, snapshot defaults, coordinator filtering, and public
interfaces remain unchanged.

## Testing

Extend the existing Veros provider regression fixture with active, present
`line_psin` metadata and a `(6, 6)` value. Before the first production change,
the test must fail through provider sampling with
`OutputVariable.dims must be unique`. Also add active, present `Ai_ez` metadata
with effective `tensor1` and `tensor2` dimensions. After only the repeated-
dimension filter, the bounded example must reproduce
`Unknown Veros output variable 'tensor2'`. After the complete change, tests
must verify that:

- `line_psin` is absent from the provider frame;
- `Ai_ez` is absent from the provider frame;
- native `time` is absent from variables while the `time` coordinate remains;
- all existing supported native variables remain present and ordered; and
- the resulting frame satisfies the shared output contracts.

Then run the focused Veros output tests, a bounded one-step reproduction of the
example output path, the repository fast and full suites, Black, flake8, mypy,
and the whitespace check. Record observed results in `PROGRESS.md`.

## Error Handling

This change does not suppress extraction errors for otherwise representable
variables. Invalid ranks, missing values, invalid metadata types, and failures
from selected supported variables continue to surface as component-scoped
provider errors. Only candidates known in advance to violate the unique-
dimension, effective-coordinate, or data-coordinate namespace invariants are
excluded from the provider universe.

## Non-Goals

- Exporting `line_psin` by inventing distinct island-axis names.
- Exporting the four `Ai_*` variables by inventing tensor-coordinate metadata.
- Exporting Veros's scalar native `time` value as a data variable alongside the
  adapter-owned `time` coordinate.
- Allowing repeated dimensions in `OutputVariable`.
- Passing `PeriodOutput.variables` into native providers.
- Refactoring the output coordinator or NetCDF writer.
- Changing Veros diagnostic or restart configuration.
