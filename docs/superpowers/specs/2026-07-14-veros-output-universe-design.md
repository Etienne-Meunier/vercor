# Veros Output Universe Bug-Fix Design

## Problem

The bundled Veros output provider samples every active entry in
`veros_state.var_meta` before the shared output coordinator applies a configured
`PeriodOutput.variables` selection. `CustomGlobalFourDegree` adds setup-local
state fields such as `sss_clim` to that metadata. Those fields are not part of
Veros's global output-variable registry, which is the registry used by VerCOR's
snapshot extraction and validation path. Sampling therefore fails with
`Unknown Veros output variable 'sss_clim'` even when the user explicitly selects
only supported variables.

## Intended Behavior

The bundled provider's output universe contains only active, present variables
defined in Veros's global variable registry. Setup-local fields such as
`sss_clim` and `sst_clim` are internal model inputs and are excluded. Explicitly
requesting an excluded setup-local field remains an unknown-variable error.

## Design

Change `_active_output_variable_names` in
`vercor/setups/_external/veros_output.py` so it filters state metadata entries
against `veros.variables.VARIABLES` before resolving dimensions, identifying
coordinate variables, or returning sample names. The remaining rules stay
unchanged:

- preserve the state metadata's manifest order;
- require the variable to be active for the current settings;
- require a corresponding value in `veros_state.variables`;
- exclude coordinate variables from the returned data-variable names.

This keeps enumeration and extraction on the same authoritative registry and
does not change the public output-provider or coordinator contracts.

## Testing

Extend the Veros provider regression fixture with active setup-local
`sss_clim` metadata and values. The provider test must first fail with the
reported unknown-variable error, then pass after the implementation change and
verify that:

- all existing supported native variables remain present and ordered;
- `sss_clim` is absent from the returned frame;
- sampling completes without inspecting unsupported setup-local dimensions.

Retain the existing unknown-variable validation behavior for explicitly
selected unsupported names. Run the focused Veros output tests, the repository
fast suite, formatting, linting, type checking, and whitespace checks.

## Non-Goals

- Exporting setup-local Veros forcing or climatology fields.
- Changing uniform `PeriodOutput.variables` selection semantics.
- Refactoring the output provider interface or passing selections into
  providers.
- Changing Veros snapshot defaults or NetCDF dimension ordering.
