# JAXGCM Runtime Dtype Warning Design

## Goal

Eliminate the JAX scatter `FutureWarning` emitted by
`test_jax_gcm_runs_inside_runtime_under_jit_and_grad` while preserving the
runtime-owned precision policy, JIT compatibility, and differentiability.

## Root cause

The coupler uses `DTypePolicy(enable_x64=False)`, so its runtime state and
physical constants are `float32`. The JAXGCM output mapper nevertheless accepts
the Python `JCM_REFERENCE_PRESSURE` and the setup's `sigma_levels` without the
runtime policy. With global JAX x64 enabled by the test, those values become
`float64` and promote the pressure and altitude increments. The altitude helper
then attempts to scatter a `float64` update into a `float32` array.

## Design

Keep the fix within the JAXGCM adapter. Pass the runtime `DTypePolicy` from
`step_jax_gcm_runtime` into `map_jcm_output_fields` as a static JIT argument.
Canonicalize every physics scalar and array at that adapter boundary with
`as_jax_real_array(value, dtype)`. This makes mapped pressure, altitude, and all
other returned fields follow the runtime precision owner before they enter
generic numerical helpers.

Do not change `get_altitudes_sigma_levels`: inferring or forcing its dtype would
alter a general numerical helper's existing mixed-dtype behavior. Do not cast
only the scatter update because that would hide the upstream promotion rather
than prevent it.

## Testing

Extend the existing JAXGCM JIT-and-gradient regression test so the incompatible
scatter `FutureWarning` is treated as an error. Assert that all floating mapped
fields use the configured `float32` runtime dtype. Run the focused test first,
then related JAXGCM and vertical-coordinate tests, the fast suite, and the full
suite before any commit.

## Scope

No public API, general vertical-coordinate semantics, or unrelated JAXGCM
behavior changes. Update `PROGRESS.md` with the RED/GREEN and suite results.
