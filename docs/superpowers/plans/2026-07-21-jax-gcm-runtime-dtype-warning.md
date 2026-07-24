# JAXGCM Runtime Dtype Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent JAXGCM runtime output mapping from promoting `float32` runtime values to `float64` and emitting JAX's incompatible-scatter `FutureWarning`.

**Architecture:** Keep precision ownership at the JAXGCM adapter boundary. Pass the runtime's frozen `DTypePolicy` as a static argument to the JIT-compiled mapper and canonicalize every mapped physics input with VerCOR's `as_jax_real_array` before numerical helpers run.

**Tech Stack:** Python 3.13, JAX, pytest, VerCOR `DTypePolicy` and dtype conversion helpers.

## Global Constraints

- No public API or generic vertical-coordinate behavior changes.
- Keep the mapping JIT-compatible and differentiable.
- Use `RuntimeOptions.dtype` as the sole precision owner.
- Do not commit until the full unit suite passes, per repository instructions.

---

### Task 1: Enforce runtime precision in JAXGCM output mapping

**Files:**
- Modify: `tests/test_coupler_runtime.py:1656`
- Modify: `tests/test_external_components_coverage.py:395`
- Modify: `vercor/setups/_external/jax_gcm_fields.py:3-163`
- Modify: `vercor/setups/_external/jax_gcm_runtime.py:220-237`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the existing `dtype: DTypePolicy` parameter of `step_jax_gcm_runtime` and `as_jax_real_array(value, policy)`.
- Produces: a keyword-only `dtype: DTypePolicy` parameter on `map_jcm_output_fields`, whose returned floating arrays all use `dtype.jax_real`.

- [x] **Step 1: Make the existing runtime regression fail on the warning and dtype drift**

Add the warning filter and mapped-field dtype assertion:

```python
@pytest.mark.filterwarnings(
    "error:scatter inputs have incompatible types:FutureWarning"
)
def test_jax_gcm_runs_inside_runtime_under_jit_and_grad() -> None:
```

Place this assertion after the existing temperature and sensible-heat-flux checks:

```python
    mapped_float_dtypes = {
        atmosphere_state.fields.get(field_name).dtype
        for field_name in (*JAXGCM_OUTPUT_GRID_FIELD_NAMES, "pressure")
    }
    assert mapped_float_dtypes == {jnp.dtype(jnp.float32)}
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n scipy pytest tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad -q --tb=short
```

Expected: FAIL at `vertical_coordinates.py` because the incompatible scatter warning is promoted to an error (`float64` update into `float32`).

- [x] **Step 3: Propagate and apply the runtime dtype policy**

In `jax_gcm_fields.py`, make `dtype` static for JIT and canonicalize inputs at entry:

```python
from functools import partial

from vercor.dtypes import DTypePolicy, as_jax_real_array


@partial(jax.jit, static_argnames=("dtype",))
def map_jcm_output_fields(
    latvap: float,
    reference_pressure: float,
    sigma_levels: object,
    mwdair: float,
    rgas: float,
    potential_temperature_reference_pressure: float,
    cappa: float,
    surface_sensible_heat_flux: object,
    surface_evaporation: object,
    downward_longwave_radiation_flux: object,
    net_shortwave_radiation_flux: object,
    normalized_surface_pressure: object,
    u_wind: object,
    v_wind: object,
    temperature: object,
    specific_humidity: object,
    *,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    latvap_array = as_jax_real_array(latvap, dtype)
    reference_pressure_array = as_jax_real_array(reference_pressure, dtype)
    sigma_levels_array = as_jax_real_array(sigma_levels, dtype)
    mwdair_array = as_jax_real_array(mwdair, dtype)
    rgas_array = as_jax_real_array(rgas, dtype)
    potential_temperature_reference_pressure_array = as_jax_real_array(
        potential_temperature_reference_pressure, dtype
    )
    cappa_array = as_jax_real_array(cappa, dtype)
    temperature_array = as_jax_real_array(temperature, dtype)
    specific_humidity_array = as_jax_real_array(specific_humidity, dtype)

    u_velocity = as_jax_real_array(u_wind, dtype)[-1, :, :].T
    v_velocity = as_jax_real_array(v_wind, dtype)[-1, :, :].T
    temperature_2m = temperature_array[-1, :, :].T
    specific_humidity_2m = specific_humidity_array[-1, :, :].T / 1000.0

    sensible_heat_flux = -jnp.sum(
        as_jax_real_array(surface_sensible_heat_flux, dtype), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        as_jax_real_array(surface_evaporation, dtype) / 1e3 * latvap_array,
        axis=2,
    ).T
    net_shortwave_radiation_flux_2m = as_jax_real_array(
        net_shortwave_radiation_flux, dtype
    ).T
    downward_longwave_radiation_flux_2m = as_jax_real_array(
        downward_longwave_radiation_flux, dtype
    ).T

    pressure = compute_sigma_pressure_levels(
        reference_pressure_array,
        as_jax_real_array(0.0, dtype),
        sigma_levels_array,
        as_jax_real_array(normalized_surface_pressure, dtype).T,
    )

    density = mwdair_array / rgas_array * pressure[-1, :, :] / temperature_2m
    potential_temperature = temperature_2m * (
        potential_temperature_reference_pressure_array / pressure[-1, :, :]
    ) ** cappa_array

    model_level_height = get_altitudes_sigma_levels(
        temperature_array.transpose((0, 2, 1))[::-1, :, :],
        pressure[::-1, :, :],
        specific_humidity_array.transpose((0, 2, 1))[::-1, :, :] / 1000.0,
    )[1, :, :]

    return {
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "temperature": temperature_2m,
        "specific_humidity": specific_humidity_2m,
        "sensible_heat_flux": sensible_heat_flux,
        "latent_heat_flux": latent_heat_flux,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux_2m,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux_2m,
        "pressure": pressure,
        "density": density,
        "potential_temperature": potential_temperature,
        "model_level_height": model_level_height,
    }
```

In `jax_gcm_runtime.py`, pass the policy explicitly:

```python
    mapped_fields = _jax_gcm_fields.map_jcm_output_fields(
        constants.latent_heat_of_vaporization,
        JCM_REFERENCE_PRESSURE,
        state.sigma_levels,
        constants.dry_air_molecular_weight,
        constants.universal_gas_constant,
        constants.reference_pressure,
        constants.dry_air_kappa,
        averaged_prediction.physics.surface_flux.shf,
        averaged_prediction.physics.surface_flux.evap,
        averaged_prediction.physics.surface_flux.rlds,
        averaged_prediction.physics.shortwave_rad.rsns,
        averaged_prediction.dynamics.normalized_surface_pressure,
        averaged_prediction.dynamics.u_wind,
        averaged_prediction.dynamics.v_wind,
        averaged_prediction.dynamics.temperature,
        averaged_prediction.dynamics.specific_humidity,
        dtype=dtype,
    )
```

Keep the direct mapper JIT coverage compatible by closing over the static policy:

```python
    mapped_fields = jax.jit(
        lambda *args: jax_gcm_fields_module.map_jcm_output_fields(
            *args,
            dtype=DTypePolicy(),
        )
    )(
```

- [x] **Step 4: Run focused and related tests and verify GREEN**

Run:

```bash
conda run -n scipy pytest tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad tests/test_hypsometric.py -q --tb=short
```

Expected: all selected tests PASS; no incompatible-scatter warning appears.

- [x] **Step 5: Update progress and run repository verification**

Add a dated entry to `PROGRESS.md` recording the root cause, adapter-only fix, and RED/GREEN counts. Then run:

```bash
conda run -n scipy pytest tests/ -q --fast
conda run -n scipy pytest tests/ -q
conda run -n scipy black --check vercor tests
conda run -n scipy flake8 vercor tests --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor tests
git diff --check
```

Expected: fast and full tests PASS without the JAX scatter warning; formatting, lint, type checking, and diff checks PASS.

- [x] **Step 6: Commit the verified change**

```bash
git add docs/superpowers/specs/2026-07-21-jax-gcm-runtime-dtype-warning-design.md docs/superpowers/plans/2026-07-21-jax-gcm-runtime-dtype-warning.md tests/test_coupler_runtime.py vercor/setups/_external/jax_gcm_fields.py vercor/setups/_external/jax_gcm_runtime.py PROGRESS.md
git commit -m "fix: preserve JAXGCM runtime dtype"
```
