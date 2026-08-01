# Differentiable VerCor - Code Changes

Goal: run `jax.grad` through a multi-step Veros ocean rollout, coupled through our `Coupler`. 

---

## 1. Safe math operations for gradients

`jnp.sqrt(x)` has derivative `0.5/sqrt(x)`, infinite at `x=0`. 

**Before:**
```python
# vercor/fluxes/bulk_formula_cesm.py
xsq = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
xqq = jnp.sqrt(xsq)
...

# vercor/fluxes/utilities.py
umps_array = as_jax_real_array(umps)
return 0.0027 / umps_array + 0.000142 + 0.0000764 * umps_array
```

**After:**
```python
# vercor/fluxes/utilities.py
@custom_jvp
def safe_sqrt(x: ArrayLike) -> jax.Array:
    return jnp.sqrt(as_jax_real_array(x))

@safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = jnp.sqrt(as_jax_real_array(x))
    grad = 0.5 / jnp.maximum(primal_out, _SAFE_SQRT_GRAD_FLOOR)  # 1e-8
    return primal_out, grad * x_dot

...
umps_array = jnp.maximum(as_jax_real_array(umps), _SAFE_SQRT_GRAD_FLOOR)
return 0.0027 / umps_array + 0.000142 + 0.0000764 * umps_array

# vercor/fluxes/bulk_formula_cesm.py
xsq = jnp.maximum(safe_sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
xqq = safe_sqrt(xsq)
```
Same forward output, floored gradient. `cdn()`'s division got the same floor treatment for the same reason.

**Files / lines:**
- `vercor/fluxes/utilities.py:10-38,78` 
- `vercor/fluxes/bulk_formula_cesm.py:8,33-34,63,141,160,268-269,342` 

---

## 2. Adapting the forcing (sanitizing exchanged fields)

Regridding returns `NAN` on Land which was dealt with a nan to num but it broke the gradients so we replace the values before.

**Before:**
```python
# vercor/setups/_external/veros_fluxes.py
as_jax_real_array(runtime_fields["model_level_height"], dtype),
as_jax_real_array(runtime_fields["potential_temperature"], dtype),
...
```

**After:**
```python
# vercor/setups/_external/veros_fluxes.py
def _sanitize_runtime_field(value, dtype):
    array = as_jax_real_array(value, dtype)
    return jnp.where(jnp.isnan(array), 1.0, array)  # masked out downstream; just needs to avoid log(0)/div-by-0

...
_sanitize_runtime_field(runtime_fields["model_level_height"], dtype),
_sanitize_runtime_field(runtime_fields["potential_temperature"], dtype),
```
Replaced NaNs *before* they reach the formula, not after. As a consequence, `nan_to_num` on the forcing output downstream became redundant and was dropped:

**Before:**
```python
# vercor/setups/_external/veros_state.py — prepare_surface_forcing_fields()._prepare
def _prepare(field: object) -> jax.Array:
    field_jax = as_jax_real_array(field)
    return jnp.nan_to_num(field_jax.T[..., jnp.newaxis])
```

**After:**
```python
# vercor/setups/_external/veros_state.py — prepare_surface_forcing_fields()._prepare
def _prepare(field: object) -> jax.Array:
    # Dropped nan_to_num: forcings must already be NaN-free here (see compute_fluxes).
    field_jax = as_jax_real_array(field)
    return field_jax.T[..., jnp.newaxis]
```

*Note* : we replace with `1.0` and not `0.0` because we have log downstream. 

**Files / lines:**
- `vercor/setups/_external/veros_fluxes.py:21,34-37,42-53,94-100,108-113`
- `vercor/setups/_external/veros_state.py:45`

---

## 3. Backend switch: numpy → JAX

`jax.grad` only works through JAX ops. Veros's numpy backend can't be differentiated at all.

**Before:**

```python
# vercor/setups/_external/veros_runtime_settings.py
_set_runtime_setting(runtime_settings, "backend", "numpy")
```

**After:**
```python
# vercor/setups/_external/veros_runtime_settings.py
_set_runtime_setting(runtime_settings, "backend", "jax")
...
_set_runtime_setting(runtime_settings, 'linear_solver', 'scipy_jax')
```

**Files / lines:**
- `vercor/setups/_external/veros_runtime_settings.py:24,27`

---

## 4. `execution` added to `VerosConfig`

- The ocean component's `ComponentSpec.execution` was hardcoded to `"host"` in `veros_gcm.py` 
- Added `execution` to `VerosConfig` ->  `execution="jax"` is now reachable.

```python
# vercor/setups/config.py
execution: Literal["jax", "host"] = "host"

# vercor/setups/_external/veros_gcm.py
execution=config.execution,  # was: execution="host"
```
Differentiable examples set both explicitly: `VerosConfig(..., jitted=True, execution="jax")`.

**Files / lines:**
- `vercor/setups/config.py:47-70`
- `vercor/setups/_external/veros_gcm.py:65`

---

## 5. State copying done as in Veros-AD

The copy of Vercor was building back the full state which was too heavy.

**Before:**

```python
# vercor/setups/_external/veros_state.py
def copy_state(tree: VerosState, jitted: bool = True) -> VerosState:
    """Return a copy of a Veros state suitable for copy-before-mutate stepping."""

    if jitted:
        dimensions = deepcopy(tree._dimensions)
        settings_meta = deepcopy(tree.settings.__metadata__)
        plugin_interfaces = deepcopy(tree._plugin_interfaces)
        var_meta = deepcopy(tree._var_meta)

        state_copy = VerosState(
            var_meta, settings_meta, dimensions, plugin_interfaces=plugin_interfaces
        )

        with state_copy.settings.unlock():
            for k, v in tree.settings.items():
                state_copy.settings.__setattr__(k, v)

        state_copy._variables = deepcopy(tree._variables)
        state_copy.timers = deepcopy(tree.timers)
        state_copy.profile_timers = deepcopy(tree.profile_timers)
    else:
        state_copy = tree

    object.__setattr__(
        state_copy.settings,
        "__fields__",
        tuple(state_copy.settings.__fields__),
    )
    return state_copy
```

**After (using the pytree functionality - it's the same I did in Veros) :**

```python
# vercor/setups/_external/veros_state.py
def copy_state(tree: VerosState, jitted: bool = True) -> VerosState:

    state_copy = tree.copy() if jitted else tree
    return state_copy
```

`copy_state()` is called from `apply_veros_forcing_fields()` every step, right before writing the new forcing into the state's variables:

```python
# vercor/setups/_external/veros_state.py
def apply_veros_forcing_fields(
    state: VerosState,
    forcing_fields: VerosForcingFields,
    *,
    jitted: bool,
) -> VerosState:
    """Write prepared VerCOR forcing fields into Veros state variables."""

    updated_state = copy_state(state, jitted=jitted)
    variables = updated_state.variables
    with variables.unlock():
        for variable_name, variable_value in zip(
            ("taux", "tauy", "qnet", "qnec"),
            forcing_fields,
            strict=True,
        ):
            current = getattr(variables, variable_name)
            updated = update_veros_interior(current, variable_value)
            setattr(
                variables,
                variable_name,
                updated
                if isinstance(updated, jax.core.Tracer)
                else runtime_array_to_host(updated),
            )
    return updated_state
```
That `isinstance(updated, jax.core.Tracer)` check is the one flagged in the open questions below — it's what decides whether the updated array stays on-device (inside a trace) or gets pulled back to host, and it's this function's `copy_state()` call whose PyTreeDef stability makes the surrounding `lax.scan` loop-carry work at all.

**Files / lines:**
- `vercor/setups/_external/veros_state.py:75-88,99,216`

---

## 7. Gradient checkpointing

Differentiating a long rollout normally keeps every step's intermediate values in memory for the backward pass. Checkpointing recomputes them instead — slower, but scales to long runs.

**Before:**
```python
# vercor/_runtime/backends.py
final_state, _ = jax.lax.scan(run_step, runtime_state, ...)
```

**After:**
```python
# vercor/_runtime/backends.py
final_state, _ = jax.lax.scan(jax.checkpoint(run_step), runtime_state, ...)
```

**Files / lines:**
- `vercor/_runtime/backends.py:124-125`

---

## 8. New ocean setups (additions)

Two new setups, selectable via `VerosConfig(setup=...)`, dispatched through a lookup table:
```python
# vercor/setups/_external/veros_gcm_state.py
VEROS_SETUP_FACTORIES: dict[str, type] = {
    "global_4deg": _veros_setup.CustomGlobalFourDegree,
    "acc": _veros_setup_acc.ACCSetup,
    "global_4deg_learning": _veros_setup_global4deg_learning.GlobalFourDegreeLearningSetup,
}
```
- **`acc`** — small, cheap, self-forcing channel setup. Used to sanity-check gradients before touching the expensive full-size model.
- **`global_4deg_learning`** — differentiable variant of our real global 4-degree setup: disables the unpatched streamfunction solver, switches the equation of state to avoid another unguarded `sqrt`, disables diagnostics that don't survive jit tracing.

**Files:**
- `vercor/setups/_external/veros_setup_acc.py` (new)
- `vercor/setups/_external/veros_setup_global4deg_learning.py` (new)
- `vercor/setups/_external/veros_gcm_state.py:29` — dispatch table

---

## 9. Examples 

- `examples/run_verosad.py` — forward-only smoke test, small setup.
- `examples/run_verosad_grad.py` — Gradient through a multi-step rollout
- `examples/run_verosad_global4deg.py` / `..._grad.py` — Full ERA5 atmosphere/land/ocean coupling differentiable
- **`examples/notebooks/verosad_grad_demo.ipynb` + `_verosad_grad_helpers.py` — Demonstrate gradients**



