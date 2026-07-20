from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, NamedTuple, cast

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor._host_arrays import runtime_array_to_host

from veros.state import VerosState


class VerosForcingFields(NamedTuple):
    """Prepared Veros surface forcing fields in host-state variable order."""

    taux: jax.Array
    tauy: jax.Array
    qnet: jax.Array
    qnec: jax.Array


@jax.jit
def update_veros_interior(
    array: object,
    interior_value: object,
) -> jax.Array:
    array_jax = as_jax_real_array(array)
    interior_value_jax = as_jax_real_array(interior_value)
    return array_jax.at[2:-2, 2:-2, ...].set(interior_value_jax)


@jax.jit
def prepare_surface_forcing_fields(
    taux: object,
    tauy: object,
    qnet: object,
    qnec: object,
    restore_to_climatology: object,
) -> VerosForcingFields:
    restore_to_climatology_jax = jnp.asarray(restore_to_climatology, dtype=bool)

    def _prepare(field: object) -> jax.Array:
        field_jax = as_jax_real_array(field)
        return jnp.nan_to_num(field_jax.T[..., jnp.newaxis])

    taux_prepared = _prepare(taux)
    tauy_prepared = _prepare(tauy)
    qnet_prepared = _prepare(qnet)
    qnec_prepared = _prepare(qnec)
    qnec_prepared = jnp.where(
        restore_to_climatology_jax, qnec_prepared, jnp.zeros_like(qnec_prepared)
    )

    return VerosForcingFields(
        taux=taux_prepared,
        tauy=tauy_prepared,
        qnet=qnet_prepared,
        qnec=qnec_prepared,
    )


@jax.jit
def extract_surface_temperature(
    temperature: object,
    tau: object,
) -> jax.Array:
    temperature_array = as_jax_real_array(temperature)
    tau_index = as_jax_index_array(tau)
    return temperature_array[2:-2, 2:-2, -1, tau_index].T + 273.15


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

    return state_copy


def pure(state: VerosState, jitted: bool, step: Callable) -> VerosState:
    """Convert an in-place Veros step into a copy-before-mutate helper."""

    n_state = copy_state(state, jitted=jitted)
    step(n_state)
    return n_state


def extract_veros_runtime_sst(state: VerosState) -> jax.Array:
    """Return the Veros surface temperature field in VerCOR runtime layout."""

    return cast(
        jax.Array,
        extract_surface_temperature(
            state.variables.temp,
            state.variables.tau,
        ),
    )


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
            setattr(variables, variable_name, runtime_array_to_host(updated))
    return updated_state


def advance_veros_substeps(
    state: VerosState,
    *,
    step_function: Callable[[VerosState], VerosState],
    model_substeps: int,
    logger: Any | None,
) -> VerosState:
    """Advance Veros through the configured number of host substeps."""

    updated_state = state
    for i in range(model_substeps):
        if logger is not None:
            logger.info(f"Veros sub-step {i+1} / {model_substeps}")
        updated_state = step_function(updated_state)
    return updated_state


__all__ = [
    "VerosForcingFields",
    "advance_veros_substeps",
    "apply_veros_forcing_fields",
    "extract_surface_temperature",
    "extract_veros_runtime_sst",
    "prepare_surface_forcing_fields",
    "update_veros_interior",
    "copy_state",
    "pure",
]
