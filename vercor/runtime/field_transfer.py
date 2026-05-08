from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.time import RuntimeStepInfo
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def receive_runtime_fields(
    component_state: RuntimeComponentState,
    contract: RuntimeComponentContract,
) -> RuntimeComponentState:
    """Move imported incoming runtime fields into component data."""

    data = component_state.data
    for field_name in contract.imports:
        data = data.set(field_name, component_state.incoming.get(field_name))
    return component_state.with_data(data)


def _select_runtime_field_for_send(
    component: "Component",
    component_state: RuntimeComponentState,
    field_name: str,
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    field = component_state.data.get(field_name)
    if step_info is None:
        return field

    if component.settings.apply_time_interpolation:
        arr = jnp.asarray(field)
        left = jnp.take(arr, step_info.monthly_index_left, axis=0)
        right = jnp.take(arr, step_info.monthly_index_right, axis=0)
        return (
            step_info.monthly_weight_left * left
            + step_info.monthly_weight_right * right
        )

    if component.settings.get_field_time_slice:
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)

    return field


def send_runtime_fields(
    component: "Component",
    component_state: RuntimeComponentState,
    step_info: RuntimeStepInfo | None = None,
    *,
    contract: RuntimeComponentContract,
) -> RuntimeComponentState:
    """Move exported component data into outgoing runtime fields."""

    outgoing = component_state.outgoing
    for field_name in contract.exports:
        outgoing = outgoing.set(
            field_name,
            _select_runtime_field_for_send(
                component,
                component_state,
                field_name,
                step_info,
            ),
        )
    return component_state.with_outgoing(outgoing)
