from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.time import RuntimeStepInfo
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def receive_runtime_fields(
    component_state: ComponentRuntimeState,
    contract: ExchangeContract,
) -> ComponentRuntimeState:
    """Move received runtime fields into component fields."""

    return component_state.with_fields(
        component_state.fields.set_many(
            {
                field_name: component_state.received.get(field_name)
                for field_name in contract.receives
            }
        )
    )


def _select_runtime_field_for_send(
    component: "Component",
    component_state: ComponentRuntimeState,
    field_name: str,
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    field = component_state.fields.get(field_name)
    if step_info is None:
        return field

    import_policy = component.import_policy

    if import_policy.time_interpolation:
        arr = jnp.asarray(field)
        left = jnp.take(arr, step_info.monthly_index_left, axis=0)
        right = jnp.take(arr, step_info.monthly_index_right, axis=0)
        return (
            step_info.monthly_weight_left * left
            + step_info.monthly_weight_right * right
        )

    if import_policy.daily_selection:
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)

    return field


def send_runtime_fields(
    component: "Component",
    component_state: ComponentRuntimeState,
    step_info: RuntimeStepInfo | None = None,
    *,
    contract: ExchangeContract,
) -> ComponentRuntimeState:
    """Move component fields into sent runtime fields."""

    return component_state.with_sent(
        component_state.sent.set_many(
            {
                field_name: _select_runtime_field_for_send(
                    component,
                    component_state,
                    field_name,
                    step_info,
                )
                for field_name in contract.sends
            }
        )
    )
