from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.components.setup_validation import validate_component_setup
from vercor.dtypes import jax_zeros
from vercor.field_layout import validate_component_data_layout
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor.output._session import validate_period_output_component_state
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def prefill_runtime_contract_fields(
    component: "Component",
    data: dict[str, RuntimeArray],
    received: dict[str, RuntimeArray],
    sent: dict[str, RuntimeArray],
    contract: ExchangeContract,
) -> None:
    """Add generic import/export fields required for stable runtime execution."""

    zeros = jax_zeros(component.grid.shape, component.settings)
    for field_name in contract.receives:
        received.setdefault(field_name, zeros)
        data.setdefault(field_name, zeros)
    for field_name in contract.sends:
        sent.setdefault(field_name, data.get(field_name, zeros))
        data.setdefault(field_name, zeros)


def create_runtime_component_state(
    component: "Component",
    *,
    prefill_missing: bool = False,
    contract: ExchangeContract,
) -> ComponentRuntimeState:
    """Create immutable runtime state from a component's seed data."""

    validate_component_setup(component)
    data = dict(component._data)
    received: dict[str, RuntimeArray] = {}
    sent: dict[str, RuntimeArray] = {}
    if prefill_missing:
        component._prefill_runtime_state_fields(data, received, sent, contract)
        prefill_runtime_contract_fields(component, data, received, sent, contract)

    validate_component_data_layout(
        component_name=component.name,
        grid_shape=component.grid.shape,
        data=data,
    )

    state = ComponentRuntimeState(
        fields=FieldStore.from_mapping(data),
        received=FieldStore.from_mapping(received),
        sent=FieldStore.from_mapping(sent),
        payload=component._create_runtime_payload(),
    )
    validate_period_output_component_state(component, state)
    return state
