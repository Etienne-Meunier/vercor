from __future__ import annotations

from typing import Any, Mapping, Sequence

from vercor.exceptions import ExchangeError
from vercor.exchanges import Exchange
from vercor._runtime.exchange_keys import exchange_regrid_key
from vercor.fields import VectorField
from vercor.state import RunState
from vercor._runtime.stores import FieldStore


def _dispatch_vector_exchange_field(
    source_fields: FieldStore,
    received_updates: dict[str, Any],
    field_name: VectorField,
    regrid: Any,
) -> None:
    """Dispatch one vector exchange field into the received update mapping."""

    if not all(name in source_fields for name in (field_name.u, field_name.v)):
        raise ExchangeError(
            f"Not all fields in vector {field_name} are present in source fields"
        )
    u_vector, v_vector = regrid.regrid_vector(
        source_fields.get(field_name.u),
        source_fields.get(field_name.v),
    )
    received_updates[field_name.u] = u_vector
    received_updates[field_name.v] = v_vector


def _dispatch_scalar_exchange_field(
    source_fields: FieldStore,
    received_updates: dict[str, Any],
    field_name: str,
    regrid: Any,
    fractional_mask: Any,
) -> None:
    """Dispatch one scalar exchange field into the received update mapping."""

    if field_name not in source_fields:
        raise ExchangeError(f"Field {field_name} not present in source fields")
    received_updates[field_name] = (
        regrid.regrid(source_fields.get(field_name)) * fractional_mask
    )


def dispatch_component_exchanges(
    state: RunState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
) -> RunState:
    """Dispatch destination-specific exchanges into one component."""

    destination_component = state._component_state(destination_name)
    destination_received = destination_component.received
    received_updates: dict[str, Any] = {}

    for exchange in exchanges:
        source_component = state._component_state(exchange.source)
        source_fields = source_component.sent
        key = (exchange.source, exchange.target, exchange_regrid_key(exchange))
        regrid = regridders[key]
        fractional_mask = state._fractional_mask(*key)

        for field_name in exchange.fields:
            if isinstance(field_name, VectorField):
                _dispatch_vector_exchange_field(
                    source_fields,
                    received_updates,
                    field_name,
                    regrid,
                )
            else:
                _dispatch_scalar_exchange_field(
                    source_fields,
                    received_updates,
                    field_name,
                    regrid,
                    fractional_mask,
                )

    destination_received = destination_received.set_many(received_updates)
    destination_component = destination_component.with_received(destination_received)
    return state._with_component_state(destination_name, destination_component)
