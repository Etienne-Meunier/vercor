from __future__ import annotations

from typing import Any, Mapping, Sequence

from vercor.exceptions import ExchangerError
from vercor.exchange import Exchange
from vercor.runtime.state import RuntimeCouplerState


def dispatch_component_exchanges(
    state: RuntimeCouplerState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
) -> RuntimeCouplerState:
    """Dispatch all exchanges targeting one destination component."""

    destination_component = state.get_component_state(destination_name)
    destination_incoming = destination_component.incoming

    for exchange in exchanges:
        if exchange.destination != destination_name:
            continue

        source_component = state.get_component_state(exchange.source)
        source_fields = source_component.outgoing
        key = (exchange.source, exchange.destination, exchange.interpolation_type)
        regrid = regridders[key]
        fractional_mask = state.get_fractional_mask(*key)

        for field_name in exchange.field_names:
            if isinstance(field_name, tuple):
                if not all(name in source_fields for name in field_name):
                    raise ExchangerError(
                        f"Not all fields in vector {field_name} are present in source fields"
                    )
                u_vector, v_vector = regrid(
                    source_fields.get(field_name[0]),
                    source_fields.get(field_name[1]),
                )
                destination_incoming = destination_incoming.set(field_name[0], u_vector)
                destination_incoming = destination_incoming.set(field_name[1], v_vector)
            else:
                if field_name not in source_fields:
                    raise ExchangerError(
                        f"Field {field_name} not present in source fields"
                    )
                scalar = regrid(source_fields.get(field_name)) * fractional_mask
                destination_incoming = destination_incoming.set(field_name, scalar)

    destination_component = destination_component.with_incoming(destination_incoming)
    return state.set_component_state(destination_name, destination_component)
