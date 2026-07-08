from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.fields import ExchangeField, flatten_field_items


@dataclass(frozen=True)
class ExchangeContract:
    """Coupler-owned runtime import/export metadata for one component."""

    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()

    @property
    def all_fields(self) -> tuple[str, ...]:
        """Return all import/export fields while preserving contract order."""

        return (*self.receives, *self.sends)


def flatten_exchange_fields(field_names: Sequence[ExchangeField]) -> list[str]:
    """Return scalar field names from scalar and vector exchange declarations."""

    return flatten_field_items(field_names)


def append_unique_runtime_fields(target: list[str], exchange_items: list[str]) -> None:
    """Append exchange field names while preserving first-seen order."""

    target.extend([item for item in exchange_items if item not in target])


def _extend_contract_fields(
    fields: tuple[str, ...],
    new_fields: list[str],
) -> tuple[str, ...]:
    """Return ``fields`` extended by new unique field names."""

    updated = list(fields)
    append_unique_runtime_fields(updated, new_fields)
    return tuple(updated)


def build_exchange_contracts(
    component_names: Sequence[str],
    exchanges: Sequence[Exchange],
    *,
    validate_endpoints: bool,
) -> dict[str, ExchangeContract]:
    """Build coupler-owned import/export contracts from exchange declarations."""

    known_components = set(component_names)
    contracts = {name: ExchangeContract() for name in component_names}
    for exchange in exchanges:
        if exchange.source not in known_components:
            if validate_endpoints:
                raise CouplerError(
                    f"Source component '{exchange.source}' not registered in coupler"
                )
            continue
        if exchange.target not in known_components:
            if validate_endpoints:
                raise CouplerError(
                    f"Destination component '{exchange.target}' not registered in coupler"
                )
            continue

        flattened_fields = flatten_exchange_fields(exchange.fields)
        source_contract = contracts[exchange.source]
        destination_contract = contracts[exchange.target]
        contracts[exchange.source] = ExchangeContract(
            receives=source_contract.receives,
            sends=_extend_contract_fields(
                source_contract.sends,
                flattened_fields,
            ),
        )
        contracts[exchange.target] = ExchangeContract(
            receives=_extend_contract_fields(
                destination_contract.receives,
                flattened_fields,
            ),
            sends=destination_contract.sends,
        )
    return contracts


def exchange_key(source: str, destination: str, regrid_key: str) -> str:
    """Return a stable field-store key for exchange metadata arrays."""

    return f"{source}|{destination}|{regrid_key}"
