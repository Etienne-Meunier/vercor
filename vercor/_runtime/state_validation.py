from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.field_layout import validate_component_data_layout
from vercor._runtime.exchange_keys import exchange_regrid_key
from vercor._runtime.contracts import ExchangeContract, exchange_key
from vercor.state import RunState
from vercor._runtime.validation import validate_component_runtime_contract_fields

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def validate_runtime_state(
    runtime_state: RunState,
    *,
    components: Mapping[str, _ComponentBinding],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, ExchangeContract],
    run_order: Sequence[str] | None,
) -> None:
    """Validate that runtime state matches the configured coupler topology."""

    if run_order is None:
        raise CouplerError("Runtime requires a configured component run sequence")

    run_order = tuple(run_order)
    expected_component_names = set(components)
    runtime_component_names = set(runtime_state.component_names)
    missing_components = sorted(expected_component_names - runtime_component_names)
    extra_components = sorted(runtime_component_names - expected_component_names)
    duplicate_components = sorted(
        name
        for name in runtime_component_names
        if runtime_state.component_names.count(name) > 1
    )
    if missing_components or extra_components or duplicate_components:
        details = []
        if missing_components:
            details.append(
                "missing from runtime state: " + ", ".join(missing_components)
            )
        if extra_components:
            details.append(f"extra components: {', '.join(extra_components)}")
        if duplicate_components:
            details.append(f"duplicate {', '.join(duplicate_components)}")
        raise CouplerError(
            "Runtime component names must exactly match registered components: "
            + "; ".join(details)
        )

    for cname in run_order:
        if cname not in components:
            raise CouplerError(
                f"Run-sequence component '{cname}' is not registered in coupler"
            )
        if cname not in runtime_component_names:
            raise CouplerError(
                f"Run-sequence component '{cname}' is missing from runtime state"
            )

    for cname, component in components.items():
        component_state = runtime_state._component_state(cname)
        contract = contracts[cname]
        validate_component_data_layout(
            component_name=component.name,
            grid_shape=component.grid.shape,
            data=component_state.fields.to_mapping(),
        )
        validate_component_runtime_contract_fields(
            component,
            component_state,
            contract,
        )
        component._validate_runtime_state(
            component_state,
            contract,
        )

    for exchange in exchanges:
        key = (exchange.source, exchange.target, exchange_regrid_key(exchange))
        if exchange.source not in runtime_component_names:
            raise CouplerError(
                f"Exchange source component '{exchange.source}' is missing from runtime state"
            )
        if exchange.target not in runtime_component_names:
            raise CouplerError(
                f"Exchange destination component '{exchange.target}' is missing from runtime state"
            )
        if key not in regridders:
            raise CouplerError(
                "Runtime requires an initialized regridder for exchange "
                f"{exchange.label}"
            )

        mask_name = exchange_key(*key)
        if mask_name not in runtime_state._fractional_masks.field_names:
            raise CouplerError(
                "Runtime requires an initialized fractional mask for exchange "
                f"{exchange.label}"
            )
        destination_shape = components[exchange.target].grid.shape
        mask_shape = jnp.asarray(runtime_state._fractional_masks.get(mask_name)).shape
        if mask_shape != destination_shape:
            raise CouplerError(
                "Runtime fractional mask for exchange "
                f"{exchange.label} has shape {mask_shape}, expected {destination_shape}"
            )
