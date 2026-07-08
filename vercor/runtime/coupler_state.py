from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from vercor.exchanges import Exchange
from vercor.runtime.component_state import create_runtime_component_state
from vercor.runtime.contracts import (
    RuntimeComponentContract,
    build_runtime_contracts,
    exchange_key_name,
)
from vercor.state import RunState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def runtime_state_from_components(
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    *,
    contracts: Mapping[str, RuntimeComponentContract] | None = None,
    prefill_missing: bool = False,
) -> RunState:
    """Create immutable runtime state from component setup objects."""

    runtime_contracts = (
        build_runtime_contracts(
            tuple(components),
            exchanges,
            validate_endpoints=False,
        )
        if contracts is None
        else contracts
    )
    runtime_components = tuple(
        create_runtime_component_state(
            component,
            prefill_missing=prefill_missing,
            contract=runtime_contracts[name],
        )
        for name, component in components.items()
    )
    runtime_fractional_masks = {
        exchange_key_name(*key): value for key, value in fractional_masks.items()
    }
    return RunState._from_runtime(
        component_names=tuple(components.keys()),
        components=runtime_components,
        fractional_masks=RuntimeFieldStore.from_mapping(runtime_fractional_masks),
        component_grids=tuple(component.grid for component in components.values()),
    )
