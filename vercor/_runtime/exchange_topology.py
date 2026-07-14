from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from vercor.dtypes import DTypePolicy, jax_ones
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor._runtime.exchange_keys import exchange_regrid_key
from vercor.jax_logging import LoggerLike
from vercor._runtime.topology_state import RuntimeTopologyMaps

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def build_exchange_topology_maps(
    *,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    dtype: DTypePolicy,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
) -> RuntimeTopologyMaps:
    """Build exchange regridders and identity masks for configured topology."""

    regridders = {} if topology_maps is None else dict(topology_maps.regridders)
    binary_masks = {} if topology_maps is None else dict(topology_maps.binary_masks)
    fractional_masks = (
        {} if topology_maps is None else dict(topology_maps.fractional_masks)
    )
    configured_keys: set[tuple[str, str, str]] = set()

    for exchange in exchanges:
        key = (exchange.source, exchange.target, exchange_regrid_key(exchange))
        if key in configured_keys:
            raise CouplerError(
                f"Duplicate exchange topology key {key!r}; merge field declarations "
                "into one Exchange or give the exchanges distinct regrid factories."
            )
        configured_keys.add(key)
        if key not in regridders:
            regridders[key] = exchange.regrid(
                components[exchange.source].grid,
                components[exchange.target].grid,
            )
            binary_masks[key] = jax_ones(
                components[exchange.target].grid.shape,
                dtype,
            )
            fractional_masks[key] = jax_ones(
                components[exchange.target].grid.shape,
                dtype,
            )

    return RuntimeTopologyMaps(
        regridders=regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )


__all__ = ["build_exchange_topology_maps"]
