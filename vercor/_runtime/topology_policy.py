from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from vercor.components.contracts import ComponentInfo
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.exchange_keys import exchange_regrid_key
import vercor._runtime.surface_masks as _surface_masks
from vercor._runtime.topology_state import RuntimeTopologyMaps, SurfaceExchangeMasks
from vercor.settings import Settings
from vercor.topology import (
    ExchangeTopologyPatch,
    SurfaceMaskPolicy,
    TopologyContext,
    TopologyPolicy,
)

if TYPE_CHECKING:
    from vercor.components.base import Component


def build_topology_context(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    logger: LoggerLike,
) -> TopologyContext:
    """Return public topology-policy context from private runtime inputs."""

    component_info = MappingProxyType(
        {
            name: ComponentInfo(
                name=component.name,
                grid=component.grid,
                spec=component.spec,
            )
            for name, component in components.items()
        }
    )
    return TopologyContext(
        components=component_info,
        exchanges=tuple(exchanges),
        exchange_keys=tuple(
            (exchange.source, exchange.target, exchange_regrid_key(exchange))
            for exchange in exchanges
        ),
        settings=settings,
        logger=logger,
    )


def apply_topology_policy(
    topology_maps: RuntimeTopologyMaps,
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    logger: LoggerLike,
    policy: TopologyPolicy | None,
) -> SurfaceExchangeMasks | None:
    """Apply an optional public topology policy to private runtime maps."""

    if policy is None:
        return None

    context = build_topology_context(
        components=components,
        exchanges=exchanges,
        settings=settings,
        logger=logger,
    )
    if not policy.applies(context):
        return None

    surface_masks = None
    if isinstance(policy, SurfaceMaskPolicy):
        patch, surface_masks = _surface_masks.build_surface_mask_topology_patch(
            context,
            policy,
        )
        logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")
    else:
        patch = policy.build(context)

    _apply_exchange_topology_patch(topology_maps, patch)
    logger.info(" Exchange topology policy patching complete")
    return surface_masks


def _apply_exchange_topology_patch(
    topology_maps: RuntimeTopologyMaps,
    patch: ExchangeTopologyPatch,
) -> None:
    """Apply public topology mask patches to runtime topology maps."""

    for key, value in patch.binary_masks.items():
        topology_maps.binary_masks[key] = value
    for key, value in patch.fractional_masks.items():
        topology_maps.fractional_masks[key] = value


__all__ = [
    "apply_topology_policy",
    "build_topology_context",
]
