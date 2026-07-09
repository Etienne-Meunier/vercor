from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import vercor._runtime.exchange_topology as _exchange_topology
import vercor._runtime.surface_masks as _surface_masks
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
)
from vercor.runtime import SurfaceMaskPolicy
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components.base import Component


def build_exchange_topology(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
    surface_mask_policy: SurfaceMaskPolicy | None = None,
) -> ExchangeTopologyState:
    """Build exchange regridders, masks, and surface topology state."""

    initialized_maps = _exchange_topology.build_exchange_topology_maps(
        components=components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        settings=settings,
        logger=logger,
    )
    surface_masks = None
    if _surface_masks.should_apply_surface_mask_policy(
        components,
        exchanges,
        surface_mask_policy,
    ):
        if surface_mask_policy is None:
            raise AssertionError("surface_mask_policy unexpectedly missing")
        surface_masks = _surface_masks.create_surface_exchange_masks(
            components,
            policy=surface_mask_policy,
            logger=logger,
        )
        _surface_masks.validate_land_mask_consistency(
            components,
            surface_masks,
            policy=surface_mask_policy,
        )
        logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")
        _surface_masks.apply_surface_exchange_masks(
            initialized_maps,
            surface_masks=surface_masks,
            policy=surface_mask_policy,
        )
        logger.info(" Exchange masks patching complete")
    return ExchangeTopologyState(
        topology_maps=initialized_maps,
        surface_masks=surface_masks,
    )


__all__ = ["build_exchange_topology"]
