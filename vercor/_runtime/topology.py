from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import vercor._runtime.exchange_topology as _exchange_topology
import vercor._runtime.topology_policy as _topology_policy
from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
)
from vercor.settings import Settings
from vercor.topology import TopologyPolicy

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def build_exchange_topology(
    *,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    dtype: DTypePolicy,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
    topology_policy: TopologyPolicy | None = None,
) -> ExchangeTopologyState:
    """Build exchange regridders, masks, and surface topology state."""

    initialized_maps = _exchange_topology.build_exchange_topology_maps(
        components=components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        dtype=dtype,
        logger=logger,
    )
    prepared_maps = _topology_policy.apply_topology_policy(
        initialized_maps,
        components=components,
        exchanges=exchanges,
        settings=settings,
        logger=logger,
        policy=topology_policy,
    )
    return ExchangeTopologyState(
        topology_maps=prepared_maps,
    )


__all__ = ["build_exchange_topology"]
