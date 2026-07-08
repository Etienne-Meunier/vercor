from __future__ import annotations

from dataclasses import dataclass, field

from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.interrupts import RuntimeInterruptController
from vercor._runtime.topology_state import RuntimeTopologyMaps


@dataclass(slots=True)
class CouplerRuntimeResources:
    """Mutable runtime-owned resources for one public coupler instance."""

    topology_maps: RuntimeTopologyMaps = field(
        default_factory=RuntimeTopologyMaps.empty
    )
    runtime_contracts: dict[str, ExchangeContract] = field(default_factory=dict)
    interrupt_controller: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController
    )


__all__ = ["CouplerRuntimeResources"]
