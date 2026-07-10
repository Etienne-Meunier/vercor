from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vercor.types import RuntimeArray


@dataclass(frozen=True, slots=True)
class RuntimeTopologyMaps:
    """Read-only grouped exchange topology maps used by runtime dispatch."""

    regridders: Mapping[tuple[str, str, str], Any]
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray]
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray]

    def __post_init__(self) -> None:
        """Copy and freeze topology mappings at the runtime boundary."""

        object.__setattr__(
            self,
            "regridders",
            MappingProxyType(dict(self.regridders)),
        )
        object.__setattr__(
            self,
            "binary_masks",
            MappingProxyType(dict(self.binary_masks)),
        )
        object.__setattr__(
            self,
            "fractional_masks",
            MappingProxyType(dict(self.fractional_masks)),
        )

    @classmethod
    def empty(cls) -> "RuntimeTopologyMaps":
        """Return an empty grouped topology-map bundle."""

        return cls(
            regridders={},
            binary_masks={},
            fractional_masks={},
        )


@dataclass(frozen=True)
class ExchangeTopologyState:
    """Prepared exchange topology maps."""

    topology_maps: RuntimeTopologyMaps


__all__ = [
    "ExchangeTopologyState",
    "RuntimeTopologyMaps",
]
