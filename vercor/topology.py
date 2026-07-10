"""Public topology policy contracts for optional exchange-map patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from vercor.components.contracts import ComponentInfo
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.exchanges import Exchange
    from vercor.jax_logging import LoggerLike
    from vercor.settings import Settings


ExchangeKey: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True)
class TopologyContext:
    """Public read-only context supplied to topology policies."""

    components: Mapping[str, ComponentInfo]
    exchanges: Sequence["Exchange"]
    exchange_keys: Sequence[ExchangeKey]
    settings: "Settings"
    logger: "LoggerLike"


@dataclass(frozen=True)
class ExchangeTopologyPatch:
    """Topology mask updates keyed by ``(source, target, regrid_key)``."""

    binary_masks: Mapping[ExchangeKey, RuntimeArray] = field(default_factory=dict)
    fractional_masks: Mapping[ExchangeKey, RuntimeArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze caller-provided patch mappings."""

        object.__setattr__(
            self, "binary_masks", MappingProxyType(dict(self.binary_masks))
        )
        object.__setattr__(
            self,
            "fractional_masks",
            MappingProxyType(dict(self.fractional_masks)),
        )


@runtime_checkable
class TopologyPolicy(Protocol):
    """Public protocol for setup-specific exchange-topology customization."""

    def applies(self, context: TopologyContext) -> bool:
        """Return whether this policy should patch the configured topology."""
        ...

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Return exchange mask updates for this topology."""
        ...


@dataclass(frozen=True)
class SurfaceMaskPolicy:
    """Policy for the bundled atmosphere/ocean/land surface-mask topology."""

    mode: Literal["auto", "required", "disabled"] = "auto"
    atmosphere: str = "ATM"
    ocean: str = "OCN"
    land: str = "LND"

    def __post_init__(self) -> None:
        """Validate surface-mask policy values."""

        if self.mode not in ("auto", "required", "disabled"):
            raise ValueError("mode must be one of 'auto', 'required', 'disabled'")
        for role, name in (
            ("atmosphere", self.atmosphere),
            ("ocean", self.ocean),
            ("land", self.land),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"{role} component name must be a non-empty string")

    def applies(self, context: TopologyContext) -> bool:
        """Return whether the configured run should use bundled surface masks."""

        from vercor._runtime.surface_masks import should_apply_surface_mask_policy

        return should_apply_surface_mask_policy(
            context.components,
            context.exchanges,
            self,
        )

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Return exchange mask updates for bundled surface exchanges."""

        from vercor._runtime.surface_masks import build_surface_mask_topology_patch

        return build_surface_mask_topology_patch(context, self)


__all__ = [
    "ExchangeKey",
    "ExchangeTopologyPatch",
    "SurfaceMaskPolicy",
    "TopologyContext",
    "TopologyPolicy",
]
