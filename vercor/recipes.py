"""Bundled exchange field recipes and plain coupler recipe objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.components import ComponentLike
from vercor.config import RuntimeOptions
from vercor.exchanges import Exchange

from vercor._exchange_recipes import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_LAND_STATE_FIELDS,
    ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS,
    ATMOSPHERE_TO_OCEAN_STATE_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    LAND_TO_ATMOSPHERE_SOIL_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    SEAICE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS,
)

if TYPE_CHECKING:
    from vercor.coupler import Coupler
    from vercor.jax_logging import LoggerLike


@dataclass(frozen=True)
class CouplerSpec:
    """Plain reusable recipe for constructing a configured coupler."""

    components: tuple[ComponentLike, ...]
    exchanges: tuple[Exchange, ...] = ()
    run_order: tuple[str, ...] = ()
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)

    def __init__(
        self,
        *,
        components: Sequence[ComponentLike],
        exchanges: Sequence[Exchange] = (),
        run_order: Sequence[str] = (),
        runtime: RuntimeOptions | None = None,
    ) -> None:
        """Create an immutable coupler recipe from public configuration objects."""

        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "exchanges", tuple(exchanges))
        object.__setattr__(self, "run_order", tuple(run_order))
        object.__setattr__(
            self,
            "runtime",
            RuntimeOptions() if runtime is None else runtime,
        )

    def build(
        self,
        clock: Clock,
        *,
        logger: "LoggerLike | None" = None,
        log_level: int | str = "INFO",
    ) -> "Coupler":
        """Build a coupler from this recipe and a concrete clock."""

        from vercor.coupler import Coupler

        return Coupler(
            clock,
            components=self.components,
            exchanges=self.exchanges,
            run_order=self.run_order,
            runtime=self.runtime,
            logger=logger,
            log_level=log_level,
        )


__all__ = [
    "ATMOSPHERE_TO_DATA_OCEAN_FIELDS",
    "ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS",
    "ATMOSPHERE_TO_LAND_BASIC_FIELDS",
    "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
    "ATMOSPHERE_TO_LAND_STATE_FIELDS",
    "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS",
    "ATMOSPHERE_TO_OCEAN_STATE_FIELDS",
    "ATMOSPHERE_TO_VEROS_FORCING_FIELDS",
    "CouplerSpec",
    "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS",
    "JCM_LAND_TO_ATMOSPHERE_FIELDS",
    "LAND_TO_ATMOSPHERE_SOIL_FIELDS",
    "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_SEAICE_SURFACE_FIELDS",
    "SEAICE_TO_OCEAN_FIELDS",
    "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS",
    "SLAB_ATMOSPHERE_TO_OCEAN_FIELDS",
    "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS",
]
