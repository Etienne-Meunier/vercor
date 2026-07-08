"""Public regridding facade for VerCOR grid-to-grid transfers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from vercor.grids import RectilinearGrid
from vercor._regridders.bilinear import bilinear as _bilinear
from vercor._regridders.conservative import conservative as _conservative
from vercor.types import RuntimeArray


@runtime_checkable
class Regridder(Protocol):
    """Public protocol for grid-to-grid transfer objects."""

    source_grid: RectilinearGrid

    @property
    def target_grid(self) -> RectilinearGrid:
        """Return the destination grid using public target terminology."""

    @property
    def has_identical_grids(self) -> bool:
        """Return whether source and target grids are identical."""

    def regrid(self, field: RuntimeArray) -> RuntimeArray:
        """Transfer one scalar field to the target grid."""

    def regrid_vector(
        self,
        u: RuntimeArray,
        v: RuntimeArray,
    ) -> tuple[RuntimeArray, RuntimeArray]:
        """Transfer one vector field pair to the target grid."""


if TYPE_CHECKING:
    RegridderFactory: TypeAlias = Callable[..., Regridder]
else:

    @runtime_checkable
    class RegridderFactory(Protocol):
        """Public protocol for factories that build regridders for two grids."""

        def __call__(
            self,
            source_grid: RectilinearGrid,
            target_grid: RectilinearGrid,
            **kwargs: Any,
        ) -> Regridder:
            """Return a regridder configured for one source/target grid pair."""


def bilinear(
    source_grid: RectilinearGrid,
    target_grid: RectilinearGrid,
    *,
    periodic_longitude: bool = True,
    nan_renorm: bool = True,
    extrapolation_mode: str | None = None,
    idw_k: int = 8,
    idw_eps: float = 1e-12,
    fill_value: float = float("nan"),
) -> Regridder:
    """Build a public bilinear regridder for one source/target grid pair."""

    return _bilinear(
        source_grid,
        target_grid,
        periodic_longitude=periodic_longitude,
        nan_renorm=nan_renorm,
        extrapolation_mode=extrapolation_mode,
        idw_k=idw_k,
        idw_eps=idw_eps,
        fill_value=fill_value,
    )


def conservative(
    source_grid: RectilinearGrid,
    target_grid: RectilinearGrid,
    *,
    source_mask: RuntimeArray | None = None,
    normalize: str = "conservation",
    radius_km: float = 6371.0,
) -> Regridder:
    """Build a public conservative scalar regridder for one grid pair."""

    return _conservative(
        source_grid,
        target_grid,
        source_mask=source_mask,
        normalize=normalize,
        radius=radius_km,
    )


__all__ = [
    "Regridder",
    "RegridderFactory",
    "bilinear",
    "conservative",
]
