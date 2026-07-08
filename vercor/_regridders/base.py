from typing import Any

from vercor.grids import RectilinearGrid
from vercor.grid_geometry import grids_identical


class _BaseRegridder:
    """Shared grid, interpolator, and display state for concrete regridders.

    Concrete subclasses own their call contracts because scalar/vector support
    differs by regridding method.
    """

    def __init__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
        *,
        interpolator: Any | None = None,
        has_identical_grids: bool | None = None,
    ) -> None:
        self.source_grid = source_grid
        self._target_grid = target_grid
        self._interpolator = interpolator
        self._has_identical_grids = (
            grids_identical(self.source_grid, self._target_grid)
            if has_identical_grids is None
            else has_identical_grids
        )

    @property
    def has_identical_grids(self) -> bool:
        """Return whether source and target grids are identical."""

        return self._has_identical_grids

    @property
    def target_grid(self) -> RectilinearGrid:
        """Return the destination grid using public target terminology."""

        return self._target_grid

    @property
    def interpolator(self) -> Any | None:
        """Return the concrete interpolator, or ``None`` for identity grids."""

        return self._interpolator

    def regrid(self, field: Any) -> Any:
        """Transfer one scalar field from the source grid to the target grid."""

        raise NotImplementedError

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        """Transfer one vector field pair from the source grid to the target grid."""

        raise NotImplementedError

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:"
            f"\n ├──Source grid:"
            f"\n │    ├──Grid type: {self.source_grid.__class__.__name__} ({self.source_grid.name})"
            f"\n │    └──Grid shape: {self.source_grid.shape}"
            f"\n └──Target grid:"
            f"\n      ├──Grid type: {self.target_grid.__class__.__name__} ({self.target_grid.name})"
            f"\n      └──Grid shape: {self.target_grid.shape}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source_grid={repr(self.source_grid)},"
            f" target_grid={repr(self.target_grid)})"
        )
