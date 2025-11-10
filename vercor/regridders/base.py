from typing import Any, Optional, Tuple
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


class Regridder:
    def __init__(
        self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.destination_grid = destination_grid
        self.interpolator: Optional[BilinearRectilinearInterpolator] = None

    @property
    def is_identical_shape(self) -> bool:
        return self.source_grid.shape == self.destination_grid.shape

    def _ensure_ready(self, args: Tuple[Any, ...]) -> None:
        if self.interpolator is None:
            raise ValueError("Regridder not properly set up")
        if len(args) not in (1, 2):
            raise TypeError("Provide scalar_src or (u_src, v_src) as positional args")

    def _apply_scalar(
        self, args: NDArray, src_mask: Optional[NDArray] = None
    ) -> NDArray:
        assert self.interpolator is not None
        result: NDArray = self.interpolator.apply_scalar(args, src_mask=src_mask)
        return result

    def _apply_vector(
        self, v0: NDArray, v1: NDArray, src_mask: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        assert self.interpolator is not None
        result: Tuple[NDArray, NDArray] = self.interpolator.apply_vector(
            v0, v1, src_mask=src_mask
        )
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}:"
            f"\n |Source grid:"
            f"\n |----Grid type: {self.source_grid.__class__.__name__} ({self.source_grid.name})"
            f"\n |----Grid shape: {self.source_grid.shape}"
            f"\n |Destination grid:"
            f"\n |----Grid type: {self.destination_grid.__class__.__name__} ({self.destination_grid.name})"
            f"\n |----Grid shape: {self.destination_grid.shape}"
        )
