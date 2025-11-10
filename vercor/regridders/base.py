from typing import Optional

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


class Regridder:
    def __init__(
        self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.destination_grid = destination_grid
        self.interpolator: Optional[BilinearRectilinearInterpolator] = None

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
