import abc
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


if TYPE_CHECKING:
    from vercor.regridders.bilinear import BilinearRectilinearRegridder


@dataclass
class Regridder(abc.ABC):
    source_grid: RectilinearGrid
    destination_grid: RectilinearGrid
    interpolator: Optional[BilinearRectilinearInterpolator] = None

    @abc.abstractmethod
    def setup(self) -> "BilinearRectilinearRegridder":
        raise NotImplementedError

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
