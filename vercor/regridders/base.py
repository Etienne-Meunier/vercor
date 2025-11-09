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
        return f"{self.__class__.__name__}(\n source_grid={self.source_grid},\n destination_grid={self.destination_grid})"
