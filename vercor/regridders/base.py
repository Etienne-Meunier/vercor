import abc
from dataclasses import dataclass
from typing import Any, Optional

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import Bilinear


@dataclass
class Regridder(abc.ABC):
    src_grid: RectilinearGrid
    dst_grid: RectilinearGrid
    interpolator: Optional[Bilinear] = None

    @abc.abstractmethod
    def setup(self) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n src_grid={self.src_grid},\n dst_grid={self.dst_grid})"
