import abc
from dataclasses import dataclass
from typing import Any, Optional

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import Bilinear


@dataclass
class Regridder(abc.ABC):
    src_grid: RectilinearGrid
    dst_grid: RectilinearGrid
    regridder: Optional[Bilinear] = None

    @abc.abstractmethod
    def prepare(self) -> Any:
        raise NotImplementedError

    def _define_source_destination_grids_and_masks(self) -> None:
        longitude_src, latitude_src = self.src_grid.longitude, self.src_grid.latitude
        src_mask = self.src_grid.mask
        longitude_dst, latitude_dst = self.dst_grid.longitude, self.dst_grid.latitude
        dst_mask = self.dst_grid.mask

        self.field_in = {"lat": latitude_src, "lon": longitude_src}
        self.field_out = {"lat": latitude_dst, "lon": longitude_dst}

        # 0 for invalid points, 1 for valid points
        if src_mask is not None:
            self.field_in["mask"] = src_mask
        if dst_mask is not None:
            self.field_out["mask"] = dst_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n src_grid={self.src_grid},\n dst_grid={self.dst_grid})"
