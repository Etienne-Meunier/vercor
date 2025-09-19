import abc
from dataclasses import dataclass
from typing import Any
from verec.fields import Field


@dataclass
class Regridder(abc.ABC):
    src_grid: Any
    src_mask: Any
    dst_grid: Any
    dst_mask: Any
    regridder: Any = None

    @abc.abstractmethod
    def prepare(self) -> "Regridder":
        raise NotImplementedError

    def _define_rectilinear_src_dst_grids_and_masks(self) -> None:
        longitude_src, latitude_src = self.src_grid.longitude, self.src_grid.latitude
        longitude_dst, latitude_dst = self.dst_grid.longitude, self.dst_grid.latitude
        self.field_in = {"lat": latitude_src, "lon": longitude_src}
        self.field_out = {"lat": latitude_dst, "lon": longitude_dst}

        # 0 for invalid points, 1 for valid points
        if hasattr(self, "src_mask") and self.src_mask is not None:
            self.field_in["mask"] = self.src_mask
        if hasattr(self, "dst_mask") and self.dst_mask is not None:
            self.field_out["mask"] = self.dst_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n src_grid={self.src_grid},\n dst_grid={self.dst_grid})"
