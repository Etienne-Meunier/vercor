import abc
from dataclasses import dataclass
from typing import Any
from ..fields import Field


@dataclass
class Regridder(abc.ABC):
    src_grid: Any
    src_mask: Any
    dst_grid: Any
    dst_mask: Any

    @abc.abstractmethod
    def prepare(self, reuse_weights: bool, extrap_method: str):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, field: Field) -> Field:
        raise NotImplementedError

    def _define_src_dst_grids_and_masks(self) -> None:
        xs, ys = self.src_grid.x, self.src_grid.y
        xd, yd = self.dst_grid.x, self.dst_grid.y
        self.ds_in = {"lat": ys, "lon": xs}
        self.ds_out = {"lat": yd, "lon": xd}

        # 0 for invalid points, 1 for valid points
        if hasattr(self, "src_mask") and self.src_mask is not None:
            self.ds_in["mask"] = self.src_mask
        if hasattr(self, "dst_mask") and self.dst_mask is not None:
            self.ds_out["mask"] = self.dst_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n src_grid={self.src_grid},\n dst_grid={self.dst_grid})"
