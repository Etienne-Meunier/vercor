import abc
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Grid(abc.ABC):
    name: str
    mask: Optional[np.ndarray] = None  # 1 for active, 0 for inactive
    area: Optional[np.ndarray] = None  # cell areas if known

    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError


@dataclass
class RectilinearGrid(Grid):
    x: Optional[np.ndarray] = None  # 1D centers or 2D centers
    y: Optional[np.ndarray] = None  # 1D centers or 2D centers

    def __post_init__(self) -> None:
        if self.x is None or self.y is None:
            raise ValueError("x and y must not be None for RectilinearGrid.")
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError(
                "RectilinearGrid expects 1D x and y coordinate arrays (centers)."
            )
        if not (np.all(np.diff(self.x) > 0) and np.all(np.diff(self.y) > 0)):
            # Monotonic increasing required for built-in regridders.
            raise ValueError("x and y must be strictly increasing.")

    @property
    def shape(self) -> tuple[int, int]:
        if self.x is None or self.y is None:
            raise ValueError("x and y must not be None to determine shape.")
        return (self.y.size, self.x.size)  # (ny, nx), row-major
