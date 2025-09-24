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
    longitude: Optional[np.ndarray] = None  # 1D centers
    latitude: Optional[np.ndarray] = None  # 1D centers

    def __post_init__(self) -> None:
        if self.longitude is None or self.latitude is None:
            raise ValueError(
                "longitude and latitude must not be None for RectilinearGrid."
            )
        if self.longitude.ndim != 1 or self.latitude.ndim != 1:
            raise ValueError(
                "RectilinearGrid expects 1D longitude and latitude coordinate arrays (centers)."
            )
        if not (
            np.all(np.diff(self.longitude) > 0) and np.all(np.diff(self.latitude) > 0)
        ):
            # Monotonic increasing required for built-in regridders.
            raise ValueError("longitude and latitude must be strictly increasing.")

    @property
    def shape(self) -> tuple[int, int]:
        if self.longitude is None or self.latitude is None:
            raise ValueError(
                "longitude and latitude must not be None to determine shape."
            )
        return (self.latitude.size, self.longitude.size)  # (ny, nx), row-major
