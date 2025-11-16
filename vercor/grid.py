import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Grid(abc.ABC):
    name: str
    binary_mask: Optional[NDArray] = None  # values of 1 for active, 0 for inactive

    def __post_init__(self) -> None:
        if self.binary_mask is not None and self.binary_mask.ndim != 2:
            raise ValueError("Mask must be a 2D array.")

    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Grid name:  {self.name}\n"
            f"├── Grid shape: {self.shape}\n"
            f"└── Binary mask: {'Provided' if self.binary_mask is not None else 'Not provided'}\n"
        )


class RectilinearGrid(Grid):
    def __init__(
        self,
        name: str,
        longitude: NDArray,
        latitude: NDArray,
        binary_mask: Optional[NDArray] = None,
    ) -> None:
        super().__init__(name=name, binary_mask=binary_mask)
        self.longitude = longitude
        self.latitude = latitude

        if self.longitude.ndim != 1 or self.latitude.ndim != 1:
            raise ValueError(
                "RectilinearGrid expects both longitude and latitude coordinates to be 1D arrays."
            )
        if not (
            np.all(np.diff(self.longitude) > 0) and np.all(np.diff(self.latitude) > 0)
        ):
            # Monotonic increasing required for built-in regridders.
            raise ValueError("longitude and latitude must be strictly monotonic.")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.latitude.size, self.longitude.size)  # (nlat, nlon), row-major
