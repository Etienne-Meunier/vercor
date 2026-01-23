from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from vercor.exceptions import RegridderError
from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)


class Regridder:
    def __init__(
        self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.destination_grid = destination_grid
        self.interpolator: Union[
            BilinearRectilinearInterpolator, ConservativeRectilinearRemapper, None
        ] = None

    @property
    def has_identical_grids(self) -> bool:
        """Check if source and destination grids are identical in shape and coordinates."""

        source = self.source_grid
        destination = self.destination_grid

        shape_condition = source.shape == destination.shape

        coord_condition = np.array_equal(
            source.latitude, destination.latitude
        ) and np.array_equal(source.longitude, destination.longitude)

        return shape_condition and coord_condition

    def _ensure_ready(self, args: Tuple[Any, ...]) -> None:
        """
        Ensure that the Regridder is properly set up before applying interpolation.
        Checks if the interpolator is initialized and if the correct number of arguments
        are provided (either one for scalar fields or two for vector fields)."""

        if self.interpolator is None:
            raise RegridderError("Regridder not properly set up")
        if len(args) not in (1, 2):
            raise TypeError("Provide scalar_src or (u_src, v_src) as positional args")

    def _apply_scalar(self, args: NDArray) -> NDArray:
        """A wrapper to call scalar interpolation."""
        assert self.interpolator is not None
        result: NDArray = self.interpolator.apply_scalar(args)
        return result

    def _apply_vector(self, v0: NDArray, v1: NDArray) -> Tuple[NDArray, NDArray]:
        """A wrapper to call vector interpolation."""
        assert self.interpolator is not None
        result: Tuple[NDArray, NDArray] = self.interpolator.apply_vector(v0, v1)
        return result

    def __call__(
        self,
        *args: NDArray,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Supported calls:
          - apply(scalar_src) -> scalar interpolation
        """

        self._ensure_ready(args)

        # Check if components have identical grids internally and
        # returns fields as-is (from source to destination) if so,
        # avoiding unnecessary computation
        if self.has_identical_grids:
            return args if len(args) == 2 else args[0]

        handlers: dict[int, Callable[..., NDArray | Tuple[NDArray, NDArray]]] = {
            1: self._apply_scalar,
            2: self._apply_vector,
        }

        return handlers[len(args)](*args)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:"
            f"\n ├──Source grid:"
            f"\n │    ├──Grid type: {self.source_grid.__class__.__name__} ({self.source_grid.name})"
            f"\n │    └──Grid shape: {self.source_grid.shape}"
            f"\n └──Destination grid:"
            f"\n      ├──Grid type: {self.destination_grid.__class__.__name__} ({self.destination_grid.name})"
            f"\n      └──Grid shape: {self.destination_grid.shape}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source_grid={repr(self.source_grid)},"
            f" destination_grid={repr(self.destination_grid)})"
        )
