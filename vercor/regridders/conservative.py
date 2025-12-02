from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid
from vercor.regridders.helpers import centers_to_edges
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.regridders.base import Regridder


class ConservativeRectilinearRegridder(Regridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        destination_grid: RectilinearGrid,
        source_mask: Optional[NDArray] = None,
        normalize: str = "conservation",  # 'conservation' | 'fracarea'
        fill_value: float = np.nan,
        radius: float = 6371.0,
    ) -> None:

        super().__init__(source_grid, destination_grid)

        if (
            source_grid.longitude_edges is not None
            and source_grid.latitude_edges is not None
        ):
            src_lon_bnds = source_grid.longitude_edges
            src_lat_bnds = source_grid.latitude_edges
        else:
            src_lon_bnds = centers_to_edges(self.source_grid.longitude, kind="lon")
            src_lat_bnds = centers_to_edges(self.source_grid.latitude, kind="lat")
            # TODO: Through a warning, to inform the user that bounds are being computed for source grid
            # print(f"Source Longitude Edges (Start/End): {src_lon_bnds[0]:.2f}, {src_lon_bnds[-1]:.2f}")
            # print(f"Source Latitude Edges (Start/End): {src_lat_bnds[0]:.2f}, {src_lat_bnds[-1]:.2f}")

        if (
            destination_grid.longitude_edges is not None
            and destination_grid.latitude_edges is not None
        ):
            dst_lon_bnds = destination_grid.longitude_edges
            dst_lat_bnds = destination_grid.latitude_edges
        else:
            dst_lon_bnds = centers_to_edges(self.destination_grid.longitude, kind="lon")
            dst_lat_bnds = centers_to_edges(self.destination_grid.latitude, kind="lat")
            # TODO: Through a warning, to inform the user that bounds are being computed for destination grid
            # print(f"Destination Longitude Edges (Start/End): {dst_lon_bnds[0]:.2f}, {dst_lon_bnds[-1]:.2f}")
            # print(f"Destination Latitude Edges (Start/End): {dst_lat_bnds[0]:.2f}, {dst_lat_bnds[-1]:.2f}")

        self.interpolator = ConservativeRectilinearRemapper(
            src_lon_bnds=src_lon_bnds,
            src_lat_bnds=src_lat_bnds,
            dst_lon_bnds=dst_lon_bnds,
            dst_lat_bnds=dst_lat_bnds,
            src_mask=source_mask,
            normalize=normalize,
            radius=radius,
        )

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
