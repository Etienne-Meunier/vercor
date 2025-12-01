from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.regridders.base import Regridder


class BilinearRectilinearRegridder(Regridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        destination_grid: RectilinearGrid,
        periodic_longitude: bool = True,
        nan_renorm: bool = False,
        extrapolation_mode: str | None = None,  # 'nearest' | 'idw'
        idw_k: int = 8,
        idw_eps: float = 1e-12,
        fill_value: float = np.nan,
    ) -> None:

        super().__init__(source_grid, destination_grid)

        self.interpolator = BilinearRectilinearInterpolator(
            self.source_grid.longitude,
            self.source_grid.latitude,
            self.destination_grid.longitude,
            self.destination_grid.latitude,
            src_mask=self.source_grid.binary_mask,
            tgt_mask=self.destination_grid.binary_mask,
            periodic_longitude=periodic_longitude,
            nan_renorm=nan_renorm,
            extrapolation_mode=extrapolation_mode,
            idw_k=idw_k,
            idw_eps=idw_eps,
            fill_value=fill_value,
        )

    def __call__(
        self,
        *args: NDArray,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Supported calls:
          - apply(scalar_src) -> scalar interpolation
          - apply(u_src, v_src) -> vector interpolation
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
