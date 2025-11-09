from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
from vercor.regridders.base import Regridder
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


class BilinearRectilinearRegridder(Regridder):
    def setup(self) -> "BilinearRectilinearRegridder":

        self.interpolator = BilinearRectilinearInterpolator(
            self.source_grid.longitude,
            self.source_grid.latitude,
            self.destination_grid.longitude,
            self.destination_grid.latitude,
            tgt_mask=self.destination_grid.mask,
            periodic_longitude=True,
            nan_renorm=True,
            extrapolation_mode="idw",
            idw_k=8,
            idw_eps=1e-12,
            fill_value=np.nan,
        )

        return self

    def __call__(self, *args, src_mask=None) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Call with positional args for fields and optional src_mask as a keyword-only arg.

        Supported calls:
          - apply(scalar_src, src_mask=...) -> scalar interpolation
          - apply(u_src, v_src, src_mask=...) -> vector interpolation

        src_mask must be provided as a keyword argument. Passing a mask as a positional
        second argument is not allowed and will raise a TypeError to avoid ambiguity.
        """

        out: Union[NDArray, Tuple[NDArray, NDArray]]

        if self.interpolator is None:
            raise ValueError("Regridder not properly set up; call setup() before using")

        if len(args) == 0:
            raise TypeError(
                "Must provide either scalar_src or (u_src, v_src) as positional arguments"
            )

        if len(args) == 1:
            scalar_src = args[0]
            out = self.interpolator.apply_scalar(scalar_src, src_mask=src_mask)

        elif len(args) == 2:
            v0, v1 = args
            out = self.interpolator.apply_vector(v0, v1, src_mask=src_mask)
        else:
            raise TypeError(
                "Must provide either scalar_src or (u_src, v_src) as positional arguments"
            )

        return out
