from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
from vercor.regridders.base import Regridder
from vercor.interpolators.bilinear_rectilinear import Bilinear


class XESMFBilinearRectilinear(Regridder):
    def prepare(self) -> "XESMFBilinearRectilinear":
        """Prepare the regridder by initializing xESMF with the source and destination grids."""

        try:
            import xesmf as xe
        except Exception as e:
            raise RuntimeError("xESMF adapter requires xesmf installed") from e

        self._define_rectilinear_src_dst_grids_and_masks()

        # Add an option to reuse weights if weights are precomputed and saved to a file
        self.regridder = xe.Regridder(
            self.field_in,
            self.field_out,
            method="bilinear",
            extrap_method="nearest_s2d",
        )

        return self

    def __call__(self, field: NDArray) -> NDArray:
        return self.regridder(field)


class BilinearRectilinear(Regridder):
    def prepare(self) -> "BilinearRectilinear":
        self._define_rectilinear_src_dst_grids_and_masks()

        self.regridder = Bilinear(
            self.field_in,
            self.field_out,
            periodic_longitude=True,
            nan_renorm=True,
            extrapolation_mode="nearest",
            idw_k=8,
            idw_eps=1e-12,
            fill_value=np.nan,
        )

        return self

    def __call__(self, *args, src_mask=None) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Call with positional args for fields and optional src_mask as a keyword-only arg.

        Supported calls:
          - __call__(scalar_src, src_mask=...) -> scalar interpolation
          - __call__(u_src, v_src, src_mask=...) -> vector interpolation

        src_mask must be provided as a keyword argument. Passing a mask as a positional
        second argument is not allowed and will raise a TypeError to avoid ambiguity.
        """
        if len(args) == 0:
            raise TypeError(
                "Must provide either scalar_src or (u_src, v_src) as positional arguments"
            )

        if len(args) == 1:
            scalar_src = args[0]
            return self.regridder.apply_scalar(scalar_src, src_mask=src_mask)

        if len(args) == 2:
            a0, a1 = args
            out = self.regridder.apply_vector(a0, a1, src_mask=src_mask)
            return (out[0], out[1])

        raise TypeError(
            "Too many positional arguments; provide either (scalar,) or (u_src, v_src)"
        )
