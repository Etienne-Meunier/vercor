import numpy as np
from verec.regridders.base import Regridder
from verec.regridders.interpolators.bilinear import Bilinear


class XESMFBilinear(Regridder):
    def prepare(
        self, reuse_weights: bool = False, extrap_method: str | None = "nearest_s2d"
    ) -> "XESMFBilinear":
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
            extrap_method=extrap_method,
            reuse_weights=reuse_weights,
        )

        return self


class BilinearRectilinear(Regridder):
    def prepare(
        self,
        reuse_weights: bool = False,
        extrap_method: str | None = "nearest",
        periodic_longitude=True,
        nan_renorm=True,
        idw_k=8,
        idw_eps=1e-12,
        fill_value=np.nan
    ) -> "BilinearRectilinear":

        self._define_rectilinear_src_dst_grids_and_masks()

        self.regridder = Bilinear(
            self.field_in,
            self.field_out,
            periodic_longitude=periodic_longitude,
            nan_renorm=nan_renorm,
            extrapolation_mode=extrap_method,
            idw_k=idw_k,
            idw_eps=idw_eps,
            fill_value=fill_value,
        )

        return self
