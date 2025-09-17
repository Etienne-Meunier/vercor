from verec.regridders.base import Regridder
from verec.fields import Field


class XESMFConservative_normed(Regridder):
    def prepare(
        self, reuse_weights: bool = False, extrap_method: str | None = None
    ) -> "XESMFConservative_normed":
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
            method="conservative_normed",
            reuse_weights=reuse_weights,
            extrap_method=extrap_method,
        )

        return self
