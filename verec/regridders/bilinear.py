from .base import Regridder
from ..fields import Field


class XESMFBilinear(Regridder):
    def prepare(
        self, reuse_weights: bool = False, extrap_method: str | None = "nearest_s2d"
    ) -> "XESMFBilinear":
        """Prepare the regridder by initializing xESMF with the source and destination grids."""

        try:
            import xesmf as xe
        except Exception as e:
            raise RuntimeError("xESMF adapter requires xesmf installed") from e

        self._define_src_dst_grids_and_masks()

        # Add an option to reuse weights if weights are precomputed and saved to a file
        self.regridder = xe.Regridder(
            self.ds_in,
            self.ds_out,
            method="bilinear",
            extrap_method=extrap_method,
            reuse_weights=reuse_weights,
        )

        return self

    def __call__(self, field: Field) -> Field:
        data = field.data
        out = self.regridder(data)
        return Field(
            name=field.name,
            data=out,
            grid=self.dst_grid,
            units=field.units,
            attrs=field.attrs,
        )
