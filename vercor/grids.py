"""Public grid constructors and grid types."""

from typing import Any

from vercor.grid import RectilinearGrid
from vercor.grid_geometry import make_rectilinear_grid


def rectilinear_grid(
    name: str,
    *,
    nlon: int,
    nlat: int,
    lon: tuple[float, float],
    lat: tuple[float, float],
    mask: Any | None = None,
) -> RectilinearGrid:
    """Build a rectilinear grid with equally spaced coordinate centers."""

    return make_rectilinear_grid(
        name,
        nlon,
        nlat,
        lon[0],
        lon[1],
        lat[0],
        lat[1],
        mask=mask,
    )


__all__ = ["RectilinearGrid", "rectilinear_grid"]
