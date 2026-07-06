"""Public grid constructors and grid types."""

from typing import Any

from vercor._grid import RectilinearGrid
from vercor.grid_geometry import make_rectilinear_grid


def rectilinear_grid(
    name: str,
    *,
    nlon: int,
    nlat: int,
    longitude: tuple[float, float],
    latitude: tuple[float, float],
    binary_mask: Any | None = None,
) -> RectilinearGrid:
    """Build a rectilinear grid with equally spaced coordinate centers."""

    return make_rectilinear_grid(
        name,
        nlon,
        nlat,
        longitude[0],
        longitude[1],
        latitude[0],
        latitude[1],
        mask=binary_mask,
    )


__all__ = ["RectilinearGrid", "rectilinear_grid"]
