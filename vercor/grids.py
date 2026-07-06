"""Public grid constructors and grid types."""

from typing import Any
import warnings

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


def rectilinear(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask: Any | None = None,
) -> RectilinearGrid:
    """Build a rectilinear grid with the deprecated v0.2 constructor shape."""

    warnings.warn(
        "vercor.grids.rectilinear(...) is deprecated; use "
        "rectilinear_grid(name, nlon=..., nlat=..., lon=(...), lat=(...)) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rectilinear_grid(
        name,
        nlon=nlon,
        nlat=nlat,
        lon=(longitude_start, longitude_end),
        lat=(latitude_start, latitude_end),
        mask=mask,
    )


__all__ = ["RectilinearGrid", "rectilinear", "rectilinear_grid"]
