import numpy as np
from numpy.typing import NDArray

from vercor.grid import RectilinearGrid


def make_rectilinear_grid(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask=None,
) -> RectilinearGrid:
    """Helper to build rectilinear grid"""

    longitude = np.linspace(longitude_start, longitude_end, nlon, dtype=float)
    latitude = np.linspace(latitude_start, latitude_end, nlat, dtype=float)

    return RectilinearGrid(
        name=name, longitude=longitude, latitude=latitude, binary_mask=mask
    )


def centers_to_bounds(centers: NDArray, kind: str = "lat") -> NDArray:
    """
    Convert grid centers to grid boundaries (edges).
    Smartly handles clamping:
    - Latitude: Always clamped to [-90, 90].
    - Longitude: Clamped only if bounds exceed 360-degree span (redundancy).
    Otherwise preserves wrapping edges (e.g. -182.5) for periodicity.
    """
    centers = np.asarray(centers, dtype=np.float64)

    if len(centers) < 2:
        half_width = 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width])

    inner_edges = 0.5 * (centers[:-1] + centers[1:])
    d_start = inner_edges[0] - centers[0]
    d_end = centers[-1] - inner_edges[-1]

    bound_start = centers[0] - d_start
    bound_end = centers[-1] + d_end

    bounds: NDArray = np.concatenate(([bound_start], inner_edges, [bound_end]))

    if kind == "lat":
        # Latitude must strictly be within physical poles
        bounds = np.clip(bounds, -90.0, 90.0)
    elif kind == "lon":
        # Check total span
        span = bounds[-1] - bounds[0]

        # Only clamp if the grid defines REDUNDANT coverage (e.g. 0 to 360 centers -> 370 span)
        # If span is ~360, it's a periodic grid; we keep the 'overhanging' edges
        # (e.g. -182.5) so they can wrap around to 177.5 in the overlap check.
        if span > 360.0 + 1e-10:
            if np.min(bounds) < -5.0:
                bounds = np.clip(bounds, -180.0, 180.0)
            else:
                bounds = np.clip(bounds, 0.0, 360.0)

    return bounds
