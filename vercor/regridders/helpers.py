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
    """
    Helper to build rectilinear grid with equally spaced coordinates.

    Arguments:
        name: grid name
        nlon: number of longitude points
        nlat: number of latitude points
        longitude_start: starting longitude value (degrees)
        longitude_end: ending longitude value (degrees)
        latitude_start: starting latitude value (degrees)
        latitude_end: ending latitude value (degrees)
        mask: optional binary mask (2D array with shape (nlat, nlon))

    Returns:
        RectilinearGrid instance
    """

    longitude = np.linspace(longitude_start, longitude_end, nlon, dtype=float)
    latitude = np.linspace(latitude_start, latitude_end, nlat, dtype=float)

    return RectilinearGrid(
        name=name, longitude=longitude, latitude=latitude, binary_mask=mask
    )


def centers_to_edges(centers: NDArray, grid_type: str) -> NDArray:
    """
    Convert grid centers to grid boundaries (edges).
    Smartly handles clamping:
    - Latitude: Always clamped to [-90, 90].
    - Longitude: Clamped only if edges exceed 360-degree span (redundancy).
    Otherwise preserves wrapping edges (e.g. -182.5) for periodicity.

    Arguments:
        centers: 1D array of grid cell centers
        kind: 'lat' for latitude edges, 'lon' for longitude edges

    Returns:
        1D array of grid cell edges
    """
    centers = np.asarray(centers, dtype=np.float64)

    if len(centers) < 2:
        half_width = 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width])

    inner_edges = 0.5 * (centers[:-1] + centers[1:])
    d_start = inner_edges[0] - centers[0]
    d_end = centers[-1] - inner_edges[-1]

    edge_start = centers[0] - d_start
    edge_end = centers[-1] + d_end

    edges: NDArray = np.concatenate(([edge_start], inner_edges, [edge_end]))

    if grid_type == "lat":
        # Latitude must strictly be within physical poles
        edges = np.clip(edges, -90.0, 90.0)
    elif grid_type == "lon":
        # Check total span
        span = edges[-1] - edges[0]

        # Only clamp if the grid defines REDUNDANT coverage (e.g. 0 to 360 centers -> 370 span)
        # If span is ~360, it's a periodic grid; we keep the 'overhanging' edges
        # (e.g. -182.5) so they can wrap around to 177.5 in the overlap check.
        if span > 360.0 + 1e-10:
            if np.min(edges) < -5.0:
                edges = np.clip(edges, -180.0, 180.0)
            else:
                edges = np.clip(edges, 0.0, 360.0)

    return edges


def compute_land_mask(ocean_fractional_mask: NDArray) -> NDArray:
    """Compute land binary mask from ocean fractional mask with thresholding.
    The ocean_fractional_mask array is conservatively remapped from ocean grid to atmospheric/land grid.

    Arguments:
        ocean_fractional_mask: 2D array with values between 0 and 1 representing ocean fraction

    Returns:
        land_binary_mask: 2D array with 1 for land, 0 for ocean

    References:
        Adapted from CESM CPL7 source code
    """

    FMINVAL = 0.001
    FMAXVAL = 1.0

    land_binary_mask = 1.0 - ocean_fractional_mask
    land_binary_mask = np.where(land_binary_mask > FMAXVAL, 1.0, land_binary_mask)
    land_binary_mask = np.where(land_binary_mask < FMINVAL, 0.0, land_binary_mask)

    land_binary_mask = np.where(land_binary_mask != 0.0, 1, 0)

    return land_binary_mask
