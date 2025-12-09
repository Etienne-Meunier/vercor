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


def compute_1d_overlap_rectilinear(
    target_edges: NDArray, source_edges: NDArray
) -> NDArray:
    """
    Computes the intersection length between target and source intervals.

    Arguments:
        target_edges: 1D array of target grid cell edges (length n_target + 1)
        source_edges: 1D array of source grid cell edges (length n_source + 1)

    Returns:
        (n_target, n_source) weights matrix
    """

    # Target intervals [t_min, t_max]
    t0 = target_edges[:-1, None]
    t1 = target_edges[1:, None]
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)

    # Source intervals [s_min, s_max]
    s0 = source_edges[None, :-1]
    s1 = source_edges[None, 1:]
    s_min = np.minimum(s0, s1)
    s_max = np.maximum(s0, s1)

    # Intersection
    inter_min = np.maximum(t_min, s_min)
    inter_max = np.minimum(t_max, s_max)

    out: NDArray = np.maximum(0, inter_max - inter_min)

    return out


def compute_grid_fraction_rectilinear(
    tgt_lat_edges: NDArray,
    tgt_lon_edges: NDArray,
    src_lat_edges: NDArray,
    src_lon_edges: NDArray,
    src_mask: NDArray,
) -> NDArray:
    """
    Calculates the fraction of Target cells covered by valid Source cells.

    Arguments:
        tgt_lat_edges, tgt_lon_edges : 1D arrays of Target cell edges.
        src_lat_edges, src_lon_edges : 1D arrays of Source cell edges.
        src_mask : 2D boolean array (Lat, Lon) matching src dimensions.
                   True = Masked/Invalid (e.g. Land in an Ocean grid).
                   False = Valid Data.

    Returns:
        2D array of fractions (Lat, Lon) for Target grid cells.
    """

    # 1. Latitude Weights (Area weighted by sin(lat))
    # Clip to valid range to avoid sin domain errors
    tgt_lat_e_rad = np.deg2rad(np.clip(tgt_lat_edges, -90, 90))
    src_lat_e_rad = np.deg2rad(np.clip(src_lat_edges, -90, 90))

    tgt_sin = np.sin(tgt_lat_e_rad)
    src_sin = np.sin(src_lat_e_rad)

    # Calculate overlap in sin-space
    w_lat = compute_1d_overlap_rectilinear(tgt_sin, src_sin)
    # w_lat shape: (n_tgt_lat, n_src_lat)

    # 2. Longitude Weights (Cyclic Handling)
    # We compute overlap for Source, Source-360, and Source+360
    # This handles -180/180 vs 0/360 mismatches automatically.
    w_lon_0 = compute_1d_overlap_rectilinear(tgt_lon_edges, src_lon_edges)
    w_lon_m = compute_1d_overlap_rectilinear(tgt_lon_edges, src_lon_edges - 360.0)
    w_lon_p = compute_1d_overlap_rectilinear(tgt_lon_edges, src_lon_edges + 360.0)

    w_lon = w_lon_0 + w_lon_m + w_lon_p
    # w_lon shape: (n_tgt_lon, n_src_lon)

    # 3. Calculate Total Area of Target Cells
    # Area ~ d(sin_lat) * d(lon)
    d_sin_tgt = np.abs(np.diff(tgt_sin))
    d_lon_tgt = np.diff(tgt_lon_edges)
    tgt_area = np.outer(d_sin_tgt, d_lon_tgt)  # (n_tgt_lat, n_tgt_lon)

    # 4. Aggregate Valid Areas
    # We only want areas where src_mask is FALSE (Valid data)
    valid_src = (~src_mask).astype(np.float64)

    # Einstein Summation:
    # i: tgt_lat, j: tgt_lon
    # k: src_lat, l: src_lon
    # w_lat[i,k] * w_lon[j,l] * valid_src[k,l] -> Result[i,j]
    covered_area = np.einsum("ik, jl, kl -> ij", w_lat, w_lon, valid_src)

    # 5. Compute Fractions
    # Fraction = Covered Area / Total Cell Area
    fractions = covered_area / tgt_area

    # Clip to [0, 1] to handle floating point noise
    output: NDArray = np.clip(fractions, 0.0, 1.0)

    return output


def compute_land_mask(ocean_binary_mask: NDArray) -> NDArray:
    """Compute land binary mask from ocean binary mask with thresholding.
    The ocean_binary_mask array is conservatively remapped from ocean grid to land grid.

    Arguments:
        ocean_binary_mask: 2D array with 1 for ocean, 0 for land

    Returns:
        land_binary_mask: 2D array with 1 for land, 0 for ocean

    References:
        Adapted from CESM CPL7 source code
    """
    FMINVAL = 0.001
    FMAXVAL = 1.0

    land_binary_mask = 1.0 - ocean_binary_mask
    land_binary_mask = np.where(land_binary_mask > FMAXVAL, 1.0, land_binary_mask)
    land_binary_mask = np.where(land_binary_mask < FMINVAL, 0.0, land_binary_mask)

    land_binary_mask = np.where(land_binary_mask != 0.0, 1, 0)

    return land_binary_mask
