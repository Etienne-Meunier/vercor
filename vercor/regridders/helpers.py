from __future__ import annotations

from typing import Any, cast

from jax import Array, lax
import jax.numpy as jnp

from vercor.grid import RectilinearGrid


def make_rectilinear_grid(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask: Any | None = None,
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

    longitude = jnp.linspace(longitude_start, longitude_end, nlon, dtype=float)
    latitude = jnp.linspace(latitude_start, latitude_end, nlat, dtype=float)

    return RectilinearGrid(
        name=name, longitude=longitude, latitude=latitude, binary_mask=mask
    )


def centers_to_edges(centers: Any, grid_type: str) -> Any:
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
    centers = jnp.asarray(centers, dtype=jnp.float64)

    if centers.size < 2:
        half_width = 0.5
        return jnp.stack((centers[0] - half_width, centers[0] + half_width))

    inner_edges = 0.5 * (centers[:-1] + centers[1:])
    d_start = inner_edges[0] - centers[0]
    d_end = centers[-1] - inner_edges[-1]

    edge_start = centers[0] - d_start
    edge_end = centers[-1] + d_end

    edges = jnp.concatenate(
        (jnp.asarray([edge_start]), inner_edges, jnp.asarray([edge_end]))
    )

    if grid_type == "lat":
        edges = jnp.clip(edges, -90.0, 90.0)
    elif grid_type == "lon":
        span = edges[-1] - edges[0]

        def clamp_lon(overhanging_edges: Array) -> Array:
            return cast(
                Array,
                lax.cond(
                    jnp.min(overhanging_edges) < -5.0,
                    lambda value: jnp.clip(value, -180.0, 180.0),
                    lambda value: jnp.clip(value, 0.0, 360.0),
                    overhanging_edges,
                ),
            )

        edges = lax.cond(
            span > 360.0 + 1e-10,
            clamp_lon,
            lambda value: value,
            edges,
        )

    return cast(Any, edges)


def compute_land_mask(ocean_fractional_mask: Any) -> Any:
    """Compute land binary mask from ocean fractional mask with thresholding.
    The ocean_fractional_mask array is conservatively remapped from ocean grid to atmospheric/land grid.

    Arguments:
        ocean_fractional_mask: 2D array with values between 0 and 1 representing ocean fraction

    Returns:
        land_binary_mask: 2D array with 1 for land, 0 for ocean

    References:
        Adapted from CESM CPL7 source code
    """

    fminval = 0.001
    fmaxval = 1.0

    land_binary_mask = 1.0 - jnp.asarray(ocean_fractional_mask)
    land_binary_mask = jnp.where(land_binary_mask > fmaxval, 1.0, land_binary_mask)
    land_binary_mask = jnp.where(land_binary_mask < fminval, 0.0, land_binary_mask)

    return cast(Any, jnp.where(land_binary_mask != 0.0, 1, 0))
