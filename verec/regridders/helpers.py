from typing import Any, Tuple
import numpy as np
from verec.grid import RectilinearGrid


def make_rectilinear_grid(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask=None,
    area=None,
) -> RectilinearGrid:
    """Helper to build rectilinear grid"""

    longitude = np.linspace(longitude_start, longitude_end, nlon, dtype=float)
    latitude = np.linspace(latitude_start, latitude_end, nlat, dtype=float)

    return RectilinearGrid(
        name=name, longitude=longitude, latitude=latitude, mask=mask, area=area
    )


def _wrap_like(lon_deg: np.ndarray, base0_deg: float) -> np.ndarray:
    """
    Map longitudes (deg) into the [base0, base0+360) interval.
    """
    return base0_deg + np.mod(lon_deg - base0_deg, 360.0)


def _unit_east_north(
    lon_rad: np.ndarray, lat_rad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return unit vectors (east, north) in 3-D for given lon/lat (radians).
    east = d r / d lon normalized; north = d r / d lat normalized.
    east is independent of latitude on the unit sphere.
    Shapes follow broadcasting of lon_rad/lat_rad.
    """
    slon, clon = np.sin(lon_rad), np.cos(lon_rad)
    slat, clat = np.sin(lat_rad), np.cos(lat_rad)

    # east: (-sin lon, cos lon, 0)
    e_east = np.stack((-slon, clon, np.zeros_like(lon_rad)), axis=-1)

    # north: (-sin lat cos lon, -sin lat sin lon, cos lat)
    e_north = np.stack((-slat * clon, -slat * slon, clat), axis=-1)
    return e_east, e_north


def _great_circle_distance_rad(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> Any:
    """
    Great-circle distance (radians) between points (supports broadcasting).
    Haversine, numerically stable.
    """
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sdlat2 = np.sin(dlat * 0.5)
    sdlon2 = np.sin(dlon * 0.5)
    a = sdlat2 * sdlat2 + np.cos(lat1) * np.cos(lat2) * sdlon2 * sdlon2
    # Clamp for safety
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
