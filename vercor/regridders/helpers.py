from typing import Dict, Tuple
import numpy as np
from vercor.grid import RectilinearGrid
from vercor.regridders.base import Regridder


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


def _scalar_field_interpolate(
    field_name: str,
    source_fields: Dict[str, np.ndarray],
    regridder: Regridder,
) -> np.ndarray:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for scalar field interpolation")

    return regridder(source_fields[field_name])


def _vector_field_interpolate(
    field_name: Tuple[str, str],
    source_fields: Dict[str, np.ndarray],
    regridder: Regridder,
) -> Tuple[np.ndarray, np.ndarray]:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for vector field interpolation")
    if len(field_name) == 2:
        field_name1, field_name2 = field_name
    else:
        raise ValueError("Vector field name must be a tuple of two strings")
    try:
        return regridder(source_fields[field_name1], source_fields[field_name2])
    except Exception as e:
        raise TypeError(
            "Regridder for vector fields must accept two arguments and return a tuple of two arrays"
        ) from e
