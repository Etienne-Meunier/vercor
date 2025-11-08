from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray
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
) -> RectilinearGrid:
    """Helper to build rectilinear grid"""

    longitude = np.linspace(longitude_start, longitude_end, nlon, dtype=float)
    latitude = np.linspace(latitude_start, latitude_end, nlat, dtype=float)

    return RectilinearGrid(name=name, longitude=longitude, latitude=latitude, mask=mask)


def _scalar_field_interpolate(
    field_name: str,
    source_fields: Dict[str, NDArray],
    regridder: Regridder,
) -> NDArray:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for scalar field interpolation")

    result: NDArray = regridder(source_fields[field_name])
    return result


def _vector_field_interpolate(
    field_name: Tuple[str, str],
    source_fields: Dict[str, NDArray],
    regridder: Regridder,
) -> Tuple[NDArray, NDArray]:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for vector field interpolation")

    if len(field_name) == 2:
        field_name1, field_name2 = field_name
    else:
        raise ValueError("Vector field name must be a tuple of two strings")

    try:
        result: Tuple[NDArray, NDArray] = regridder(source_fields[field_name1], source_fields[field_name2])
        return result
    except Exception as e:
        raise TypeError(
            "Regridder for vector fields must accept two arguments and return a tuple of two arrays"
        ) from e
