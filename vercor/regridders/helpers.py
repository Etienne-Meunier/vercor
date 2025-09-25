from typing import Dict, Tuple
import numpy as np
from vercor.grid import RectilinearGrid
from vercor.regridders.base import Regridder
from vercor.fields import Field


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
    source_fields: Dict[str, Field],
    regridder: Regridder,
) -> Field:
    if not callable(regridder):
        raise TypeError("Regridder must be callable for scalar field interpolation")

    destination_field = regridder(source_fields[field_name])

    return destination_field


def _vector_field_interpolate(
    field_name: Tuple[str, str],
    source_fields: Dict[str, Field],
    regridder: Regridder,
) -> Tuple[Field, Field]:
    if len(field_name) == 2:
        src_field_name, alt_field_name = field_name
    else:
        raise ValueError("Vector field name must be a tuple of two strings")
    if not callable(regridder):
        raise TypeError("Regridder must be callable for vector field interpolation")
    try:
        destination_field_lon, destination_field_lat = regridder(
            source_fields[src_field_name], source_fields[alt_field_name]
        )
    except Exception as e:
        raise TypeError(
            "Regridder for vector fields must accept two arguments and return a tuple of two Fields"
        ) from e
    return (destination_field_lon, destination_field_lat)
