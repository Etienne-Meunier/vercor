import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.forcing import ForcingData

from vercor.components.base import DataComponent
from vercor.field_layout import canonicalize_time_last_surface_field
from vercor.grid import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn

_JCM_LAND_FIELD_NAMES = ("land_surface_temperature", "soil_moisture")


def _coordinates_in_degrees(
    longitude_radians: ArrayLike,
    latitude_radians: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Convert JCM coordinates from radians to degrees on the JAX runtime path."""
    return _jcm_coordinates_in_degrees(longitude_radians, latitude_radians)


def _jcm_coordinates_in_degrees(
    longitude_radians: ArrayLike,
    latitude_radians: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Convert JCM horizontal coordinates from radians to degrees."""
    return (
        jnp.rad2deg(jnp.asarray(longitude_radians)),
        jnp.rad2deg(jnp.asarray(latitude_radians)),
    )


def _prepare_jcm_land_runtime_fields(
    longitude_radians: ArrayLike,
    latitude_radians: ArrayLike,
    land_surface_temperature: ArrayLike,
    soil_moisture: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare JCM land coordinates and monthly forcing fields for VerCOR storage."""
    longitude, latitude = _jcm_coordinates_in_degrees(
        longitude_radians, latitude_radians
    )
    land_surface_temperature_array = jnp.asarray(land_surface_temperature)
    soil_moisture_array = jnp.asarray(soil_moisture)
    if land_surface_temperature_array.ndim == 3:
        prepared_land_surface_temperature = canonicalize_time_last_surface_field(
            land_surface_temperature_array
        )
    else:
        prepared_land_surface_temperature = land_surface_temperature_array.T
    if soil_moisture_array.ndim == 3:
        prepared_soil_moisture = canonicalize_time_last_surface_field(
            soil_moisture_array
        )
    else:
        prepared_soil_moisture = soil_moisture_array.T
    return (
        longitude,
        latitude,
        prepared_land_surface_temperature,
        prepared_soil_moisture,
    )


class JCMLand(DataComponent):
    def __init__(
        self,
        jcm_coords: CoordinateSystem,
        jcm_forcing: ForcingData,
        ocn_grid: RectilinearGrid,
        name: str = "LND",
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            jcm_coords (CoordinateSystem): JCM coordinate system object
            jcm_forcing (ForcingData): JCM forcing data object

        Attributes of parent classes to be initialized:
            Component
                name: str
                grid: RectilinearGrid
        """

        (
            longitude,
            latitude,
            land_surface_temperature,
            soil_moisture,
        ) = _prepare_jcm_land_runtime_fields(
            jcm_coords.horizontal.longitudes,
            jcm_coords.horizontal.latitudes,
            jcm_forcing.stl_am,
            jcm_forcing.soilw_am,
        )
        lnd_bmask, _ = create_lnd_mask_from_ocn(
            atm_lat=latitude,
            atm_lon=longitude,
            ocn_grid=ocn_grid,
        )

        grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=lnd_bmask,
        )

        super().__init__(name, grid=grid)
        self.declare_fields(outputs=_JCM_LAND_FIELD_NAMES)

        self.update_settings(get_field_time_slice=True)

        # Units: [K]
        self.seed_field("land_surface_temperature", land_surface_temperature)
        # Units: [???]
        self.seed_field("soil_moisture", soil_moisture)
