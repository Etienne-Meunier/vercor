import jax.numpy as jnp
from typing import TYPE_CHECKING, Any

from vercor.components import (
    ComponentSpec,
    DataComponent,
    TransferPolicy,
)
from vercor.dtypes import as_jax_real_array
from vercor.grids import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.setups._data._field_helpers import canonicalize_surface_field

if TYPE_CHECKING:
    from dinosaur.coordinate_systems import CoordinateSystem as _CoordinateSystem
    from jcm.forcing import ForcingData as _ForcingData
else:
    _CoordinateSystem = Any
    _ForcingData = Any

_JCM_LAND_INPUT_NAMES = ("latent_heat_flux", "sensible_heat_flux")
_JCM_LAND_FIELD_NAMES = ("land_surface_temperature", "soil_moisture")


def make_jcm_land(
    jcm_coords: _CoordinateSystem | Any,
    jcm_forcing: _ForcingData | Any,
    ocn_grid: RectilinearGrid,
    name: str = "LND",
) -> DataComponent:
    """Return a JCM land forcing component."""

    longitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.longitudes))
    latitude = jnp.rad2deg(as_jax_real_array(jcm_coords.horizontal.latitudes))
    land_surface_temperature = canonicalize_surface_field(jcm_forcing.stl_am)
    soil_moisture = canonicalize_surface_field(jcm_forcing.soilw_am)
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

    component = DataComponent(
        name,
        grid,
        {
            "land_surface_temperature": land_surface_temperature,
            "soil_moisture": soil_moisture,
        },
        spec=ComponentSpec(
            inputs=_JCM_LAND_INPUT_NAMES,
            outputs=_JCM_LAND_FIELD_NAMES,
            initial_fields={field_name: 0.0 for field_name in _JCM_LAND_INPUT_NAMES},
            transfer=TransferPolicy(time_selection="daily"),
        ),
    )

    return component
