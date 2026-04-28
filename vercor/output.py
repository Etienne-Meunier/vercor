from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from vercor.runtime_views import RuntimeComponentView
from vercor.types import RuntimeArray


def _runtime_array_to_host(array: RuntimeArray) -> NDArray[Any]:
    """Transfer a runtime array to host memory for xarray output."""

    return np.asarray(jax.device_get(jnp.asarray(array)))


def _write_runtime_component_to_netcdf(
    view: RuntimeComponentView,
    filename: Path,
    *,
    masks: dict[str, RuntimeArray] | None = None,
) -> None:
    """Write final runtime component fields to a netCDF file.

    Arguments:
        view: runtime component view containing fields to write
        filename: path to the output NetCDF file
        masks: optional mask fields to include in the same output
    """

    lat = xr.DataArray(
        _runtime_array_to_host(view.grid.latitude), dims=("nlat",), name="latitude"
    )
    lon = xr.DataArray(
        _runtime_array_to_host(view.grid.longitude), dims=("nlon",), name="longitude"
    )

    data_vars = {}
    for store_name, store in (
        ("incoming", view.incoming),
        ("outgoing", view.outgoing),
    ):
        for name, value in zip(store.field_names, store.values):
            data_vars[f"{store_name}_{name}"] = xr.DataArray(
                data=_runtime_array_to_host(value),
                dims=("nlat", "nlon"),
                coords={"latitude": lat, "longitude": lon},
                attrs={
                    "component": view.name,
                    "runtime_store": store_name,
                    "field_name": name,
                },
            )

    for name, value in (masks or {}).items():
        data_vars[name] = xr.DataArray(
            data=_runtime_array_to_host(value),
            dims=("nlat", "nlon"),
            coords={"latitude": lat, "longitude": lon},
            attrs={
                "component": view.name,
                "runtime_store": "mask",
                "field_name": name,
            },
        )

    xr.Dataset(
        data_vars=data_vars,
        coords={"latitude": lat, "longitude": lon},
    ).to_netcdf(filename)


def write_runtime_component_view_to_netcdf(
    view: RuntimeComponentView,
    filename: Path,
    *,
    masks: dict[str, RuntimeArray] | None = None,
) -> None:
    """Write final runtime fields from a single runtime component view."""

    _write_runtime_component_to_netcdf(
        view,
        filename,
        masks=masks,
    )
