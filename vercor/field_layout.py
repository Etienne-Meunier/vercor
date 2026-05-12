from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.exceptions import ComponentError
from vercor.types import RuntimeArray

CANONICAL_DATA_LAYOUTS = (
    "(nLat, nLon)",
    "(nTime, nLat, nLon)",
    "(nLev, nLat, nLon)",
    "(nTime, nLev, nLat, nLon)",
)


def canonical_data_layout_description() -> str:
    """Return the accepted component data layout description."""

    return ", ".join(CANONICAL_DATA_LAYOUTS)


def is_canonical_grid_field_shape(
    shape: tuple[int, ...],
    grid_shape: tuple[int, int],
) -> bool:
    """Return whether ``shape`` follows VerCOR's canonical grid-field layout."""

    return 2 <= len(shape) <= 4 and shape[-2:] == grid_shape


def validate_component_data_layout(
    *,
    component_name: str,
    grid_shape: tuple[int, int],
    data: Mapping[str, RuntimeArray],
) -> None:
    """Validate all component data arrays against canonical grid-field layouts."""

    for field_name, field_value in data.items():
        shape = tuple(int(size) for size in jnp.asarray(field_value).shape)
        if not is_canonical_grid_field_shape(shape, grid_shape):
            raise ComponentError(
                f"Component '{component_name}' data field '{field_name}' has "
                f"shape {shape}; expected canonical grid-field layout "
                f"{canonical_data_layout_description()} with trailing grid "
                f"shape {grid_shape}. Non-grid metadata must be stored outside "
                "Component.data."
            )


def canonicalize_time_last_surface_field(field: ArrayLike) -> jax.Array:
    """Convert a ``(nLon, nLat, nTime)`` field to ``(nTime, nLat, nLon)``."""

    field_array = as_jax_real_array(field)
    if field_array.ndim != 3:
        raise ValueError(
            "Expected a time-last surface field with shape (nLon, nLat, nTime)."
        )
    return field_array.transpose((2, 1, 0))


def canonicalize_time_last_level_field(field: ArrayLike) -> jax.Array:
    """Convert ``(nLon, nLat, nLev, nTime)`` to ``(nTime, nLev, nLat, nLon)``."""

    field_array = as_jax_real_array(field)
    if field_array.ndim != 4:
        raise ValueError(
            "Expected a time-last level field with shape " "(nLon, nLat, nLev, nTime)."
        )
    return field_array.transpose((3, 2, 1, 0))
