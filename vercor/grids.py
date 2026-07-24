from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp

from vercor.dtypes import (
    PrecisionPolicy as _PrecisionPolicy,
    as_jax_real_array as _as_jax_real_array,
    jax_linspace as _jax_linspace,
)
from vercor.exceptions import GridError as _GridError
from vercor._pytree import PyTreeNodeMixin as _PyTreeNodeMixin
from vercor.types import RuntimeArray as _RuntimeArray


def _is_strictly_increasing(values: jax.Array) -> bool:
    return bool(jnp.all(jnp.diff(values) > 0.0))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False, repr=False, kw_only=True)
class RectilinearGrid(_PyTreeNodeMixin):
    """JAX-friendly public rectilinear grid with 1D lon/lat coordinates."""

    pytree_children = (
        "longitude",
        "latitude",
        "longitude_edges",
        "latitude_edges",
        "binary_mask",
    )
    pytree_aux_data = ("name",)

    name: str
    binary_mask: _RuntimeArray | None
    longitude: _RuntimeArray
    latitude: _RuntimeArray
    longitude_edges: _RuntimeArray | None
    latitude_edges: _RuntimeArray | None

    def __init__(
        self,
        name: str,
        *,
        longitude: _RuntimeArray,
        latitude: _RuntimeArray,
        longitude_edges: _RuntimeArray | None = None,
        latitude_edges: _RuntimeArray | None = None,
        binary_mask: _RuntimeArray | None = None,
        policy: _PrecisionPolicy = None,
    ) -> None:
        """Create a rectilinear grid from explicit coordinates."""

        longitude_array = _as_jax_real_array(longitude, policy)
        latitude_array = _as_jax_real_array(latitude, policy)
        longitude_edges_array = (
            None
            if longitude_edges is None
            else _as_jax_real_array(longitude_edges, policy)
        )
        latitude_edges_array = (
            None
            if latitude_edges is None
            else _as_jax_real_array(latitude_edges, policy)
        )
        binary_mask_array = (
            None if binary_mask is None else _as_jax_real_array(binary_mask, policy)
        )

        if longitude_array.ndim != 1 or latitude_array.ndim != 1:
            raise _GridError(
                "RectilinearGrid expects both longitude and latitude coordinates to be 1D arrays."
            )

        if not (
            _is_strictly_increasing(longitude_array)
            and _is_strictly_increasing(latitude_array)
        ):
            raise _GridError("longitude and latitude must be strictly monotonic.")

        if binary_mask_array is not None and binary_mask_array.ndim != 2:
            raise _GridError("Mask must be a 2D array.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "binary_mask", binary_mask_array)
        object.__setattr__(self, "longitude", longitude_array)
        object.__setattr__(self, "latitude", latitude_array)
        object.__setattr__(self, "longitude_edges", longitude_edges_array)
        object.__setattr__(self, "latitude_edges", latitude_edges_array)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(nlat, nlon)`` for horizontal grid-shaped fields."""

        return (int(self.latitude.size), int(self.longitude.size))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Grid name:  {self.name}\n"
            f"├── Grid shape: {self.shape}\n"
            f"└── Binary mask: "
            f"{'Provided' if self.binary_mask is not None else 'Not provided'}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape})"

    def with_precision(self, policy: _PrecisionPolicy) -> "RectilinearGrid":
        """Return this grid with real arrays converted to ``policy`` precision."""

        return RectilinearGrid(
            name=self.name,
            longitude=self.longitude,
            latitude=self.latitude,
            longitude_edges=self.longitude_edges,
            latitude_edges=self.latitude_edges,
            binary_mask=self.binary_mask,
            policy=policy,
        )

    @classmethod
    def from_coordinates(
        cls,
        name: str,
        *,
        longitude: Any,
        latitude: Any,
        longitude_edges: Any | None = None,
        latitude_edges: Any | None = None,
        binary_mask: Any | None = None,
        policy: _PrecisionPolicy = None,
    ) -> Self:
        """Build a rectilinear grid from explicit coordinate arrays."""

        return cls(
            name=name,
            longitude=longitude,
            latitude=latitude,
            longitude_edges=longitude_edges,
            latitude_edges=latitude_edges,
            binary_mask=binary_mask,
            policy=policy,
        )

    @classmethod
    def uniform(
        cls,
        name: str,
        *,
        nlon: int,
        nlat: int,
        longitude: tuple[float, float],
        latitude: tuple[float, float],
        binary_mask: Any | None = None,
        policy: _PrecisionPolicy = None,
    ) -> Self:
        """Build a rectilinear grid with equally spaced coordinate centers."""

        return cls.from_coordinates(
            name,
            longitude=_jax_linspace(longitude[0], longitude[1], nlon, policy=policy),
            latitude=_jax_linspace(latitude[0], latitude[1], nlat, policy=policy),
            binary_mask=binary_mask,
            policy=policy,
        )


__all__ = [
    "RectilinearGrid",
]
