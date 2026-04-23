from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


def _is_strictly_increasing(values: jax.Array) -> bool:
    return bool(jnp.all(jnp.diff(values) > 0.0))


@dataclass(frozen=True)
class Grid(abc.ABC):
    name: str
    binary_mask: Any | None = None  # values of 1 for active, 0 for inactive

    def __post_init__(self) -> None:
        if self.binary_mask is not None:
            mask = jnp.asarray(self.binary_mask)
            if mask.ndim != 2:
                raise ValueError("Mask must be a 2D array.")
            object.__setattr__(self, "binary_mask", mask)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Grid name:  {self.name}\n"
            f"├── Grid shape: {self.shape}\n"
            f"└── Binary mask: {'Provided' if self.binary_mask is not None else 'Not provided'}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape})"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False, repr=False, kw_only=True)
class RectilinearGrid(Grid):
    longitude: Any
    latitude: Any
    longitude_edges: Any | None
    latitude_edges: Any | None

    def __init__(
        self,
        name: str,
        longitude: Any,
        latitude: Any,
        longitude_edges: Any | None = None,
        latitude_edges: Any | None = None,
        binary_mask: Any | None = None,
    ) -> None:
        longitude_array = jnp.asarray(longitude)
        latitude_array = jnp.asarray(latitude)
        longitude_edges_array = (
            None if longitude_edges is None else jnp.asarray(longitude_edges)
        )
        latitude_edges_array = (
            None if latitude_edges is None else jnp.asarray(latitude_edges)
        )
        binary_mask_array = None if binary_mask is None else jnp.asarray(binary_mask)

        if longitude_array.ndim != 1 or latitude_array.ndim != 1:
            raise ValueError(
                "RectilinearGrid expects both longitude and latitude coordinates to be 1D arrays."
            )

        if not (
            _is_strictly_increasing(longitude_array)
            and _is_strictly_increasing(latitude_array)
        ):
            raise ValueError("longitude and latitude must be strictly monotonic.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "binary_mask", binary_mask_array)
        Grid.__post_init__(self)
        object.__setattr__(self, "longitude", longitude_array)
        object.__setattr__(self, "latitude", latitude_array)
        object.__setattr__(self, "longitude_edges", longitude_edges_array)
        object.__setattr__(self, "latitude_edges", latitude_edges_array)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.latitude.size), int(self.longitude.size))

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[str]]:
        children = (
            self.longitude,
            self.latitude,
            self.longitude_edges,
            self.latitude_edges,
            self.binary_mask,
        )
        return children, (self.name,)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[str], children: tuple[Any, ...]
    ) -> "RectilinearGrid":
        name = aux_data[0]
        longitude, latitude, longitude_edges, latitude_edges, binary_mask = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "name", name)
        object.__setattr__(obj, "binary_mask", binary_mask)
        object.__setattr__(obj, "longitude", longitude)
        object.__setattr__(obj, "latitude", latitude)
        object.__setattr__(obj, "longitude_edges", longitude_edges)
        object.__setattr__(obj, "latitude_edges", latitude_edges)
        return obj
