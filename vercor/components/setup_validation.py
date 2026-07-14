from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from vercor.exceptions import ComponentError
from vercor.field_layout import validate_component_data_layout
from vercor.grids import RectilinearGrid


def validate_component_setup(component: Any) -> None:
    """Raise a clear error when a component skipped base initialization."""

    required_attributes = ("name", "grid", "spec", "_data")
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(component, attribute)
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ComponentError(
            f"Component '{component.__class__.__name__}' is missing required setup "
            f"attribute(s): {missing_names}. Components must be normalized by "
            "the private preparation bridge before runtime state creation."
        )

    if not isinstance(component.grid, RectilinearGrid):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'grid'; "
            "expected RectilinearGrid."
        )
    if not isinstance(component._data, Mapping):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute '_data'; "
            "expected an immutable field mapping."
        )
    validate_component_data_layout(
        component_name=component.name,
        grid_shape=component.grid.shape,
        data=component._data,
    )
