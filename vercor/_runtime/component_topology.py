from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.exceptions import CouplerError

if TYPE_CHECKING:
    from vercor.components.base import Component


def require_component(components: Mapping[str, Component], name: str) -> Component:
    """Return the component registered under ``name`` or raise a coupler error."""

    try:
        component = components[name]
    except KeyError as exc:
        raise CouplerError(f"No component of type {name!r} registered") from exc

    if component.name != name:
        raise CouplerError(
            f"Component registered under key {name!r} has name {component.name!r}; "
            "component mapping keys must match component.name"
        )
    return component
