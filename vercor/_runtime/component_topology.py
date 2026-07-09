from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

from vercor.exceptions import CouplerError


class _NamedComponent(Protocol):
    @property
    def name(self) -> str:
        """Return the registered component name."""
        ...


_ComponentT = TypeVar("_ComponentT", bound=_NamedComponent)


def require_component(
    components: Mapping[str, _ComponentT],
    name: str,
) -> _ComponentT:
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
