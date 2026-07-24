"""Public exchange declarations for coupled component field transfers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import vercor.fields as _fields
import vercor.regridding as _regridding
from vercor.fields import _normalize_field_items
from vercor.regridding import bilinear as _bilinear


@dataclass(frozen=True, init=False)
class Exchange:
    """Declare one stable, named field-transfer route between components."""

    source: str
    target: str
    fields: Sequence[_fields.ExchangeField]
    route_id: str
    regridder_factory: _regridding.RegridderFactory

    def __init__(
        self,
        source: str,
        target: str,
        fields: Sequence[_fields.ExchangeField],
        *,
        route_id: str | None = None,
        regridder_factory: _regridding.RegridderFactory = _bilinear,
    ) -> None:
        """Create an exchange with an explicit or endpoint-derived route ID."""

        for label, value in (("source", source), ("target", target)):
            if not isinstance(value, str):
                raise TypeError(f"{label} must be a string")
            if not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if route_id is not None and not isinstance(route_id, str):
            raise TypeError("route_id must be a string or None")
        normalized_route_id = f"{source}->{target}" if route_id is None else route_id
        if not normalized_route_id.strip():
            raise ValueError("route_id must be a non-empty string")
        if not callable(regridder_factory):
            raise TypeError("regridder_factory must be callable")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "fields", _normalize_field_items(fields))
        object.__setattr__(self, "route_id", normalized_route_id)
        object.__setattr__(self, "regridder_factory", regridder_factory)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Route ID: {self.route_id}\n"
            f"├── Source component: {self.source}\n"
            f"└── Target component: {self.target}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(route_id={self.route_id!r}, "
            f"source={self.source!r}, target={self.target!r}, fields={self.fields!r})"
        )


__all__ = [
    "Exchange",
]
