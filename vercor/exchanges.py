"""Public exchange declarations for coupled component field transfers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable

import vercor.fields as _fields
import vercor.regridding as _regridding
from vercor.fields import _normalize_field_items
from vercor.regridding import bilinear as _bilinear


def _regridder_factory_name(regridder_factory: Callable[..., object]) -> str:
    """Return a stable display name for a regridder factory callable."""

    if isinstance(regridder_factory, partial):
        wrapped_factory = regridder_factory.func
        if callable(wrapped_factory):
            return _regridder_factory_name(wrapped_factory)
        return wrapped_factory.__class__.__name__

    name = getattr(regridder_factory, "__name__", None)
    if isinstance(name, str):
        return name
    return regridder_factory.__class__.__name__


@dataclass(frozen=True, init=False)
class Exchange:
    """Public exchange declaration connecting source fields to a target."""

    source: str
    target: str
    fields: Sequence[_fields.ExchangeField]
    regrid: _regridding.RegridderFactory
    _label: str | None
    _regrid_key: str

    def __init__(
        self,
        source: str,
        target: str,
        fields: Sequence[_fields.ExchangeField],
        *,
        regrid: _regridding.RegridderFactory = _bilinear,
        label: str | None = None,
    ) -> None:
        """Create an exchange declaration."""

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "fields", _normalize_field_items(fields))
        object.__setattr__(self, "regrid", regrid)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_regrid_key", _regridder_factory_name(regrid))

    @property
    def label(self) -> str:
        """Return explicit name or a stable derived logging label."""

        if self._label is not None:
            return self._label
        return f"{self.source} --({self._regrid_key})--> {self.target}"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Name: {self.label}\n"
            f"├── Source component: {self.source}\n"
            f"└── Target component: {self.target}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(label={self._label}, source={self.source},"
            f" target={self.target}, fields={self.fields})"
        )


__all__ = [
    "Exchange",
]
