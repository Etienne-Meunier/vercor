from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable, TypeAlias

from vercor.fields import ExchangeField, normalize_field_items
from vercor._regridders.bilinear import BilinearRectilinearRegridder, bilinear
from vercor._regridders.conservative import ConservativeRectilinearRegridder

RegridderFactory: TypeAlias = Callable[
    ..., BilinearRectilinearRegridder | ConservativeRectilinearRegridder
]


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
    """Public exchange declaration connecting source fields to a destination.

    Exchange objects are static configuration. The coupler converts them into
    runtime contracts and dispatch metadata before execution so traced runtime
    state only carries arrays and stable field-store metadata.
    """

    source: str
    target: str
    fields: Sequence[ExchangeField]
    regrid: RegridderFactory
    _label: str | None
    _regrid_key: str

    def __init__(
        self,
        source: str,
        target: str,
        fields: Sequence[ExchangeField],
        regrid: RegridderFactory = bilinear,
        label: str | None = None,
    ) -> None:
        """Create an exchange declaration."""

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "fields", normalize_field_items(fields))
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


def _exchange_regrid_key(exchange: Exchange) -> str:
    """Return the private stable regrid key for runtime mask bookkeeping."""

    return exchange._regrid_key
