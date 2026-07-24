"""Dependency-light structural component protocol shared by public contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax

from vercor.components.contexts import StepContext
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contracts import ComponentSpec

_KEEP_PAYLOAD = object()


def _snapshot_mapping(
    values: Mapping[str, Any] | None,
    *,
    label: str = "field values",
) -> Mapping[str, Any]:
    """Return a detached read-only insertion-ordered mapping snapshot."""

    if values is not None and not isinstance(values, Mapping):
        raise TypeError(
            f"{label} must be a mapping or None; got {type(values).__name__}"
        )
    snapshot: dict[str, Any] = {}
    for name, value in (values or {}).items():
        if not isinstance(name, str) or not name:
            raise TypeError("component field names must be non-empty strings")
        copy = getattr(value, "copy", None)
        snapshot[name] = (
            copy()
            if callable(copy) and type(value).__module__.startswith("numpy")
            else value
        )
    return MappingProxyType(snapshot)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class StepResult:
    """Field and optional payload updates returned by a component step."""

    __module__ = "vercor.components.contracts"

    fields: Mapping[str, RuntimeArray]
    payload: Any

    def __init__(
        self,
        fields: Mapping[str, RuntimeArray] | None = None,
        payload: Any = _KEEP_PAYLOAD,
    ) -> None:
        object.__setattr__(
            self,
            "fields",
            _snapshot_mapping(fields, label="StepResult.fields"),
        )
        object.__setattr__(self, "payload", payload)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten fields and payload while preserving mapping order."""

        names = tuple(self.fields)
        preserve_payload = self.payload is _KEEP_PAYLOAD
        children = tuple(self.fields.values())
        if not preserve_payload:
            children = (*children, self.payload)
        return children, (names, preserve_payload)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]
    ) -> "StepResult":
        """Restore a result from JAX PyTree leaves."""

        names, preserve_payload = aux_data
        field_count = len(names)
        payload = _KEEP_PAYLOAD if preserve_payload else children[field_count]
        return cls(dict(zip(names, children[:field_count], strict=True)), payload)


@runtime_checkable
class Component(Protocol):
    """Structural contract implemented by every VerCOR model component."""

    @property
    def name(self) -> str:
        """Return the unique component name."""
        ...

    @property
    def grid(self) -> RectilinearGrid:
        """Return the component's rectilinear grid."""
        ...

    @property
    def spec(self) -> "ComponentSpec":
        """Return the immutable component declaration."""
        ...

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray] | StepResult:
        """Return declared field updates for one component step."""
        ...


Component.__module__ = "vercor.components.contracts"


def _resolve_component_spec(component_spec: type[Any]) -> None:
    """Resolve the public protocol annotation once declarations finish loading."""

    globals()["ComponentSpec"] = component_spec


__all__: list[str] = []
