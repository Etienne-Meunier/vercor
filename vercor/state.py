from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import jax

from vercor.grids import RectilinearGrid
from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.contracts import exchange_key_name
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime.state import RuntimeComponentState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RunState(PyTreeNodeMixin):
    """Immutable coupled model state returned by the public coupler facade."""

    pytree_children = ("components", "fractional_masks")
    pytree_aux_data = ("component_names",)

    component_names: tuple[str, ...]
    components: tuple["RuntimeComponentState", ...]
    fractional_masks: RuntimeFieldStore
    component_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that component names and states stay aligned."""

        if len(self.component_names) != len(self.components):
            raise ValueError("component_names and components must have equal length")
        object.__setattr__(
            self,
            "component_indices",
            {name: index for index, name in enumerate(self.component_names)},
        )

    def _pytree_post_unflatten(self) -> None:
        """Validate that component names and states stay aligned."""

        self.__post_init__()

    def get_component_state(self, name: str) -> "RuntimeComponentState":
        """Return one component state by name."""

        try:
            index = self.component_indices[name]
        except KeyError as exc:
            raise KeyError(f"Runtime component {name!r} not found") from exc
        return self.components[index]

    def set_component_state(
        self,
        name: str,
        component_state: "RuntimeComponentState",
    ) -> "RunState":
        """Return a new run state with one component replaced."""

        if name not in self.component_indices:
            raise KeyError(f"Runtime component {name!r} not found")
        components = list(self.components)
        components[self.component_indices[name]] = component_state
        return RunState(
            component_names=self.component_names,
            components=tuple(components),
            fractional_masks=self.fractional_masks,
        )

    def get_fractional_mask(
        self,
        source: str,
        destination: str,
        regrid_key: str,
    ) -> RuntimeArray:
        """Return the fractional mask for an exchange."""

        return self.fractional_masks.get(
            exchange_key_name(source, destination, regrid_key)
        )


@dataclass(frozen=True)
class ComponentView:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: RectilinearGrid
    data: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    incoming: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    outgoing: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)

    def field_candidates(self, name: str) -> list[RuntimeArray]:
        """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

        return runtime_field_candidates(self, name)

    def field(self, name: str) -> RuntimeArray:
        """Return the first runtime field named ``name`` from this view."""

        return runtime_field(self, name)

    def iter_store_fields(
        self,
        *store_names: str,
    ) -> Iterator[tuple[str, str, RuntimeArray]]:
        """Yield ``(store_name, field_name, value)`` for selected runtime stores."""

        stores = {
            "data": self.data,
            "incoming": self.incoming,
            "outgoing": self.outgoing,
        }
        selected_store_names = store_names or tuple(stores)
        for store_name in selected_store_names:
            try:
                store = stores[store_name]
            except KeyError as exc:
                raise KeyError(f"Runtime view store {store_name!r} not found") from exc
            for field_name, value in zip(store.field_names, store.values, strict=True):
                yield store_name, field_name, value

    @classmethod
    def from_component_state(
        cls,
        name: str,
        grid: RectilinearGrid,
        component_state: "RuntimeComponentState",
    ) -> "ComponentView":
        """Create a field view from component metadata and runtime state."""

        return cls(
            name=name,
            grid=grid,
            data=component_state.data,
            incoming=component_state.incoming,
            outgoing=component_state.outgoing,
        )


if TYPE_CHECKING:
    RuntimeFieldSource: TypeAlias = ComponentView | RuntimeComponentState
else:
    RuntimeFieldSource: TypeAlias = Any


def runtime_field_candidates(
    source: RuntimeFieldSource,
    name: str,
) -> list[RuntimeArray]:
    """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

    candidates: list[RuntimeArray] = []
    for store in (source.data, source.incoming, source.outgoing):
        if name in store:
            candidates.append(store.get(name))
    return candidates


def runtime_field(source: RuntimeFieldSource, name: str) -> RuntimeArray:
    """Return the first runtime field named ``name`` from a view or state."""

    candidates = runtime_field_candidates(source, name)
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {name!r} not found")


__all__ = ["RunState", "ComponentView"]
