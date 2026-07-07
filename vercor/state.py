from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import jax

from vercor.grids import RectilinearGrid
from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.contracts import exchange_key_name
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime.state import RuntimeComponentState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class RunState(PyTreeNodeMixin):
    """Immutable coupled model state returned by the public coupler facade."""

    pytree_children = ("component_grids", "_components", "_fractional_masks")
    pytree_aux_data = ("component_names",)

    component_names: tuple[str, ...]
    component_grids: tuple[RectilinearGrid | None, ...]
    _components: tuple["RuntimeComponentState", ...]
    _fractional_masks: RuntimeFieldStore
    component_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        component_names: tuple[str, ...],
        components: tuple["RuntimeComponentState", ...],
        fractional_masks: RuntimeFieldStore,
        component_grids: tuple[RectilinearGrid | None, ...] = (),
    ) -> None:
        """Create an opaque public run state from private runtime containers."""

        object.__setattr__(self, "component_names", component_names)
        object.__setattr__(
            self,
            "component_grids",
            component_grids or tuple(None for _ in component_names),
        )
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_fractional_masks", fractional_masks)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate that component names and states stay aligned."""

        if len(self.component_names) != len(self._components):
            raise ValueError("component_names and components must have equal length")
        if len(self.component_names) != len(self.component_grids):
            raise ValueError(
                "component_names and component_grids must have equal length"
            )
        object.__setattr__(
            self,
            "component_indices",
            {name: index for index, name in enumerate(self.component_names)},
        )

    def _pytree_post_unflatten(self) -> None:
        """Validate that component names and states stay aligned."""

        self.__post_init__()

    def component(self, name: str) -> "ComponentState":
        """Return a public field view for one component."""

        return ComponentState.from_component_state(
            name,
            self.component_grids[self._component_index(name)],
            self._component_state(name),
        )

    def with_component_fields(
        self,
        name: str,
        fields: Mapping[str, RuntimeArray],
    ) -> "RunState":
        """Return a new run state with existing data fields replaced."""

        component_state = self._component_state(name)
        missing = tuple(field for field in fields if field not in component_state.data)
        if missing:
            raise KeyError(f"Runtime field {missing[0]!r} not found")
        return self._with_component_state(
            name,
            component_state.with_data(component_state.data.replace_many(fields)),
        )

    def _component_index(self, name: str) -> int:
        try:
            return self.component_indices[name]
        except KeyError as exc:
            raise KeyError(f"Runtime component {name!r} not found") from exc

    def _component_state(self, name: str) -> "RuntimeComponentState":
        """Return one component state by name."""

        return self._components[self._component_index(name)]

    def _with_component_state(
        self,
        name: str,
        component_state: "RuntimeComponentState",
    ) -> "RunState":
        """Return a new run state with one component replaced."""

        components = list(self._components)
        components[self._component_index(name)] = component_state
        return RunState(
            component_names=self.component_names,
            components=tuple(components),
            fractional_masks=self._fractional_masks,
            component_grids=self.component_grids,
        )

    def _fractional_mask(
        self,
        source: str,
        destination: str,
        regrid_key: str,
    ) -> RuntimeArray:
        """Return the fractional mask for an exchange."""

        return self._fractional_masks.get(
            exchange_key_name(source, destination, regrid_key)
        )


@dataclass(frozen=True, init=False)
class ComponentState:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: RectilinearGrid | None
    _data: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    _incoming: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    _outgoing: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid | None,
        data: RuntimeFieldStore | None = None,
        incoming: RuntimeFieldStore | None = None,
        outgoing: RuntimeFieldStore | None = None,
    ) -> None:
        """Create a public component-state view from runtime stores."""

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(
            self,
            "_data",
            RuntimeFieldStore.empty() if data is None else data,
        )
        object.__setattr__(
            self,
            "_incoming",
            RuntimeFieldStore.empty() if incoming is None else incoming,
        )
        object.__setattr__(
            self,
            "_outgoing",
            RuntimeFieldStore.empty() if outgoing is None else outgoing,
        )

    def field_candidates(self, name: str) -> list[RuntimeArray]:
        """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

        return runtime_field_candidates(self, name)

    def field(
        self,
        name: str,
        *,
        store: Literal["data", "incoming", "outgoing"] | None = None,
    ) -> RuntimeArray:
        """Return one runtime field, optionally constrained to one store."""

        if store is not None:
            return self._store(store).get(name)

        return runtime_field(self, name)

    def fields(
        self,
        *,
        store: Literal["data", "incoming", "outgoing"] = "data",
    ) -> Mapping[str, RuntimeArray]:
        """Return one runtime store as a read-only field mapping."""

        return MappingProxyType(self._store(store).to_mapping())

    def iter_store_fields(
        self,
        *store_names: str,
    ) -> Iterator[tuple[str, str, RuntimeArray]]:
        """Yield ``(store_name, field_name, value)`` for selected runtime stores."""

        stores = self._stores()
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
        grid: RectilinearGrid | None,
        component_state: "RuntimeComponentState",
    ) -> "ComponentState":
        """Create a field view from component metadata and runtime state."""

        return cls(
            name=name,
            grid=grid,
            data=component_state.data,
            incoming=component_state.incoming,
            outgoing=component_state.outgoing,
        )

    def _stores(self) -> dict[str, RuntimeFieldStore]:
        return {
            "data": self._data,
            "incoming": self._incoming,
            "outgoing": self._outgoing,
        }

    def _store(
        self,
        name: Literal["data", "incoming", "outgoing"],
    ) -> RuntimeFieldStore:
        try:
            return self._stores()[name]
        except KeyError as exc:
            raise KeyError(f"Runtime view store {name!r} not found") from exc


if TYPE_CHECKING:
    RuntimeFieldSource: TypeAlias = ComponentState | RuntimeComponentState
else:
    RuntimeFieldSource: TypeAlias = Any


def runtime_field_candidates(
    source: RuntimeFieldSource,
    name: str,
) -> list[RuntimeArray]:
    """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

    candidates: list[RuntimeArray] = []
    stores = (
        (source._data, source._incoming, source._outgoing)
        if isinstance(source, ComponentState)
        else (source.data, source.incoming, source.outgoing)
    )
    for store in stores:
        if name in store:
            candidates.append(store.get(name))
    return candidates


def runtime_field(source: RuntimeFieldSource, name: str) -> RuntimeArray:
    """Return the first runtime field named ``name`` from a view or state."""

    candidates = runtime_field_candidates(source, name)
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {name!r} not found")


__all__ = ["RunState", "ComponentState"]
