from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import jax

from vercor.grids import RectilinearGrid
from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.contracts import exchange_key_name
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime.state import RuntimeComponentState

FieldStore: TypeAlias = Literal["data", "incoming", "outgoing"]


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

    def __init__(self) -> None:
        """Block direct public construction of opaque runtime state."""

        raise TypeError(
            "RunState is created by Coupler.initial_state() or Coupler.run(); "
            "use those Coupler methods to create coupled runtime state."
        )

    @classmethod
    def _from_runtime(
        cls,
        *,
        component_names: tuple[str, ...],
        components: tuple["RuntimeComponentState", ...],
        fractional_masks: RuntimeFieldStore,
        component_grids: tuple[RectilinearGrid | None, ...] = (),
    ) -> "RunState":
        """Create an opaque public run state from private runtime containers."""

        state = object.__new__(cls)
        state._initialize_runtime(
            component_names=component_names,
            components=components,
            fractional_masks=fractional_masks,
            component_grids=component_grids,
        )
        return state

    def _initialize_runtime(
        self,
        *,
        component_names: tuple[str, ...],
        components: tuple["RuntimeComponentState", ...],
        fractional_masks: RuntimeFieldStore,
        component_grids: tuple[RectilinearGrid | None, ...] = (),
    ) -> None:
        """Assign private runtime containers after construction is authorized."""

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

        return ComponentState._from_runtime(
            name,
            self.component_grids[self._component_index(name)],
            self._component_state(name),
        )

    def components(
        self,
        names: Sequence[str] | None = None,
    ) -> Mapping[str, "ComponentState"]:
        """Return public field views for selected components."""

        selected_names = self.component_names if names is None else tuple(names)
        return MappingProxyType({name: self.component(name) for name in selected_names})

    def with_fields(
        self,
        component: str,
        fields: Mapping[str, RuntimeArray],
        *,
        store: FieldStore = "data",
    ) -> "RunState":
        """Return a new run state with existing component fields replaced."""

        component_state = self._component_state(component)
        runtime_store = _runtime_store(component_state, store)
        missing = tuple(field for field in fields if field not in runtime_store)
        if missing:
            raise KeyError(f"Runtime field {missing[0]!r} not found")

        updated_store = runtime_store.replace_many(fields)
        if store == "data":
            updated_component = component_state.with_data(updated_store)
        elif store == "incoming":
            updated_component = component_state.with_incoming(updated_store)
        else:
            updated_component = component_state.with_outgoing(updated_store)
        return self._with_component_state(component, updated_component)

    def with_component_fields(
        self,
        name: str,
        fields: Mapping[str, RuntimeArray],
    ) -> "RunState":
        """Return a new run state with existing data fields replaced."""

        warnings.warn(
            "RunState.with_component_fields() is deprecated; use "
            "RunState.with_fields(component, fields) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.with_fields(name, fields)

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
        return RunState._from_runtime(
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
        *,
        fields: Mapping[str, Any] | None = None,
        incoming: Mapping[str, Any] | None = None,
        outgoing: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a public component-state view from plain field mappings."""

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "_data", _field_store_from_mapping(fields))
        object.__setattr__(self, "_incoming", _field_store_from_mapping(incoming))
        object.__setattr__(self, "_outgoing", _field_store_from_mapping(outgoing))

    def field_candidates(self, name: str) -> list[RuntimeArray]:
        """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

        return runtime_field_candidates(self, name)

    def field(
        self,
        name: str,
        *,
        store: FieldStore | None = None,
    ) -> RuntimeArray:
        """Return one runtime field, optionally constrained to one store."""

        if store is not None:
            return self._store(store).get(name)

        return runtime_field(self, name)

    def fields(
        self,
        *,
        store: FieldStore = "data",
    ) -> Mapping[str, RuntimeArray]:
        """Return one runtime store as a read-only field mapping."""

        return MappingProxyType(self._store(store).to_mapping())

    def iter_fields(
        self,
        *stores: FieldStore,
    ) -> Iterator[tuple[FieldStore, str, RuntimeArray]]:
        """Yield ``(store_name, field_name, value)`` for selected runtime stores."""

        runtime_stores = self._stores()
        selected_store_names = stores or tuple(runtime_stores)
        for store_name in selected_store_names:
            try:
                store = runtime_stores[store_name]
            except KeyError as exc:
                raise KeyError(f"Runtime view store {store_name!r} not found") from exc
            for field_name, value in zip(store.field_names, store.values, strict=True):
                yield store_name, field_name, value

    def iter_store_fields(
        self,
        *store_names: FieldStore,
    ) -> Iterator[tuple[FieldStore, str, RuntimeArray]]:
        """Deprecated wrapper for :meth:`iter_fields`."""

        warnings.warn(
            "ComponentState.iter_store_fields() is deprecated; use "
            "ComponentState.iter_fields() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield from self.iter_fields(*store_names)

    @classmethod
    def _from_runtime(
        cls,
        name: str,
        grid: RectilinearGrid | None,
        component_state: "RuntimeComponentState",
    ) -> "ComponentState":
        """Create a field view from component metadata and runtime state."""

        view = object.__new__(cls)
        object.__setattr__(view, "name", name)
        object.__setattr__(view, "grid", grid)
        object.__setattr__(view, "_data", component_state.data)
        object.__setattr__(view, "_incoming", component_state.incoming)
        object.__setattr__(view, "_outgoing", component_state.outgoing)
        return view

    def _stores(self) -> dict[FieldStore, RuntimeFieldStore]:
        return {
            "data": self._data,
            "incoming": self._incoming,
            "outgoing": self._outgoing,
        }

    def _store(
        self,
        name: FieldStore,
    ) -> RuntimeFieldStore:
        try:
            return self._stores()[name]
        except KeyError as exc:
            raise KeyError(f"Runtime view store {name!r} not found") from exc


def _field_store_from_mapping(
    fields: Mapping[str, Any] | None,
) -> RuntimeFieldStore:
    """Return a runtime field store from public plain field mappings."""

    if fields is None:
        return RuntimeFieldStore.empty()
    if isinstance(fields, RuntimeFieldStore):
        raise TypeError(
            "ComponentState expects plain field mappings, not runtime stores"
        )
    return RuntimeFieldStore.from_mapping(fields)


def _runtime_store(
    component_state: "RuntimeComponentState",
    store: FieldStore,
) -> RuntimeFieldStore:
    """Return a runtime component store by public store name."""

    if store == "data":
        return component_state.data
    if store == "incoming":
        return component_state.incoming
    if store == "outgoing":
        return component_state.outgoing
    raise KeyError(f"Runtime view store {store!r} not found")


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
