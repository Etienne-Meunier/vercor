"""Single prepared-coupling boundary for private runtime orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from inspect import isroutine
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax

from vercor.clock import Clock
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor.physics import PhysicalConstants, _physical_constants_for_dtype
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor._runtime.initialization import initialize_coupler_runtime
from vercor._runtime.interrupts import RuntimeInterruptController
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.runtime import RuntimeOptions
from vercor.settings import Settings

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding, _ComponentDeclaration


@dataclass(frozen=True)
class _ComponentConfigurationSnapshot:
    """Comparable post-initialization component configuration snapshot."""

    runtime_configuration: tuple[Any, ...]
    public_configuration: tuple[Any, ...]


@dataclass(frozen=True)
class _PreparedConfigurationSnapshot:
    """Comparable snapshot of every owner used to prepare the runtime."""

    components: Mapping[str, _ComponentConfigurationSnapshot]
    clock: tuple[Any, ...]
    runtime: tuple[Any, ...]
    settings: tuple[Any, ...]
    constants: tuple[Any, ...]


@dataclass(frozen=True)
class PreparedCoupling:
    """Immutable static runtime resources prepared once for one Coupler."""

    components: Mapping[str, "_ComponentBinding"]
    exchanges: tuple[Exchange, ...]
    run_order: tuple[str, ...]
    contracts: Mapping[str, ExchangeContract]
    topology_maps: RuntimeTopologyMaps
    dispatch_context: RuntimeDispatchContext
    clock: Clock
    settings: Settings
    constants: PhysicalConstants
    runtime: RuntimeOptions
    interrupts: RuntimeInterruptController
    configuration_snapshot: _PreparedConfigurationSnapshot

    def validate_configuration(
        self,
        components: Mapping[str, "_ComponentDeclaration"],
        *,
        clock: Clock,
        settings: Settings,
        constants: PhysicalConstants,
        runtime: RuntimeOptions,
    ) -> None:
        """Reject direct configuration mutation after preparation."""

        snapshot = self.configuration_snapshot
        current_names = tuple(components)
        if current_names != tuple(snapshot.components):
            _raise_configuration_mutation("registered component mapping")
        for registered_name, component in components.items():
            current = _component_configuration_snapshot(component)
            if current != snapshot.components[registered_name]:
                _raise_configuration_mutation(f"component {registered_name!r}")
        if _clock_configuration_snapshot(clock) != snapshot.clock:
            _raise_configuration_mutation("clock")
        if _runtime_configuration_snapshot(runtime) != snapshot.runtime:
            _raise_configuration_mutation("runtime options")
        if _settings_configuration_snapshot(settings) != snapshot.settings:
            _raise_configuration_mutation("coupler settings")
        if _constants_configuration_snapshot(constants) != snapshot.constants:
            _raise_configuration_mutation("physical constants")


def prepare_coupling(
    *,
    components: Mapping[str, "_ComponentDeclaration"],
    exchanges: Sequence[Exchange],
    run_order: Sequence[str],
    clock: Clock,
    settings: Settings,
    constants: PhysicalConstants,
    runtime: RuntimeOptions,
    logger: LoggerLike,
) -> PreparedCoupling:
    """Initialize lifecycle state and build one immutable runtime boundary."""

    _ensure_jax_precision_capability(runtime)
    immutable_declarations = MappingProxyType(dict(components))
    immutable_exchanges = tuple(exchanges)
    immutable_run_order = tuple(run_order)
    runtime_constants = _physical_constants_for_dtype(constants, runtime.dtype)
    initialized = initialize_coupler_runtime(
        clock=clock,
        components=dict(immutable_declarations),
        exchanges=immutable_exchanges,
        run_order=immutable_run_order,
        settings=settings,
        constants=runtime_constants,
        dtype=runtime.dtype,
        logger=logger,
        topology_policy=runtime.topology,
    )
    contracts = MappingProxyType(dict(initialized.runtime_contracts))
    immutable_components = initialized.components
    topology_maps = initialized.topology.topology_maps
    dispatch_context = build_runtime_dispatch_context(
        immutable_components,
        immutable_exchanges,
        topology_maps.regridders,
        contracts,
        dt_seconds=clock.dt_seconds,
        settings=settings,
        constants=runtime_constants,
        dtype=runtime.dtype,
    )
    configuration_snapshot = _prepared_configuration_snapshot(
        components=immutable_declarations,
        clock=clock,
        settings=settings,
        constants=constants,
        runtime=runtime,
    )
    return PreparedCoupling(
        components=immutable_components,
        exchanges=immutable_exchanges,
        run_order=immutable_run_order,
        contracts=contracts,
        topology_maps=topology_maps,
        dispatch_context=dispatch_context,
        clock=clock,
        settings=settings,
        constants=runtime_constants,
        runtime=runtime,
        interrupts=RuntimeInterruptController(),
        configuration_snapshot=configuration_snapshot,
    )


def _ensure_jax_precision_capability(runtime: RuntimeOptions) -> None:
    """Enable process x64 capability when the requested allocation policy needs it."""

    if runtime.dtype.enable_x64 and not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)


def _component_configuration_snapshot(
    component: "_ComponentDeclaration",
) -> _ComponentConfigurationSnapshot:
    """Return runtime and public-owner structural configuration snapshots."""

    public_component = component.component
    return _ComponentConfigurationSnapshot(
        runtime_configuration=_object_configuration_snapshot(component),
        public_configuration=_object_configuration_snapshot(public_component),
    )


def _prepared_configuration_snapshot(
    *,
    components: Mapping[str, "_ComponentDeclaration"],
    clock: Clock,
    settings: Settings,
    constants: PhysicalConstants,
    runtime: RuntimeOptions,
) -> _PreparedConfigurationSnapshot:
    """Return the complete post-initialization configuration snapshot."""

    component_snapshots = MappingProxyType(
        {
            name: _component_configuration_snapshot(component)
            for name, component in components.items()
        }
    )
    return _PreparedConfigurationSnapshot(
        components=component_snapshots,
        clock=_clock_configuration_snapshot(clock),
        runtime=_runtime_configuration_snapshot(runtime),
        settings=_settings_configuration_snapshot(settings),
        constants=_constants_configuration_snapshot(constants),
    )


def _clock_configuration_snapshot(clock: Clock) -> tuple[Any, ...]:
    """Return identity and values for the configured clock."""

    return (id(clock), _configuration_value_snapshot(clock))


def _runtime_configuration_snapshot(runtime: RuntimeOptions) -> tuple[Any, ...]:
    """Return stable runtime-option identity and policy configuration."""

    execution = runtime.execution
    execution_snapshot = (
        execution if isinstance(execution, str) else (type(execution), id(execution))
    )
    return (
        id(runtime),
        _configuration_value_snapshot(runtime.dtype),
        execution_snapshot,
        _topology_configuration_snapshot(runtime.topology),
        runtime.model_year_seconds,
    )


def _topology_configuration_snapshot(topology: object | None) -> Any:
    """Return identity and mutable configuration for one topology policy."""

    if topology is None:
        return None
    return (
        type(topology),
        id(topology),
        _instance_configuration_snapshot(topology),
    )


def _settings_configuration_snapshot(settings: Settings) -> tuple[Any, ...]:
    """Return identity and values for one settings owner."""

    return (id(settings), _configuration_value_snapshot(settings.as_dict()))


def _constants_configuration_snapshot(
    constants: PhysicalConstants,
) -> tuple[Any, ...]:
    """Return identity and traced values for physical constants."""

    return (id(constants), _configuration_value_snapshot(constants))


def _object_configuration_snapshot(component: object) -> tuple[Any, ...]:
    """Return the supported mutation-sensitive component configuration."""

    name = getattr(component, "name", None)
    grid = getattr(component, "grid", None)
    spec = getattr(component, "spec", None)
    seeded_fields = getattr(component, "_data", None)
    settings = getattr(component, "settings", None)
    settings_values = settings.as_dict() if isinstance(settings, Settings) else {}
    return (
        id(component),
        name,
        id(grid),
        _configuration_value_snapshot(grid),
        id(spec),
        _configuration_value_snapshot(spec),
        tuple(
            (
                method_name,
                _callable_identity_snapshot(getattr(component, method_name, None)),
            )
            for method_name in ("initial_fields", "initialize", "step")
        ),
        _author_callable_configuration_snapshot(
            getattr(component, "_author_step", None),
            include_bound_owner_state=True,
        ),
        _field_configuration_snapshot(seeded_fields),
        id(settings),
        _configuration_value_snapshot(settings_values),
        _instance_configuration_snapshot(
            component,
            excluded_names=frozenset({"_component", "_setup_metadata"}),
        ),
    )


def _callable_identity_snapshot(value: object) -> tuple[Any, ...]:
    """Return stable callable identity without traversing operational state."""

    if not callable(value):
        return ("invalid", type(value))
    function = getattr(value, "__func__", None)
    owner = getattr(value, "__self__", None)
    if function is not None:
        return (type(value), id(function), id(owner))
    return (type(value), id(value))


def _author_callable_configuration_snapshot(
    value: object,
    *,
    include_bound_owner_state: bool = False,
    seen: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    """Return bounded configuration for the explicit author-step owner."""

    if not callable(value):
        return ("invalid", type(value))
    if id(value) in seen:
        return ("cycle", type(value))
    if isinstance(value, partial):
        nested_seen = seen | {id(value)}
        return (
            type(value),
            id(value),
            _author_callable_configuration_snapshot(
                value.func,
                include_bound_owner_state=True,
                seen=nested_seen,
            ),
            _configuration_value_snapshot(value.args, seen=nested_seen),
            _configuration_value_snapshot(value.keywords or {}, seen=nested_seen),
        )
    function = getattr(value, "__func__", None)
    owner = getattr(value, "__self__", None)
    if function is not None:
        snapshot: tuple[Any, ...] = (
            type(value),
            id(function),
            id(owner),
        )
        if include_bound_owner_state and owner is not None:
            return (
                *snapshot,
                _instance_configuration_snapshot(owner, seen=seen | {id(value)}),
            )
        return snapshot
    if isroutine(value):
        return (type(value), id(value))
    return (
        type(value),
        id(value),
        _instance_configuration_snapshot(value, seen=seen),
    )


def _instance_configuration_snapshot(
    value: object,
    *,
    seen: frozenset[int] = frozenset(),
    excluded_names: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    """Return bounded instance state without traversing functions or globals."""

    if id(value) in seen:
        return ("cycle", type(value))
    nested_seen = seen | {id(value)}
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping) and excluded_names:
        # Adapter ownership is captured separately, while setup metadata is
        # diagnostic state rather than runtime component configuration.
        state = {
            name: item for name, item in state.items() if name not in excluded_names
        }
    mapping_state = (
        _configuration_value_snapshot(state, seen=nested_seen)
        if isinstance(state, Mapping)
        else None
    )
    slot_state = []
    for owner in reversed(type(value).__mro__):
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot_name in slots:
            if slot_name in ("__dict__", "__weakref__") or slot_name in excluded_names:
                continue
            storage_name = slot_name
            if slot_name.startswith("__") and not slot_name.endswith("__"):
                storage_name = f"_{owner.__name__.lstrip('_')}{slot_name}"
            try:
                slot_value = getattr(value, storage_name)
            except AttributeError:
                continue
            slot_state.append(
                (
                    owner,
                    slot_name,
                    _configuration_value_snapshot(slot_value, seen=nested_seen),
                )
            )
    return (mapping_state, tuple(slot_state))


def _field_configuration_snapshot(values: object) -> tuple[Any, ...]:
    """Return field names plus array identity, shape, and dtype metadata."""

    if not isinstance(values, Mapping):
        return ("invalid", type(values).__qualname__)
    result = []
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None and dtype is None:
            value_snapshot = _configuration_value_snapshot(value)
        else:
            value_snapshot = _array_configuration_snapshot(value, shape, dtype)
        result.append((name, value_snapshot))
    return tuple(result)


def _configuration_value_snapshot(
    value: object,
    *,
    seen: frozenset[int] = frozenset(),
) -> Any:
    """Return a stable comparable snapshot of nested configuration values."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if id(value) in seen:
        return ("cycle", type(value))
    nested_seen = seen | {id(value)}
    if isinstance(value, Mapping):
        return tuple(
            (key, _configuration_value_snapshot(item, seen=nested_seen))
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            _configuration_value_snapshot(item, seen=nested_seen) for item in value
        )
    if callable(value):
        # Generic/spec/lifecycle callables are identity-only. Their defaults,
        # closures, globals, and callable-object state can hold legitimate
        # operational logs/counters and are not supported configuration owners.
        return _callable_identity_snapshot(value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (
                    field.name,
                    _configuration_value_snapshot(
                        getattr(value, field.name),
                        seen=nested_seen,
                    ),
                )
                for field in fields(value)
            ),
        )
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return _array_configuration_snapshot(value, shape, dtype)
    # Opaque external objects are supported by identity/replacement only. Their
    # hidden internals are not traversed because they may own unbounded graphs.
    return (type(value), id(value))


def _array_configuration_snapshot(
    value: object,
    shape: Any,
    dtype: Any,
) -> tuple[Any, ...]:
    """Return array identity and metadata without materializing its values."""

    return (
        id(value),
        tuple(shape) if shape is not None else None,
        None if dtype is None else str(dtype),
    )


def _raise_configuration_mutation(owner: str) -> None:
    """Raise the actionable direct-mutation error for a prepared coupling."""

    raise CouplerError(
        f"Coupler configuration for {owner} changed after preparation. Configure "
        "components, clock, runtime options, and settings before preparation, "
        "or create a new Coupler and prepare it again."
    )


__all__ = ["PreparedCoupling", "prepare_coupling"]
