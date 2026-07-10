"""Single prepared-coupling boundary for private runtime orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax

from vercor.clock import Clock
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
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
    from vercor.components.base import Component


@dataclass(frozen=True)
class _ComponentConfigurationFingerprint:
    """Comparable post-initialization component configuration snapshot."""

    runtime_configuration: tuple[Any, ...]
    public_configuration: tuple[Any, ...]


@dataclass(frozen=True)
class PreparedCoupling:
    """Immutable static runtime resources prepared once for one Coupler."""

    components: Mapping[str, "Component"]
    exchanges: tuple[Exchange, ...]
    run_order: tuple[str, ...]
    contracts: Mapping[str, ExchangeContract]
    topology_maps: RuntimeTopologyMaps
    dispatch_context: RuntimeDispatchContext
    clock: Clock
    settings: Settings
    runtime: RuntimeOptions
    interrupts: RuntimeInterruptController
    component_fingerprints: Mapping[str, _ComponentConfigurationFingerprint]

    def validate_component_configuration(
        self,
        components: Mapping[str, "Component"],
    ) -> None:
        """Reject direct component mutation after this boundary was prepared."""

        current_names = tuple(components)
        if current_names != tuple(self.component_fingerprints):
            _raise_component_mutation("registered component mapping")
        for registered_name, component in components.items():
            current = _component_fingerprint(component)
            if current != self.component_fingerprints[registered_name]:
                _raise_component_mutation(registered_name)


def prepare_coupling(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    run_order: Sequence[str],
    clock: Clock,
    settings: Settings,
    runtime: RuntimeOptions,
    logger: LoggerLike,
) -> PreparedCoupling:
    """Initialize lifecycle state and build one immutable runtime boundary."""

    _ensure_jax_precision_capability(runtime)
    immutable_components = MappingProxyType(dict(components))
    immutable_exchanges = tuple(exchanges)
    immutable_run_order = tuple(run_order)
    initialized = initialize_coupler_runtime(
        clock=clock,
        components=dict(immutable_components),
        exchanges=immutable_exchanges,
        run_order=immutable_run_order,
        settings=settings,
        logger=logger,
        topology_policy=runtime.topology,
    )
    contracts = MappingProxyType(dict(initialized.runtime_contracts))
    topology_maps = initialized.topology.topology_maps
    dispatch_context = build_runtime_dispatch_context(
        immutable_components,
        immutable_exchanges,
        topology_maps.regridders,
        contracts,
        dt_seconds=clock.dt_seconds,
        settings=settings,
    )
    fingerprints = MappingProxyType(
        {
            name: _component_fingerprint(component)
            for name, component in immutable_components.items()
        }
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
        runtime=runtime,
        interrupts=RuntimeInterruptController(),
        component_fingerprints=fingerprints,
    )


def _ensure_jax_precision_capability(runtime: RuntimeOptions) -> None:
    """Enable process x64 capability when the requested allocation policy needs it."""

    if runtime.dtype.enable_x64 and not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)


def _component_fingerprint(
    component: "Component",
) -> _ComponentConfigurationFingerprint:
    """Return runtime and public-owner configuration fingerprints."""

    public_component = getattr(component, "_component", component)
    return _ComponentConfigurationFingerprint(
        runtime_configuration=_object_configuration(component),
        public_configuration=_object_configuration(public_component),
    )


def _object_configuration(component: object) -> tuple[Any, ...]:
    """Return the minimum mutation-sensitive configuration for a component."""

    name = getattr(component, "name", None)
    grid = getattr(component, "grid", None)
    spec = getattr(component, "spec", None)
    initial_fields = getattr(component, "initial_fields", None)
    seeded_fields = initial_fields() if callable(initial_fields) else {}
    settings = getattr(component, "settings", None)
    settings_values = settings.as_dict() if isinstance(settings, Settings) else {}
    return (
        name,
        id(grid),
        getattr(grid, "shape", None),
        id(spec),
        _configuration_value(spec),
        _seeded_field_fingerprint(seeded_fields),
        _configuration_value(settings_values),
    )


def _seeded_field_fingerprint(values: object) -> tuple[Any, ...]:
    """Return field names plus array identity, shape, and dtype metadata."""

    if not isinstance(values, Mapping):
        return ("invalid", type(values).__qualname__)
    result = []
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None and dtype is None:
            value_fingerprint = _configuration_value(value)
        else:
            value_fingerprint = (
                id(value),
                tuple(shape) if shape is not None else None,
                None if dtype is None else str(dtype),
            )
        result.append((name, value_fingerprint))
    return tuple(result)


def _configuration_value(value: object) -> Any:
    """Return a stable comparable snapshot of nested configuration values."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return tuple((key, _configuration_value(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_configuration_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, _configuration_value(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return (
            id(value),
            tuple(shape) if shape is not None else None,
            None if dtype is None else str(dtype),
        )
    if callable(value):
        return (type(value), id(value))
    return (type(value), repr(value))


def _raise_component_mutation(component_name: str) -> None:
    """Raise the actionable direct-mutation error for a prepared coupling."""

    raise CouplerError(
        f"Component {component_name!r} changed after preparation. Configure "
        "components before preparation, or create/reconfigure the Coupler "
        "through its public mutators before preparing it again."
    )


__all__ = ["PreparedCoupling", "prepare_coupling"]
