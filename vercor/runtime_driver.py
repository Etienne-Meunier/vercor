from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from logging import Logger
from typing import Any

from vercor.clock import ModelDateTime
from vercor.components.base import Component, HostRuntimeComponent, RuntimeStepContext
from vercor.exchange import Exchange
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeCouplerState,
    RuntimeStepInfo,
    dispatch_component_exchanges,
)
from vercor.runtime_components import receive_runtime_fields, send_runtime_fields
from vercor.settings import VercorSettings


def host_component_names(components: Mapping[str, Component]) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if isinstance(component, HostRuntimeComponent)
    ]


def _step_component_runtime_state(
    runtime_state: RuntimeCouplerState,
    component_name: str,
    step_info: RuntimeStepInfo,
    *,
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    dt_seconds: float,
    settings: VercorSettings,
    host_enabled: bool,
    time: datetime | ModelDateTime | None,
    logger: Logger | None,
) -> RuntimeCouplerState:
    """Advance one component through dispatch, receive, step, and send phases."""

    runtime_state = dispatch_component_exchanges(
        runtime_state,
        component_name,
        exchanges,
        regridders,
    )
    component_state = runtime_state.get_component_state(component_name)
    component = components[component_name]
    contract = contracts.get(
        component_name,
        RuntimeComponentContract.empty(),
    )
    component_state = receive_runtime_fields(
        component_state,
        contract,
    )
    step_context = RuntimeStepContext(
        dt_seconds=dt_seconds,
        settings=settings,
        time=time,
        logger=logger,
    )
    if host_enabled and isinstance(component, HostRuntimeComponent):
        component_state = component._step_host_runtime_state(
            component_state,
            step_context,
        )
    else:
        component_state = component.step_runtime_state(
            component_state,
            step_context,
        )
    component_state = send_runtime_fields(
        component,
        component_state,
        step_info,
        contract=contract,
    )
    return runtime_state.set_component_state(
        component_name,
        component_state,
    )


def step_runtime_component_pure(
    runtime_state: RuntimeCouplerState,
    component_name: str,
    step_info: RuntimeStepInfo,
    *,
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    dt_seconds: float,
    settings: VercorSettings,
) -> RuntimeCouplerState:
    """Advance one differentiable component on the pure scanned runtime path."""

    return _step_component_runtime_state(
        runtime_state,
        component_name,
        step_info,
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        contracts=contracts,
        dt_seconds=dt_seconds,
        settings=settings,
        host_enabled=False,
        time=None,
        logger=None,
    )


def prime_runtime_outgoing(
    runtime_state: RuntimeCouplerState,
    run_sequence: Sequence[str],
    *,
    components: Mapping[str, Component],
    contracts: Mapping[str, RuntimeComponentContract],
    step_info: RuntimeStepInfo,
) -> RuntimeCouplerState:
    """Populate outgoing stores once before the first exchange dispatch."""

    for component_name in run_sequence:
        component_state = runtime_state.get_component_state(component_name)
        component_state = send_runtime_fields(
            components[component_name],
            component_state,
            step_info,
            contract=contracts[component_name],
        )
        runtime_state = runtime_state.set_component_state(
            component_name,
            component_state,
        )
    return runtime_state


def step_runtime_component_host_enabled(
    runtime_state: RuntimeCouplerState,
    component_name: str,
    step_info: RuntimeStepInfo,
    *,
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    dt_seconds: float,
    settings: VercorSettings,
    time: datetime | ModelDateTime,
    logger: Logger,
) -> RuntimeCouplerState:
    """Advance one component on the Python runtime path with host adapters enabled."""

    return _step_component_runtime_state(
        runtime_state,
        component_name,
        step_info,
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        contracts=contracts,
        dt_seconds=dt_seconds,
        settings=settings,
        host_enabled=True,
        time=time,
        logger=logger,
    )
