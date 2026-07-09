from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.components.setup_validation import validate_component_setup
from vercor.components.contexts import SetupContext
from vercor.dtypes import as_jax_real_array
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.contracts import ExchangeContract, build_exchange_contracts
from vercor._runtime.topology import build_exchange_topology
from vercor._runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
)
from vercor._runtime.validation import (
    check_not_empty_import_export_lists,
    validate_exchange_fields_declared,
)
from vercor.settings import Settings
from vercor.config import SurfaceMaskPolicy

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass(frozen=True)
class RuntimeInitializationState:
    """Validated setup-time state required by the runtime facade."""

    runtime_contracts: dict[str, ExchangeContract]
    topology: ExchangeTopologyState


def apply_run_precision_to_component(
    component: Component,
    settings: Settings,
) -> None:
    """Synchronize component-owned setup arrays with the coupler precision."""

    component.settings.set("enable_x64", settings.enable_x64)
    component.grid = component.grid.with_precision(settings)
    component._data = {
        field_name: as_jax_real_array(field_value, settings)
        for field_name, field_value in component._data.items()
    }
    spec = component.spec
    if spec.defaults:
        component.declare_fields(
            inputs=spec.inputs,
            outputs=spec.outputs,
            defaults=spec.defaults,
        )


def initialize_coupler_runtime(
    *,
    clock: Clock,
    components: dict[str, Component],
    exchanges: Sequence[Exchange],
    run_order: Sequence[str],
    settings: Settings,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
    surface_mask_policy: SurfaceMaskPolicy | None = SurfaceMaskPolicy(),
) -> RuntimeInitializationState:
    """Initialize components, contracts, and exchange topology for a coupler."""

    logger.info(" Initializing coupler and components")

    logger.info(
        f" Setting default precision for JAX computations: {settings.enable_x64}"
    )

    if settings.enable_x64:
        import jax

        jax.config.update("jax_enable_x64", True)

    for component in components.values():
        validate_component_setup(component)

    for component in components.values():
        apply_run_precision_to_component(component, settings)

    init_context = SetupContext(
        start=clock.start,
        dt_seconds=clock.dt_seconds,
        run_order=run_order,
        settings=settings,
        logger=logger,
    )

    for name, component in components.items():
        component.initialize(init_context)
        validate_component_setup(component)
        logger.info(f" Initialized {name}")

    runtime_contracts = build_exchange_contracts(
        tuple(components),
        exchanges,
        validate_endpoints=True,
    )

    for name, component in components.items():
        contract = runtime_contracts[name]
        if contract.all_fields:
            check_not_empty_import_export_lists(component, contract)
            validate_exchange_fields_declared(component, contract)

    topology = build_exchange_topology(
        components=components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        surface_mask_policy=surface_mask_policy,
        settings=settings,
        logger=logger,
    )
    return RuntimeInitializationState(
        runtime_contracts=runtime_contracts,
        topology=topology,
    )
