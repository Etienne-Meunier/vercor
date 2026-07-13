from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.errors import TracerBoolConversionError

from vercor.components.contracts import ComponentInfo
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.exchange_keys import exchange_regrid_key
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.settings import Settings
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
    TopologyPolicy,
)

if TYPE_CHECKING:
    from vercor.components.base import Component


def build_topology_context(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    logger: LoggerLike,
) -> TopologyContext:
    """Return public topology-policy context from private runtime inputs."""

    component_info = MappingProxyType(
        {
            name: ComponentInfo(
                name=component.name,
                grid=component.grid,
                spec=component.spec,
            )
            for name, component in components.items()
        }
    )
    return TopologyContext(
        components=component_info,
        exchanges=tuple(exchanges),
        exchange_keys=tuple(
            (exchange.source, exchange.target, exchange_regrid_key(exchange))
            for exchange in exchanges
        ),
        settings=settings,
        logger=logger,
    )


def apply_topology_policy(
    topology_maps: RuntimeTopologyMaps,
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: Settings,
    logger: LoggerLike,
    policy: TopologyPolicy | None,
) -> RuntimeTopologyMaps:
    """Apply an optional public topology policy to private runtime maps."""

    if policy is None:
        return topology_maps

    context = build_topology_context(
        components=components,
        exchanges=exchanges,
        settings=settings,
        logger=logger,
    )
    if not policy.applies(context):
        return topology_maps

    patch = policy.build(context)
    prepared_maps = _apply_exchange_topology_patch(
        topology_maps,
        patch,
        components=context.components,
    )
    logger.info("Exchange topology policy patching complete")
    return prepared_maps


def _apply_exchange_topology_patch(
    topology_maps: RuntimeTopologyMaps,
    patch: ExchangeTopologyPatch,
    *,
    components: Mapping[str, ComponentInfo],
) -> RuntimeTopologyMaps:
    """Apply public topology mask patches to runtime topology maps."""

    binary_masks = dict(topology_maps.binary_masks)
    fractional_masks = dict(topology_maps.fractional_masks)
    for key, value in patch.binary_masks.items():
        _validate_patch_item(topology_maps, components, key, value, "binary")
        binary_masks[key] = value
    for key, value in patch.fractional_masks.items():
        _validate_patch_item(topology_maps, components, key, value, "fractional")
        fractional_masks[key] = value
    return RuntimeTopologyMaps(
        regridders=topology_maps.regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )


def _validate_patch_item(
    topology_maps: RuntimeTopologyMaps,
    components: Mapping[str, ComponentInfo],
    key: tuple[str, str, str],
    value: object,
    mask_kind: str,
) -> None:
    """Validate one public topology patch item before map replacement."""

    if key not in topology_maps.regridders:
        raise CouplerError(
            f"Topology policy {mask_kind} mask key {key!r} does not match a "
            "configured topology key."
        )
    try:
        mask_array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CouplerError(
            f"Topology policy {mask_kind} mask for key {key!r} must be a "
            "concrete numeric or bool array."
        ) from exc
    is_real_numeric = (
        jnp.issubdtype(mask_array.dtype, jnp.bool_)
        or jnp.issubdtype(mask_array.dtype, jnp.integer)
        or jnp.issubdtype(mask_array.dtype, jnp.floating)
    )
    if not is_real_numeric:
        raise CouplerError(
            f"Topology policy {mask_kind} mask for key {key!r} must be a "
            "concrete numeric or bool array."
        )
    target_shape = components[key[1]].grid.shape
    mask_shape = mask_array.shape
    if mask_shape != target_shape:
        raise CouplerError(
            f"Topology policy {mask_kind} mask for key {key!r} has shape "
            f"{mask_shape}, expected {target_shape} for target component {key[1]!r}."
        )
    try:
        all_finite = bool(jnp.all(jnp.isfinite(mask_array)))
        all_binary = bool(jnp.all(jnp.logical_or(mask_array == 0, mask_array == 1)))
        all_fractional = bool(
            jnp.all(jnp.logical_and(mask_array >= 0, mask_array <= 1))
        )
    except TracerBoolConversionError as exc:
        raise CouplerError(
            f"Topology policy {mask_kind} mask for key {key!r} must be a "
            "concrete numeric or bool array."
        ) from exc
    if not all_finite:
        raise CouplerError(
            f"Topology policy {mask_kind} mask for key {key!r} must contain "
            "only finite values."
        )
    if mask_kind == "binary" and not all_binary:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must contain only "
            "values in {0, 1}."
        )
    if mask_kind == "fractional" and not all_fractional:
        raise CouplerError(
            f"Topology policy fractional mask for key {key!r} must contain "
            "values in [0, 1]."
        )


__all__ = [
    "apply_topology_policy",
    "build_topology_context",
]
