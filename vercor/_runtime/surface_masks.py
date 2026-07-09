from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp

from vercor.exceptions import ComponentError, CouplerError
from vercor.grid_masks import (
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.grid_geometry import grids_identical
from vercor.jax_logging import LoggerLike
from vercor._regridders.conservative import ConservativeRectilinearRegridder
from vercor._runtime.component_topology import require_component
from vercor._runtime.topology_state import RuntimeTopologyMaps, SurfaceExchangeMasks
from vercor.topology import ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext

if TYPE_CHECKING:
    from vercor.exchanges import Exchange
    from vercor.grids import RectilinearGrid


class _SurfaceRoleComponent(Protocol):
    @property
    def name(self) -> str:
        """Return the registered component name."""
        ...

    @property
    def grid(self) -> "RectilinearGrid":
        """Return the component grid."""
        ...


def should_apply_surface_mask_policy(
    components: Mapping[str, _SurfaceRoleComponent],
    exchanges: Sequence["Exchange"],
    policy: SurfaceMaskPolicy | None,
) -> bool:
    """Return whether configured exchanges should use the surface-mask policy."""

    if policy is None or policy.mode == "disabled":
        return False
    if policy.mode == "required":
        return True
    role_names = (policy.atmosphere, policy.ocean, policy.land)
    if not all(name in components for name in role_names):
        return False
    return any(
        exchange.target == policy.atmosphere
        and exchange.source in (policy.ocean, policy.land)
        for exchange in exchanges
    )


def _require_surface_role(
    components: Mapping[str, _SurfaceRoleComponent],
    role_name: str,
) -> _SurfaceRoleComponent:
    """Return a surface-role component with a policy-oriented error."""

    try:
        return require_component(components, role_name)
    except CouplerError as exc:
        raise CouplerError(
            f"Surface mask policy requires role component {role_name!r} to be registered"
        ) from exc


def create_surface_exchange_masks(
    components: Mapping[str, _SurfaceRoleComponent],
    *,
    policy: SurfaceMaskPolicy,
    logger: LoggerLike,
) -> SurfaceExchangeMasks:
    """Create atmosphere-grid ocean/land masks required by exchange setup."""

    land_component = _require_surface_role(components, policy.land)
    atmosphere_component = _require_surface_role(components, policy.atmosphere)
    ocean_component = _require_surface_role(components, policy.ocean)

    if not grids_identical(land_component.grid, atmosphere_component.grid):
        raise CouplerError(
            "Land and atmospheric components must use identical horizontal grids"
        )

    regridder = ConservativeRectilinearRegridder(
        ocean_component.grid,
        atmosphere_component.grid,
    )

    ocean_binary_mask = ocean_component.grid.binary_mask
    if ocean_binary_mask is None:
        raise ComponentError(
            f"Ocean component {ocean_component.name} has no binary mask defined"
        )

    (
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

    check_remap_conservation(
        regridder,
        ocean_binary_mask,
        ocn_fmask_on_atm_grid,
        logger=logger,
    )

    check_total_lnd_ocn_mask_sum(lnd_fmask_on_atm_grid, ocn_fmask_on_atm_grid)
    return SurfaceExchangeMasks(
        ocn_fmask_on_atm_grid=ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid=lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid=lnd_bmask_on_atm_grid,
    )


def validate_land_mask_consistency(
    components: Mapping[str, _SurfaceRoleComponent],
    surface_masks: SurfaceExchangeMasks,
    *,
    policy: SurfaceMaskPolicy,
) -> None:
    """Validate that a component land mask matches the remapped exchange mask."""

    land_component = _require_surface_role(components, policy.land)
    lnd_mask_from_component = land_component.grid.binary_mask
    if lnd_mask_from_component is not None:
        component_mask = jnp.asarray(lnd_mask_from_component)
        remapped_mask = jnp.asarray(surface_masks.lnd_bmask_on_atm_grid)
        if component_mask.shape != surface_masks.lnd_bmask_on_atm_grid.shape:
            raise CouplerError(
                "Land binary mask read from component does not match atmospheric grid shape"
            )
        if not bool(jnp.all(component_mask == remapped_mask)):
            mismatch = int(jnp.count_nonzero(component_mask != remapped_mask))
            raise CouplerError(
                "Land binary mask created from remapped ocean mask does not match component-provided mask "
                f"(mismatched points: {mismatch})"
            )


def apply_surface_exchange_masks(
    topology_maps: RuntimeTopologyMaps,
    *,
    surface_masks: SurfaceExchangeMasks,
    policy: SurfaceMaskPolicy,
) -> RuntimeTopologyMaps:
    """Patch special land/ocean masks onto bilinear atmosphere exchanges."""

    for key in topology_maps.binary_masks.keys():
        source, destination, interp_type = key
        if "bilinear" in interp_type:
            if source == policy.ocean and destination == policy.atmosphere:
                topology_maps.fractional_masks[key] = (
                    surface_masks.ocn_fmask_on_atm_grid
                )
            elif source == policy.land and destination == policy.atmosphere:
                topology_maps.binary_masks[key] = surface_masks.lnd_bmask_on_atm_grid
                topology_maps.fractional_masks[key] = (
                    surface_masks.lnd_fmask_on_atm_grid
                )
    return topology_maps


def build_surface_mask_topology_patch(
    context: TopologyContext,
    policy: SurfaceMaskPolicy,
) -> tuple[ExchangeTopologyPatch, SurfaceExchangeMasks]:
    """Return public topology patch data plus private derived surface masks."""

    surface_masks = create_surface_exchange_masks(
        context.components,
        policy=policy,
        logger=context.logger,
    )
    validate_land_mask_consistency(
        context.components,
        surface_masks,
        policy=policy,
    )
    binary_masks = {}
    fractional_masks = {}
    for key in context.exchange_keys:
        source, destination, regrid_key = key
        if "bilinear" not in regrid_key:
            continue
        if source == policy.ocean and destination == policy.atmosphere:
            fractional_masks[key] = surface_masks.ocn_fmask_on_atm_grid
        elif source == policy.land and destination == policy.atmosphere:
            binary_masks[key] = surface_masks.lnd_bmask_on_atm_grid
            fractional_masks[key] = surface_masks.lnd_fmask_on_atm_grid
    return (
        ExchangeTopologyPatch(
            binary_masks=binary_masks,
            fractional_masks=fractional_masks,
        ),
        surface_masks,
    )


__all__ = [
    "apply_surface_exchange_masks",
    "build_surface_mask_topology_patch",
    "create_surface_exchange_masks",
    "should_apply_surface_mask_policy",
    "validate_land_mask_consistency",
]
