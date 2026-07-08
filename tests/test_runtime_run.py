from __future__ import annotations

from datetime import datetime
import inspect
import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp

from tests._coverage_support import make_test_grid
from tests._runtime_helpers import (
    replace_runtime_topology_maps,
)
from vercor.clock import Clock
from vercor.setups.slab.atmosphere import make_slab_atmosphere
from vercor.setups.slab.land import make_slab_land
from vercor.setups.slab.ocean import make_slab_ocean
from vercor.setups.slab.seaice import make_slab_seaice
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor._runtime.state import ComponentRuntimeState
from vercor.state import RunState
from vercor._runtime.stores import FieldStore


class _IdentityRegridder:
    def regrid(self, field: Any) -> Any:
        return jnp.asarray(field)

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        return jnp.asarray(u), jnp.asarray(v)


def _identity_factory(*args: Any, **kwargs: Any) -> _IdentityRegridder:
    _ = args, kwargs
    return _IdentityRegridder()


def _component_state(
    name: str,
    data: dict[str, jax.Array],
    receives: tuple[str, ...],
    sends: tuple[str, ...],
) -> ComponentRuntimeState:
    _ = name
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return ComponentRuntimeState(
        fields=FieldStore.from_mapping(
            {
                field: data.get(field, zeros)
                for field in sorted(set(data) | set(receives) | set(sends))
            }
        ),
        received=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in receives}
        ),
        sent=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in sends}
        ),
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="runtime-run")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=(
            make_slab_atmosphere(grid),
            make_slab_ocean(grid),
            make_slab_land(grid),
            make_slab_seaice(grid),
        ),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="OCN",
                fields=["sensible_heat_flux", "latent_heat_flux"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=["latent_heat_flux"],
                regrid=cast(Any, _identity_factory),
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=["sea_surface_temperature"],
                regrid=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "ATM",
            "OCN",
            "LND",
            "ICE",
        ),
    )
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "OCN", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "LND", "_identity_factory"): _IdentityRegridder(),
            ("OCN", "ICE", "_identity_factory"): _IdentityRegridder(),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            key: jnp.ones((2, 2), dtype=jnp.float64) for key in regridders
        },
    )
    return coupler


def _make_initial_state(sea_surface_temperature: jax.Array) -> RunState:
    zeros = jnp.zeros_like(sea_surface_temperature)
    temperature_2m = jnp.full_like(sea_surface_temperature, 288.15)
    components = (
        _component_state(
            "ATM",
            {
                "temperature_2m": temperature_2m,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
                "u_velocity_10m": zeros,
                "v_velocity_10m": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            receives=("sea_surface_temperature",),
            sends=(
                "temperature_2m",
                "sensible_heat_flux",
                "latent_heat_flux",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
        ),
        _component_state(
            "OCN",
            {
                "sea_surface_temperature": sea_surface_temperature,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
            },
            receives=("sensible_heat_flux", "latent_heat_flux"),
            sends=("sea_surface_temperature",),
        ),
        _component_state(
            "LND",
            {
                "soil_moisture": jnp.full_like(sea_surface_temperature, 0.3),
                "land_surface_temperature": temperature_2m,
                "latent_heat_flux": zeros,
            },
            receives=("latent_heat_flux",),
            sends=("soil_moisture", "land_surface_temperature"),
        ),
        _component_state(
            "ICE",
            {
                "ice_fraction": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            receives=("sea_surface_temperature",),
            sends=("ice_fraction",),
        ),
    )
    return RunState._from_runtime(
        component_names=("ATM", "OCN", "LND", "ICE"),
        components=components,
        fractional_masks=FieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|OCN|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|LND|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "OCN|ICE|_identity_factory": jnp.ones_like(sea_surface_temperature),
            }
        ),
    )


def _runtime_state_with_sst(value: float) -> RunState:
    return _make_initial_state(jnp.full((2, 2), value, dtype=jnp.float64))


def _block_until_ready(value: RunState) -> RunState:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _runtime_treedef_repr(value: RunState) -> str:
    return repr(jax.tree_util.tree_structure(value))


def test_run_executes_pure_scanned_runtime_for_same_shapes_and_metadata() -> None:
    coupler = _make_coupler(steps=2)

    first = _block_until_ready(coupler.run(_runtime_state_with_sst(288.15)))
    second = _block_until_ready(coupler.run(_runtime_state_with_sst(291.15)))

    assert first._component_state("OCN").fields.get(
        "sea_surface_temperature"
    ).shape == (
        2,
        2,
    )
    assert second._component_state("OCN").fields.get(
        "sea_surface_temperature"
    ).shape == (
        2,
        2,
    )


def test_run_api_does_not_expose_state_donation() -> None:
    signature = inspect.signature(Coupler.run)

    assert "donate_state" not in signature.parameters


def test_run_preserves_runtime_treedef() -> None:
    coupler = _make_coupler(steps=1)

    first_state = _runtime_state_with_sst(287.15)
    second_state = _runtime_state_with_sst(292.15)
    first_final = _block_until_ready(coupler.run(first_state))
    second_final = _block_until_ready(coupler.run(second_state))

    expected_treedef = _runtime_treedef_repr(first_state)
    assert _runtime_treedef_repr(second_state) == expected_treedef
    assert _runtime_treedef_repr(first_final) == expected_treedef
    assert _runtime_treedef_repr(second_final) == expected_treedef
    assert first_final.component_names == first_state.component_names

    for before, after in zip(first_state._components, first_final._components):
        assert after.fields.field_names == before.fields.field_names
        assert after.received.field_names == before.received.field_names
        assert after.sent.field_names == before.sent.field_names


def test_runtime_profile_harness_exposes_cli_entrypoint() -> None:
    profile_runtime = importlib.import_module("examples.profile_runtime")

    assert callable(profile_runtime.main)
    parser = profile_runtime.build_parser()
    args = parser.parse_args(["--steps", "3", "--log-level", "WARNING"])
    assert args.steps == 3
    assert args.log_level == "WARNING"
    assert not hasattr(args, "donate_state")


def test_runtime_profile_harness_runs_small_slab_profile() -> None:
    profile_runtime = importlib.import_module("examples.profile_runtime")

    result = profile_runtime.profile_runtime(
        steps=1,
        grid_nx=2,
        grid_ny=2,
        log_level="WARNING",
    )

    assert result.run_seconds >= 0.0
    assert not hasattr(result, "compiled_cache_entries")
    assert result.final_state_leaves > 0
