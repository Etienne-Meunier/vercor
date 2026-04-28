from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components.slab.atmosphere import Atmosphere
from vercor.components.slab.land import Land
from vercor.components.slab.ocean import Ocean
from vercor.components.slab.seaice import SeaIce
from vercor.coupler import Coupler
from vercor.exchange import Exchange
from vercor.run_sequence import RunSequence
from vercor.runtime import RuntimeComponentState, RuntimeCouplerState, RuntimeFieldStore


class _IdentityRegridder:
    def __call__(self, *args: Any) -> Any:
        if len(args) == 1:
            return jnp.asarray(args[0])
        return tuple(jnp.asarray(arg) for arg in args)


def _identity_factory(*args: Any, **kwargs: Any) -> _IdentityRegridder:
    _ = args, kwargs
    return _IdentityRegridder()


def _component_state(
    name: str,
    data: dict[str, jax.Array],
    imports: tuple[str, ...],
    exports: tuple[str, ...],
) -> RuntimeComponentState:
    _ = name
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                field: data.get(field, zeros)
                for field in sorted(set(data) | set(imports) | set(exports))
            }
        ),
        incoming=RuntimeFieldStore.from_mapping(
            {field: data.get(field, zeros) for field in imports}
        ),
        outgoing=RuntimeFieldStore.from_mapping(
            {field: data.get(field, zeros) for field in exports}
        ),
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="compile-cache")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.components = {
        "ATM": Atmosphere(grid),
        "OCN": Ocean(grid),
        "LND": Land(grid),
        "ICE": SeaIce(grid),
    }
    coupler.run_sequence = RunSequence(order=["ATM", "OCN", "LND", "ICE"])
    coupler.exchanges = [
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sea_surface_temperature"],
            regridder_factory=cast(Any, _identity_factory),
        ),
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["sensible_heat_flux", "latent_heat_flux"],
            regridder_factory=cast(Any, _identity_factory),
        ),
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["latent_heat_flux"],
            regridder_factory=cast(Any, _identity_factory),
        ),
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sea_surface_temperature"],
            regridder_factory=cast(Any, _identity_factory),
        ),
    ]
    coupler._regridders = cast(
        Any,
        {
            ("OCN", "ATM", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "OCN", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "LND", "_identity_factory"): _IdentityRegridder(),
            ("OCN", "ICE", "_identity_factory"): _IdentityRegridder(),
        },
    )
    coupler._fractional_masks = {
        key: jnp.ones((2, 2), dtype=jnp.float64) for key in coupler._regridders
    }
    return coupler


def _make_initial_state(sea_surface_temperature: jax.Array) -> RuntimeCouplerState:
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
            imports=("sea_surface_temperature",),
            exports=(
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
            imports=("sensible_heat_flux", "latent_heat_flux"),
            exports=("sea_surface_temperature",),
        ),
        _component_state(
            "LND",
            {
                "soil_moisture": jnp.full_like(sea_surface_temperature, 0.3),
                "land_surface_temperature": temperature_2m,
                "latent_heat_flux": zeros,
            },
            imports=("latent_heat_flux",),
            exports=("soil_moisture", "land_surface_temperature"),
        ),
        _component_state(
            "ICE",
            {
                "ice_fraction": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            imports=("sea_surface_temperature",),
            exports=("ice_fraction",),
        ),
    )
    return RuntimeCouplerState(
        component_names=("ATM", "OCN", "LND", "ICE"),
        components=components,
        fractional_masks=RuntimeFieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|OCN|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|LND|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "OCN|ICE|_identity_factory": jnp.ones_like(sea_surface_temperature),
            }
        ),
        binary_masks=RuntimeFieldStore.empty(),
    )


def _runtime_state_with_sst(value: float) -> RuntimeCouplerState:
    return _make_initial_state(jnp.full((2, 2), value, dtype=jnp.float64))


def _block_until_ready(value: RuntimeCouplerState) -> RuntimeCouplerState:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _runtime_treedef_repr(value: RuntimeCouplerState) -> str:
    return repr(jax.tree_util.tree_structure(value))


def test_compiled_runtime_reuses_cache_for_same_shapes_and_metadata() -> None:
    coupler = _make_coupler(steps=2)
    compiled = coupler.compile_runtime(donate_state=False)
    compiled_for_cache = cast(Any, compiled)
    compiled_for_cache.clear_cache()

    first = _block_until_ready(compiled(_runtime_state_with_sst(288.15)))
    first_cache_size = compiled_for_cache._cache_size()

    second = _block_until_ready(compiled(_runtime_state_with_sst(291.15)))
    second_cache_size = compiled_for_cache._cache_size()

    assert first_cache_size == 1
    assert second_cache_size == first_cache_size
    assert first.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    ).shape == (2, 2)
    assert second.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    ).shape == (2, 2)


def test_compiled_runtime_donation_runs_with_fresh_consumed_state() -> None:
    coupler = _make_coupler(steps=2)
    compiled = coupler.compile_runtime(donate_state=True)
    compiled_for_cache = cast(Any, compiled)
    compiled_for_cache.clear_cache()

    final_state = _block_until_ready(compiled(_runtime_state_with_sst(289.15)))

    assert compiled_for_cache._cache_size() == 1
    assert final_state.component_names == ("ATM", "OCN", "LND", "ICE")
    assert np.all(
        np.isfinite(
            np.asarray(
                final_state.get_component_state("OCN").data.get(
                    "sea_surface_temperature"
                )
            )
        )
    )


def test_non_donating_compiled_runtime_preserves_runtime_treedef() -> None:
    coupler = _make_coupler(steps=1)
    compiled = coupler.compile_runtime(donate_state=False)

    first_state = _runtime_state_with_sst(287.15)
    second_state = _runtime_state_with_sst(292.15)
    first_final = _block_until_ready(compiled(first_state))
    second_final = _block_until_ready(compiled(second_state))

    expected_treedef = _runtime_treedef_repr(first_state)
    assert _runtime_treedef_repr(second_state) == expected_treedef
    assert _runtime_treedef_repr(first_final) == expected_treedef
    assert _runtime_treedef_repr(second_final) == expected_treedef
    assert first_final.component_names == first_state.component_names

    for before, after in zip(first_state.components, first_final.components):
        assert after.data.field_names == before.data.field_names
        assert after.incoming.field_names == before.incoming.field_names
        assert after.outgoing.field_names == before.outgoing.field_names
