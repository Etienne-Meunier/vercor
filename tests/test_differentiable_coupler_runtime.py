from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import DummyComponent, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components.slab.atmosphere import Atmosphere
from vercor.components.slab.land import Land
from vercor.components.slab.ocean import Ocean
from vercor.components.slab.seaice import SeaIce
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.exchange import Exchange
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear, conservative
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
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return RuntimeComponentState(
        name=name,
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
        fields_to_import=imports,
        fields_to_export=exports,
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="slab")
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
    key = ("OCN", "ATM", "_identity_factory")
    coupler._regridders = cast(
        Any,
        {
            key: _IdentityRegridder(),
            ("ATM", "OCN", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "LND", "_identity_factory"): _IdentityRegridder(),
            ("OCN", "ICE", "_identity_factory"): _IdentityRegridder(),
        },
    )
    coupler._fractional_masks = {
        runtime_key: jnp.ones((2, 2)) for runtime_key in coupler._regridders
    }
    return coupler


def _make_initialized_slab_coupler(steps: int) -> Coupler:
    longitude = np.asarray([0.0, 1.0], dtype=float)
    latitude = np.asarray([-1.0, 1.0], dtype=float)
    ocean_mask = np.ones((2, 2), dtype=float)
    land_mask = np.zeros((2, 2), dtype=float)

    atmosphere_grid = make_test_grid(
        name="ATM",
        longitude=longitude,
        latitude=latitude,
    )
    ocean_grid = make_test_grid(
        name="OCN",
        longitude=longitude,
        latitude=latitude,
        binary_mask=ocean_mask,
    )
    land_grid = make_test_grid(
        name="LND",
        longitude=longitude,
        latitude=latitude,
        binary_mask=land_mask,
    )
    ice_grid = make_test_grid(
        name="ICE",
        longitude=longitude,
        latitude=latitude,
    )

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.register(Atmosphere(atmosphere_grid))
    coupler.register(Ocean(ocean_grid))
    coupler.register(Land(land_grid))
    coupler.register(SeaIce(ice_grid))
    coupler.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["land_surface_temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ICE",
            destination="ATM",
            field_names=["ice_fraction"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["sensible_heat_flux", "latent_heat_flux"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["latent_heat_flux"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.set_components_run_sequence(RunSequence(order=["ATM", "OCN", "LND", "ICE"]))
    coupler.initialize()
    return coupler


def _make_initialized_mixed_grid_slab_coupler(steps: int) -> Coupler:
    atmosphere_longitude = np.asarray([0.0, 1.0], dtype=float)
    atmosphere_latitude = np.asarray([-1.0, 1.0], dtype=float)
    atmosphere_longitude_edges = np.asarray([-0.25, 0.5, 1.25], dtype=float)
    atmosphere_latitude_edges = np.asarray([-1.5, 0.0, 1.5], dtype=float)
    ocean_longitude = np.asarray([0.0, 0.5, 1.0], dtype=float)
    ocean_latitude = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    ocean_longitude_edges = np.asarray([-0.25, 0.25, 0.75, 1.25], dtype=float)
    ocean_latitude_edges = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)
    ocean_mask = np.ones((3, 3), dtype=float)
    land_mask = np.zeros((2, 2), dtype=float)

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atmosphere_longitude,
        latitude=atmosphere_latitude,
        longitude_edges=atmosphere_longitude_edges,
        latitude_edges=atmosphere_latitude_edges,
    )
    ocean_grid = RectilinearGrid(
        name="OCN",
        longitude=ocean_longitude,
        latitude=ocean_latitude,
        longitude_edges=ocean_longitude_edges,
        latitude_edges=ocean_latitude_edges,
        binary_mask=ocean_mask,
    )
    land_grid = RectilinearGrid(
        name="LND",
        longitude=atmosphere_longitude,
        latitude=atmosphere_latitude,
        longitude_edges=atmosphere_longitude_edges,
        latitude_edges=atmosphere_latitude_edges,
        binary_mask=land_mask,
    )
    ice_grid = RectilinearGrid(
        name="ICE",
        longitude=ocean_longitude,
        latitude=ocean_latitude,
        longitude_edges=ocean_longitude_edges,
        latitude_edges=ocean_latitude_edges,
    )

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.register(Atmosphere(atmosphere_grid))
    coupler.register(Ocean(ocean_grid))
    coupler.register(Land(land_grid))
    coupler.register(SeaIce(ice_grid))
    coupler.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sea_surface_temperature"],
            regridder_factory=conservative,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["land_surface_temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ICE",
            destination="ATM",
            field_names=["ice_fraction"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["sensible_heat_flux", "latent_heat_flux"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["latent_heat_flux"],
            regridder_factory=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="OCN",
            destination="ICE",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.set_components_run_sequence(RunSequence(order=["ATM", "OCN", "LND", "ICE"]))
    coupler.initialize()
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


def _with_ocean_sst(
    state: RuntimeCouplerState, sea_surface_temperature: jax.Array
) -> RuntimeCouplerState:
    ocean = state.get_component_state("OCN")
    ocean = ocean.with_data(
        ocean.data.set("sea_surface_temperature", sea_surface_temperature)
    )
    ocean = ocean.with_outgoing(
        ocean.outgoing.set("sea_surface_temperature", sea_surface_temperature)
    )
    return state.set_component_state(ocean)


def test_run_differentiable_supports_jit_grad_and_jvp() -> None:
    coupler = _make_coupler(steps=2)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    initial_state = _make_initial_state(initial_sst)

    final_state = jax.jit(lambda state: coupler.run_differentiable(state))(
        initial_state
    )
    ocean_sst = final_state.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    )

    assert ocean_sst.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sst: jax.Array) -> jax.Array:
        state = _make_initial_state(sst)
        result = coupler.run_differentiable(state)
        return jnp.sum(
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    _, tangent = jax.jvp(loss, (initial_sst,), (jnp.ones_like(initial_sst),))

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(tangent))


def test_run_differentiable_matches_one_step_closed_form_for_slab_ocean() -> None:
    coupler = _make_coupler(steps=1)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    final_state = coupler.run_differentiable(_make_initial_state(initial_sst))

    ocean_sst = final_state.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    )
    sensible = -10.0 * (288.15 - 286.15)
    latent = -0.5 * sensible
    restoring = (np.asarray(initial_sst) - 288.15) / (30.0 * 86400.0)
    expected = (
        np.asarray(initial_sst)
        + ((sensible + latent) / (1025.0 * 3990.0 * 30.0) + restoring) * 3600.0
    )

    assert_allclose_compact(ocean_sst, expected)


def test_initialized_slab_coupler_creates_jittable_differentiable_state() -> None:
    coupler = _make_initialized_slab_coupler(steps=2)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    initial_state = _with_ocean_sst(
        coupler.create_differentiable_state(),
        initial_sst,
    )

    final_state = jax.jit(lambda state: coupler.run_differentiable(state))(
        initial_state
    )
    ocean_sst = final_state.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    )

    assert final_state.component_names == ("ATM", "OCN", "LND", "ICE")
    assert ocean_sst.shape == (2, 2)
    assert isinstance(ocean_sst, jax.Array)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        state = _with_ocean_sst(initial_state, sea_surface_temperature)
        result = coupler.run_differentiable(state)
        return jnp.sum(
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_mixed_grid_slab_coupler_runs_with_real_regridders_under_jit_grad_and_jvp() -> (
    None
):
    coupler = _make_initialized_mixed_grid_slab_coupler(steps=2)
    initial_state = coupler.create_differentiable_state()
    initial_sst = jnp.linspace(285.15, 287.15, 9, dtype=jnp.float64).reshape((3, 3))
    initial_state = _with_ocean_sst(initial_state, initial_sst)

    final_state = jax.jit(lambda state: coupler.run_differentiable(state))(
        initial_state
    )

    atmosphere = final_state.get_component_state("ATM")
    ocean = final_state.get_component_state("OCN")
    ice = final_state.get_component_state("ICE")
    atmosphere_sst = atmosphere.incoming.get("sea_surface_temperature")
    ocean_sst = ocean.data.get("sea_surface_temperature")
    ice_sst = ice.incoming.get("sea_surface_temperature")

    assert atmosphere_sst.shape == (2, 2)
    assert ocean_sst.shape == (3, 3)
    assert ice_sst.shape == (3, 3)
    assert isinstance(atmosphere_sst, jax.Array)
    assert isinstance(ocean_sst, jax.Array)
    assert isinstance(ice_sst, jax.Array)
    assert np.all(np.isfinite(np.asarray(atmosphere_sst)))
    assert np.all(np.isfinite(np.asarray(ocean_sst)))
    assert np.all(np.isfinite(np.asarray(ice_sst)))

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        state = _with_ocean_sst(initial_state, sea_surface_temperature)
        result = coupler.run_differentiable(state)
        return jnp.sum(
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    _, tangent = jax.jvp(loss, (initial_sst,), (jnp.ones_like(initial_sst),))

    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(tangent))


def test_run_differentiable_validates_missing_run_sequence() -> None:
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )

    with pytest.raises(CouplerError, match="run sequence"):
        coupler.run_differentiable()


def test_run_differentiable_rejects_non_slab_components() -> None:
    grid = make_test_grid(name="dummy")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": cast(Any, DummyComponent("ATM", grid))}
    coupler.run_sequence = RunSequence(order=["ATM"])

    with pytest.raises(ComponentError, match="slab components only"):
        coupler.run_differentiable()


def test_run_differentiable_validates_regridders_and_fractional_masks() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    coupler._regridders = {}

    with pytest.raises(CouplerError, match="initialized regridder"):
        coupler.run_differentiable(state)

    coupler = _make_coupler(steps=1)
    state = RuntimeCouplerState(
        components=state.components,
        fractional_masks=RuntimeFieldStore.empty(),
        binary_masks=RuntimeFieldStore.empty(),
    )

    with pytest.raises(CouplerError, match="fractional mask"):
        coupler.run_differentiable(state)


def test_run_differentiable_validates_missing_source_fields_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    ocean = state.get_component_state("OCN").with_outgoing(RuntimeFieldStore.empty())
    state = state.set_component_state(ocean)

    with pytest.raises(CouplerError, match="source field"):
        coupler.run_differentiable(state)


def test_run_differentiable_validates_fractional_mask_shape_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    state = RuntimeCouplerState(
        components=state.components,
        fractional_masks=RuntimeFieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones((1, 1), dtype=jnp.float64),
                "ATM|OCN|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
                "ATM|LND|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
                "OCN|ICE|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
            }
        ),
        binary_masks=state.binary_masks,
    )

    with pytest.raises(CouplerError, match="fractional mask.*shape"):
        coupler.run_differentiable(state)
