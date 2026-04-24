from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
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
