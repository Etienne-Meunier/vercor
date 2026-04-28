from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import numpy as np
from numpy.typing import NDArray
import pytest

import vercor.components.data.era5_atmosphere as era5_atmosphere_module
import vercor.components.data.era5_land as era5_land_module
import vercor.components.data.era5_ocean as era5_ocean_module
import vercor.components.data.erainterim_ocean as erainterim_ocean_module
import vercor.components.data.jcm_land as jcm_land_module
from tests._coverage_support import CoverageCouplerStub, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.components.base import RuntimeStepContext
from vercor.components.data.era5_atmosphere import ERA5Atmosphere
from vercor.components.data.era5_land import ERA5Land
from vercor.components.data.era5_ocean import ERA5Ocean
from vercor.components.data.erainterim_ocean import ERAInterimOcean
from vercor.components.data.jcm_land import JCMLand
from vercor.components.slab.atmosphere import Atmosphere
from vercor.components.slab.land import Land
from vercor.components.slab.ocean import Ocean
from vercor.components.slab.seaice import SeaIce
from vercor.runtime_components import create_runtime_component_state
from vercor.types import RuntimeArray


def _step_component(
    component: Any,
    dt: timedelta,
    time: datetime,
    coupler: Any,
) -> Any:
    """Advance one component through the runtime-state API."""

    return component.step_runtime_state(
        create_runtime_component_state(component, prefill_missing=True),
        RuntimeStepContext(
            dt_seconds=dt.total_seconds(),
            settings=coupler.settings,
            time=time,
            logger=coupler.logger,
        ),
    )


@pytest.mark.fast_always
def test_slab_component_initialize_and_step_behaviors() -> None:
    coupler = cast(Any, CoverageCouplerStub())
    timestamp = datetime(2000, 1, 1, 0, 0, 0)
    dt = timedelta(hours=1)
    grid = make_test_grid(
        name="toy",
        longitude=np.asarray([0.0, 90.0]),
        latitude=np.asarray([-30.0, 30.0]),
    )

    atmosphere = Atmosphere(grid=grid)
    atmosphere.initialize(coupler.init_context())
    atmosphere_state = _step_component(atmosphere, dt, timestamp, coupler)
    assert_allclose_compact(
        atmosphere_state.data.get("sensible_heat_flux"),
        np.zeros(grid.shape),
    )
    assert_allclose_compact(
        atmosphere_state.data.get("latent_heat_flux"),
        np.zeros(grid.shape),
    )

    atmosphere.data["sea_surface_temperature"] = np.asarray(
        [[280.0, 281.0], [282.0, 283.0]]
    )
    initial_temperature_2m = np.asarray(atmosphere.data["temperature_2m"]).copy()
    atmosphere_state = _step_component(atmosphere, dt, timestamp, coupler)
    atmosphere_data = atmosphere_state.data
    assert atmosphere_data.get("sensible_heat_flux").shape == grid.shape
    assert atmosphere_data.get("latent_heat_flux").shape == grid.shape
    assert atmosphere_data.get("u_velocity_10m").shape == grid.shape
    assert atmosphere_data.get("v_velocity_10m").shape == grid.shape
    assert np.all(np.asarray(atmosphere_data.get("sensible_heat_flux")) < 0.0)
    assert np.all(np.asarray(atmosphere_data.get("latent_heat_flux")) > 0.0)
    assert np.any(np.asarray(atmosphere_data.get("u_velocity_10m")) != 0.0)
    assert np.any(np.asarray(atmosphere_data.get("v_velocity_10m")) != 0.0)
    assert np.all(
        np.asarray(atmosphere_data.get("temperature_2m")) < initial_temperature_2m
    )

    ocean = Ocean(grid=grid)
    ocean_state = _step_component(ocean, dt, timestamp, coupler)
    assert ocean.data == {}
    assert ocean_state.data.field_names == ()

    ocean.initialize(coupler.init_context())
    ocean.data["sensible_heat_flux"] = np.full(grid.shape, 20.0)
    ocean.data["latent_heat_flux"] = np.full(grid.shape, 10.0)
    starting_sst = ocean.data["sea_surface_temperature"].copy()
    ocean_state = _step_component(ocean, dt, timestamp, coupler)
    ocean_sst = ocean_state.data.get("sea_surface_temperature")
    assert ocean_sst.shape == grid.shape
    assert np.all(np.asarray(ocean_sst) > starting_sst)

    land = Land(grid=grid)
    land.initialize(coupler.init_context())
    land.data["latent_heat_flux"] = np.full(grid.shape, 100.0)
    land_state = _step_component(land, timedelta(seconds=10.0), timestamp, coupler)
    soil_moisture = land_state.data.get("soil_moisture")
    assert soil_moisture.shape == grid.shape
    assert np.all(np.asarray(soil_moisture) < 0.3)

    seaice = SeaIce(grid=grid)
    seaice_state = _step_component(seaice, dt, timestamp, coupler)
    assert seaice.data == {}
    assert seaice_state.data.field_names == ()

    seaice.initialize(coupler.init_context())
    seaice.data["sea_surface_temperature"] = np.asarray(
        [[270.0, 272.0], [274.0, 276.0]]
    )
    seaice_state = _step_component(seaice, dt, timestamp, coupler)
    ice_fraction = seaice_state.data.get("ice_fraction")
    cold = ice_fraction[0, 0]
    warm = ice_fraction[1, 1]
    assert cold > warm
    assert 0.0 < warm < 1.0


def test_era5_land_constructor_uses_masked_grid_and_enables_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/era5_land_masked.nc")
    forcing: dict[str, NDArray] = {
        "lon": np.asarray([0.0, 120.0, 240.0]),
        "lat": np.asarray([-30.0, 30.0]),
        "mask": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "skt": np.asarray([[[280.0]], [[281.0]], [[282.0]]]),
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "era5_land_masked"
        return fake_path

    def fake_read_forcing(
        self: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "surface"
        assert not flip_y
        return forcing[variable]

    monkeypatch.setattr(era5_land_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(ERA5Land, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = ERA5Land()
    component.initialize(coupler.init_context())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["surface"] == str(fake_path)
    assert component.settings.apply_time_interpolation
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.data["land_surface_temperature"], jax.Array)
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert isinstance(binary_mask, jax.Array)
    assert_allclose_compact(binary_mask, forcing["mask"].T)
    assert_allclose_compact(component.data["land_surface_temperature"], forcing["skt"])


def test_era5_ocean_constructor_applies_land_mask_and_reverses_latitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/era5_surface.nc")
    forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0]),
        "latitude": np.asarray([10.0, -10.0]),
        "lsm": np.asarray(
            [
                [[1.0], [0.4]],
                [[0.0], [1.0]],
            ]
        ),
        "sst": np.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "era5_surface"
        return fake_path

    def fake_read_forcing(
        self: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "surface"
        return forcing[variable]

    monkeypatch.setattr(era5_ocean_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(ERA5Ocean, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = ERA5Ocean()
    component.initialize(coupler.init_context())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["surface"] == str(fake_path)
    assert component.settings.apply_time_interpolation
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.data["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(component.grid.latitude, np.asarray([-10.0, 10.0]))
    expected_mask = np.asarray([[0.0, 1.0], [0.0, 0.0]])
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert_allclose_compact(binary_mask, expected_mask)
    assert np.isnan(component.data["sea_surface_temperature"][0, 0, 0])
    assert np.isclose(component.data["sea_surface_temperature"][1, 0, 0], 284.0)


def test_erainterim_ocean_constructor_builds_global_masked_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path = Path("/tmp/erainterim_4deg.nc")
    yt = np.arange(-78.0, 82.0, 4.0)
    sss = np.ones((2, yt.size, 12), dtype=float)
    sst = 10.0 * np.ones((2, yt.size, 12), dtype=float)
    forcing: dict[str, NDArray] = {
        "xt": np.asarray([0.0, 4.0]),
        "yt": yt,
        "sss": sss,
        "sst": sst,
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        assert file_type == "erainterim_ocean_4deg"
        return fake_path

    def fake_read_forcing(
        self: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        assert where == "model_level"
        assert not flip_y
        return forcing[variable]

    monkeypatch.setattr(
        erainterim_ocean_module,
        "get_forcing_data",
        fake_get_forcing_data,
    )
    monkeypatch.setattr(ERAInterimOcean, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = ERAInterimOcean(resolution="4deg")
    component.initialize(coupler.init_context())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["model_level"] == str(fake_path)
    assert component.settings.apply_time_interpolation
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.data["sea_surface_temperature"], jax.Array)
    assert component.grid.shape == (46, 2)
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert np.all(binary_mask[3:-3, :] == 1.0)
    assert np.all(binary_mask[:3, :] == 0.0)
    assert np.isnan(component.data["sea_surface_temperature"][0, 0, 0])
    assert np.isclose(component.data["sea_surface_temperature"][0, 3, 0], 283.15)


def test_era5_atmosphere_constructor_initialize_and_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_level_path = Path("/tmp/era5_model_levels.nc")
    surface_path = Path("/tmp/era5_surface.nc")

    forcing: dict[str, NDArray] = {
        "longitude": np.asarray([0.0, 180.0], dtype=float),
        "latitude": np.asarray([45.0, 0.0, -45.0], dtype=float),
        "hyai": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
        "hybi": np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float),
        "hyam": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=float),
        "hybm": np.asarray([30.0, 31.0, 32.0, 33.0], dtype=float),
        "lnsp": np.log(
            np.arange(1, 1 + (2 * 3 * 1 * 12), dtype=float).reshape(2, 3, 1, 12)
        ),
        "q": np.arange(1, 1 + (2 * 3 * 3 * 12), dtype=float).reshape(2, 3, 3, 12)
        / 1000.0,
        "t": 250.0 + np.arange(2 * 3 * 3 * 12, dtype=float).reshape(2, 3, 3, 12),
        "u": np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "v": -np.arange(1, 1 + (2 * 3 * 2 * 12), dtype=float).reshape(2, 3, 2, 12),
        "msnswrf": np.full((2, 3, 12), 150.0, dtype=float),
        "msdwlwrf": np.full((2, 3, 12), 75.0, dtype=float),
    }

    physics_calls: dict[str, list[Any]] = {
        "pressure": [],
        "height": [],
        "density": [],
        "theta": [],
    }

    def fake_get_forcing_data(file_type: str) -> Path:
        if file_type == "era5_model_levels":
            return model_level_path
        if file_type == "era5_surface":
            return surface_path
        raise AssertionError(f"Unexpected forcing lookup: {file_type}")

    def fake_read_forcing(
        self: Any,
        variable: str,
        where: str,
        flip_y: bool = False,
    ) -> RuntimeArray:
        if where == "model_level":
            assert variable in {
                "longitude",
                "latitude",
                "hyai",
                "hybi",
                "hyam",
                "hybm",
                "lnsp",
                "q",
                "t",
                "u",
                "v",
            }
        else:
            assert where == "surface"
            assert variable in {"msnswrf", "msdwlwrf"}
        _ = flip_y
        return forcing[variable]

    def fake_compute_pressure_levels(
        surface_pressure: NDArray,
        hya: NDArray,
        hyb: NDArray,
    ) -> NDArray:
        physics_calls["pressure"].append(
            (surface_pressure.copy(), hya.copy(), hyb.copy())
        )
        base = float(surface_pressure.mean())
        nlev = hya.size
        return np.stack(
            [
                np.full(surface_pressure.shape, base + float(level), dtype=float)
                for level in range(nlev)
            ],
            axis=2,
        )

    def fake_get_altitudes_hybrid_sigma_levels(
        settings: Any,
        temperature_3d: NDArray,
        specific_humidity_3d: NDArray,
        pressure_h: NDArray,
    ) -> NDArray:
        physics_calls["height"].append(
            (
                settings,
                temperature_3d.copy(),
                specific_humidity_3d.copy(),
                pressure_h.copy(),
            )
        )
        height = np.zeros((2, 3, 2), dtype=float)
        height[..., 1] = float(pressure_h.mean())
        return height

    def fake_compute_air_density(
        settings: Any,
        pressure_level: NDArray,
        temperature: NDArray,
    ) -> NDArray:
        physics_calls["density"].append(
            (settings, pressure_level.copy(), temperature.copy())
        )
        return np.asarray(pressure_level + temperature, dtype=float)

    def fake_compute_potential_temperature(
        settings: Any,
        temperature: NDArray,
        pressure_level: NDArray,
    ) -> NDArray:
        physics_calls["theta"].append(
            (settings, temperature.copy(), pressure_level.copy())
        )
        return np.asarray(pressure_level - temperature, dtype=float)

    monkeypatch.setattr(
        era5_atmosphere_module,
        "get_forcing_data",
        fake_get_forcing_data,
    )
    monkeypatch.setattr(ERA5Atmosphere, "_read_forcing", fake_read_forcing)
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_pressure_levels",
        fake_compute_pressure_levels,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "get_altitudes_hybrid_sigma_levels",
        fake_get_altitudes_hybrid_sigma_levels,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_air_density",
        fake_compute_air_density,
    )
    monkeypatch.setattr(
        era5_atmosphere_module,
        "compute_potential_temperature",
        fake_compute_potential_temperature,
    )

    coupler = cast(Any, CoverageCouplerStub())
    component = ERA5Atmosphere()

    assert component.DATA_FILES == {
        "model_level": str(model_level_path),
        "surface": str(surface_path),
    }
    assert component.settings.apply_time_interpolation
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.data["surface_pressure"], jax.Array)
    assert_allclose_compact(component.grid.longitude, forcing["longitude"])
    assert_allclose_compact(component.grid.latitude, np.asarray([-45.0, 0.0, 45.0]))
    assert_allclose_compact(component.data["hyai"], np.asarray([2.0, 3.0, 4.0]))
    assert_allclose_compact(component.data["hybi"], np.asarray([12.0, 13.0, 14.0]))
    assert_allclose_compact(component.data["hyam"], np.asarray([22.0, 23.0]))
    assert_allclose_compact(component.data["hybm"], np.asarray([32.0, 33.0]))
    assert component.data["surface_pressure"].shape == (2, 3, 12)
    assert component.data["specific_humidity_3d"].shape == (2, 3, 2, 12)
    assert component.data["temperature_3d"].shape == (2, 3, 2, 12)
    assert component.data["u_velocity"].shape == (2, 3, 12)
    assert component.data["v_velocity"].shape == (2, 3, 12)

    component.initialize(coupler.init_context())

    assert len(physics_calls["pressure"]) == 24
    assert len(physics_calls["height"]) == 12
    assert len(physics_calls["density"]) == 12
    assert len(physics_calls["theta"]) == 12
    assert component.data["model_level_height"].shape == (2, 3, 12)
    assert component.data["density"].shape == (2, 3, 12)
    assert component.data["potential_temperature"].shape == (2, 3, 12)
    assert np.all(component.data["model_level_height"] > 0.0)

    component.data["land_surface_temperature"] = np.asarray(
        [[np.nan, 270.0, 271.0], [272.0, np.nan, 273.0]]
    )
    component.data["sea_surface_temperature"] = np.asarray(
        [[274.0, np.nan, 275.0], [276.0, 277.0, np.nan]]
    )

    component_state = _step_component(
        component, timedelta(hours=1), datetime(2000, 1, 1), coupler
    )

    expected_total = np.asarray(
        [[274.0, 270.0, 546.0], [548.0, 277.0, 273.0]],
        dtype=float,
    )
    assert_allclose_compact(
        component_state.data.get("total_surface_temperature"), expected_total
    )


def test_jcm_land_constructor_converts_coords_and_preserves_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    recorded_inputs: dict[str, NDArray] = {}

    def fake_create_lnd_mask_from_ocn(
        atm_lat: NDArray,
        atm_lon: NDArray,
        ocn_grid: Any,
    ) -> tuple[NDArray, NDArray]:
        recorded_inputs["atm_lat"] = np.asarray(atm_lat)
        recorded_inputs["atm_lon"] = np.asarray(atm_lon)
        recorded_inputs["ocn_lon"] = np.asarray(ocn_grid.longitude)
        return expected_mask, np.zeros_like(expected_mask)

    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        fake_create_lnd_mask_from_ocn,
    )

    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=np.deg2rad(np.asarray([0.0, 180.0], dtype=float)),
            latitudes=np.deg2rad(np.asarray([-45.0, 45.0], dtype=float)),
        )
    )
    forcing = SimpleNamespace(
        stl_am=np.asarray([[280.0, 281.0], [282.0, 283.0]], dtype=float),
        soilw_am=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
    )
    ocn_grid = make_test_grid(name="ocn")

    component = JCMLand(
        jcm_coords=cast(Any, coords),
        jcm_forcing=cast(Any, forcing),
        ocn_grid=ocn_grid,
    )
    coupler = cast(Any, CoverageCouplerStub())
    component.initialize(coupler.init_context())
    _step_component(component, timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.settings.get_field_time_slice
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.data["land_surface_temperature"], jax.Array)
    assert isinstance(component.data["soil_moisture"], jax.Array)
    assert_allclose_compact(recorded_inputs["atm_lon"], np.asarray([0.0, 180.0]))
    assert_allclose_compact(recorded_inputs["atm_lat"], np.asarray([-45.0, 45.0]))
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert_allclose_compact(binary_mask, expected_mask)
    assert_allclose_compact(
        component.data["land_surface_temperature"],
        forcing.stl_am.T,
    )
    assert_allclose_compact(component.data["soil_moisture"], forcing.soilw_am.T)
