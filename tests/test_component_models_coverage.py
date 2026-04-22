from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import vercor.components.data.era5_land as era5_land_module
import vercor.components.data.era5_ocean as era5_ocean_module
import vercor.components.data.erainterim_ocean as erainterim_ocean_module
from tests._coverage_support import CoverageCouplerStub, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.components.data.era5_land import ERA5Land
from vercor.components.data.era5_ocean import ERA5Ocean
from vercor.components.data.erainterim_ocean import ERAInterimOcean
from vercor.components.slab.atmosphere import Atmosphere
from vercor.components.slab.land import Land
from vercor.components.slab.ocean import Ocean
from vercor.components.slab.seaice import SeaIce


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
    atmosphere.initialize(coupler)
    atmosphere.step(dt, timestamp, coupler)
    assert_allclose_compact(
        atmosphere.data["sensible_heat_flux"],
        np.zeros(grid.shape),
    )
    assert_allclose_compact(
        atmosphere.data["latent_heat_flux"],
        np.zeros(grid.shape),
    )

    atmosphere.data["sea_surface_temperature"] = np.asarray(
        [[280.0, 281.0], [282.0, 283.0]]
    )
    atmosphere.step(dt, timestamp, coupler)
    ta0 = np.full(grid.shape, 288.15)
    dt_air = ta0 - atmosphere.data["sea_surface_temperature"]
    expected_shf = -10.0 * dt_air
    assert_allclose_compact(atmosphere.data["sensible_heat_flux"], expected_shf)
    assert_allclose_compact(atmosphere.data["latent_heat_flux"], -0.5 * expected_shf)
    assert np.isclose(
        atmosphere.data["u_velocity_10m"][0, 0], np.cos(np.deg2rad(-30.0))
    )
    assert np.isclose(atmosphere.data["v_velocity_10m"][0, 1], -0.5)

    ocean = Ocean(grid=grid)
    ocean.step(dt, timestamp, coupler)
    assert ocean.data == {}

    ocean.initialize(coupler)
    ocean.data["sensible_heat_flux"] = np.full(grid.shape, 20.0)
    ocean.data["latent_heat_flux"] = np.full(grid.shape, 10.0)
    starting_sst = ocean.data["sea_surface_temperature"].copy()
    ocean.step(dt, timestamp, coupler)
    tendency = 30.0 / (ocean.rho * ocean.cp * ocean.H)
    expected_sst = starting_sst + tendency * dt.total_seconds()
    assert_allclose_compact(ocean.data["sea_surface_temperature"], expected_sst)

    land = Land(grid=grid)
    land.initialize(coupler)
    land.data["latent_heat_flux"] = np.full(grid.shape, 100.0)
    land.step(timedelta(seconds=10.0), timestamp, coupler)
    expected_soil = np.full(grid.shape, 0.3 - 100.0 * 1e-9 * 10.0)
    assert_allclose_compact(land.data["soil_moisture"], expected_soil)

    seaice = SeaIce(grid=grid)
    seaice.step(dt, timestamp, coupler)
    assert seaice.data == {}

    seaice.initialize(coupler)
    seaice.data["sea_surface_temperature"] = np.asarray(
        [[270.0, 272.0], [274.0, 276.0]]
    )
    seaice.step(dt, timestamp, coupler)
    cold = seaice.data["ice_fraction"][0, 0]
    warm = seaice.data["ice_fraction"][1, 1]
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
    ) -> NDArray:
        assert where == "surface"
        assert not flip_y
        return forcing[variable]

    monkeypatch.setattr(era5_land_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(ERA5Land, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = ERA5Land()
    component.initialize(coupler)
    component.step(timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["surface"] == str(fake_path)
    assert component.settings.apply_time_interpolation
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
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
    ) -> NDArray:
        assert where == "surface"
        return forcing[variable]

    monkeypatch.setattr(era5_ocean_module, "get_forcing_data", fake_get_forcing_data)
    monkeypatch.setattr(ERA5Ocean, "_read_forcing", fake_read_forcing)

    coupler = cast(Any, CoverageCouplerStub())
    component = ERA5Ocean()
    component.initialize(coupler)
    component.step(timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["surface"] == str(fake_path)
    assert component.settings.apply_time_interpolation
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
    ) -> NDArray:
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
    component.initialize(coupler)
    component.step(timedelta(hours=1), datetime(2000, 1, 1), coupler)

    assert component.DATA_FILES["model_level"] == str(fake_path)
    assert component.settings.apply_time_interpolation
    assert component.grid.shape == (46, 2)
    binary_mask = component.grid.binary_mask
    assert binary_mask is not None
    assert np.all(binary_mask[3:-3, :] == 1.0)
    assert np.all(binary_mask[:3, :] == 0.0)
    assert np.isnan(component.data["sea_surface_temperature"][0, 0, 0])
    assert np.isclose(component.data["sea_surface_temperature"][0, 3, 0], 283.15)
