from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

import vercor.components.data.camulator_land as camulator_land_module
import vercor.components.external.camulator as camulator_module
from tests.assertions import assert_allclose_compact
from vercor.components.external.camulator import (
    CAMulatorGCM,
    _initialize_camulator_runtime_fields,
    _map_camulator_prediction_arrays,
    _prepare_camulator_surface_forcing,
)
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


def _make_coupler(start: datetime) -> Any:
    return SimpleNamespace(
        clock=SimpleNamespace(start=start, dt_seconds=21600),
        logger=_RecordingLogger(),
    )


def test_camulator_runtime_field_initializer_returns_jax_arrays() -> None:
    fields = _initialize_camulator_runtime_fields((2, 3))

    assert fields
    assert all(isinstance(value, jax.Array) for value in fields.values())
    assert fields["temperature"].shape == (2, 3)
    assert_allclose_compact(fields["temperature"], np.zeros((2, 3)))


def test_prepare_camulator_surface_forcing_supports_jit_and_gradients() -> None:
    sea_surface_temperature = jnp.asarray([[jnp.nan, 2.0], [5.0, 7.0]])
    land_surface_temperature = jnp.asarray([[10.0, jnp.nan], [15.0, 20.0]])
    land_mask_coslat = jnp.asarray([[0.5, 1.0], [1.2, 0.0]])

    total_surface_temperature, rescaled_total_surface_temperature = jax.jit(
        _prepare_camulator_surface_forcing
    )(
        sea_surface_temperature,
        land_surface_temperature,
        land_mask_coslat,
    )

    expected_total = np.asarray([[10.0, 283.0], [283.0, 27.0]])
    expected_rescaled = (expected_total - np.nanmean(expected_total)) / np.nanstd(
        expected_total
    )
    assert_allclose_compact(total_surface_temperature, expected_total)
    assert_allclose_compact(
        rescaled_total_surface_temperature,
        expected_rescaled,
        rtol=1e-7,
        atol=1e-7,
    )

    weights = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    gradient = jax.grad(
        lambda sst: jnp.sum(
            _prepare_camulator_surface_forcing(
                sst,
                jnp.asarray([[10.0, 11.0], [12.0, 13.0]]),
                jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            )[1]
            * weights
        )
    )(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert gradient.shape == sea_surface_temperature.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions() -> (
    None
):
    settings = VercorSettings()
    hyai = jnp.asarray([0.01, 0.02, 0.03])
    hybi = jnp.asarray([0.10, 0.20, 0.30])
    hyam = jnp.asarray([0.015, 0.025])
    hybm = jnp.asarray([0.15, 0.25])
    u_wind = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    v_wind = u_wind + 10.0
    surface_temperature = jnp.asarray([[280.0, 281.0], [282.0, 283.0]])
    temperature_3d = jnp.asarray(
        [
            [[270.0, 271.0], [272.0, 273.0]],
            [[280.0, 281.0], [282.0, 283.0]],
        ]
    )
    specific_humidity_3d = jnp.full((2, 2, 2), 0.002)
    net_shortwave_accumulated = jnp.full((2, 2), 21600.0 * 8.0)
    net_longwave_accumulated = jnp.full((2, 2), -21600.0 * 3.0)
    surface_pressure = jnp.full((2, 2), 100000.0)

    mapped_fields = jax.jit(_map_camulator_prediction_arrays)(
        settings.earth_radius,
        settings.gravity,
        settings.rdair,
        settings.zvir,
        settings.mwdair,
        settings.rgas,
        settings.p0,
        settings.cappa,
        settings.stefBoltz,
        100000.0,
        hyai,
        hybi,
        hyam,
        hybm,
        u_wind,
        v_wind,
        surface_temperature,
        temperature_3d,
        specific_humidity_3d,
        net_shortwave_accumulated,
        net_longwave_accumulated,
        surface_pressure,
    )

    assert_allclose_compact(mapped_fields["u_velocity"], np.asarray(u_wind[-1]))
    assert_allclose_compact(mapped_fields["v_velocity"], np.asarray(v_wind[-1]))
    assert_allclose_compact(
        mapped_fields["temperature"], np.asarray(temperature_3d[-1])
    )
    assert_allclose_compact(mapped_fields["specific_humidity"], np.full((2, 2), 0.002))
    assert_allclose_compact(
        mapped_fields["net_shortwave_radiation_flux"],
        np.full((2, 2), 8.0),
    )
    assert_allclose_compact(
        mapped_fields["downward_longwave_radiation_flux"],
        settings.stefBoltz * np.asarray(surface_temperature) ** 4 - 3.0,
    )
    assert mapped_fields["model_level_height"].shape == (2, 2)
    assert mapped_fields["density"].shape == (2, 2)
    assert mapped_fields["potential_temperature"].shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(mapped_fields["model_level_height"])))
    assert np.all(np.isfinite(np.asarray(mapped_fields["density"])))
    assert np.all(np.isfinite(np.asarray(mapped_fields["potential_temperature"])))


def test_camulator_constructor_builds_jax_backed_grid(monkeypatch: Any) -> None:
    latlons = SimpleNamespace(
        longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        latitude=SimpleNamespace(values=np.asarray([-45.0, 0.0, 45.0])),
    )
    monkeypatch.setattr(
        camulator_module,
        "initialize_camulator",
        lambda **kwargs: {
            "conf": {
                "data": {
                    "dynamic_forcing_variables": ["U"],
                    "lead_time_periods": 6,
                },
                "predict": {"timesteps_fast_climate": 1},
            },
            "stepper": SimpleNamespace(),
            "forcing_dataset": xr.Dataset(),
            "static_forcing": object(),
            "initial_state": object(),
            "latlons": latlons,
            "metadata": {},
            "device": "cpu",
            "state_transformer": object(),
        },
    )

    component = CAMulatorGCM(config_path="dummy.yaml", device="cpu")

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert_allclose_compact(component.grid.binary_mask, np.ones((3, 2)))


def test_camulator_land_stores_jax_runtime_arrays(
    monkeypatch: Any,
) -> None:
    start = datetime(2000, 1, 1, 0, 0, 0)
    forcing_ds = xr.Dataset(
        data_vars={
            "TS": (
                ("time", "lat", "lon"),
                np.asarray(
                    [
                        [[281.0, 282.0], [283.0, 284.0]],
                        [[285.0, 286.0], [287.0, 288.0]],
                    ]
                ),
            )
        },
        coords={"time": [start, datetime(2000, 1, 1, 6, 0, 0)]},
    )

    monkeypatch.setattr(
        camulator_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (jnp.ones((2, 2)), jnp.zeros((2, 2))),
    )
    monkeypatch.setattr(
        camulator_land_module,
        "initialize_camulator",
        lambda **kwargs: {
            "conf": {
                "data": {"lead_time_periods": 6},
                "predict": {"start_datetime": start},
            },
            "forcing_dataset_raw": forcing_ds,
        },
    )

    camulator_grid = RectilinearGrid(
        name="atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    ocean_grid = RectilinearGrid(
        name="ocn",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
        binary_mask=jnp.ones((2, 2)),
    )
    component = camulator_land_module.CAMulatorLand(
        config_path="dummy.yaml",
        camulator_grid=camulator_grid,
        ocn_grid=ocean_grid,
    )

    component.initialize(_make_coupler(start))
    assert isinstance(component.data["land_surface_temperature"], jax.Array)
    assert_allclose_compact(
        component.data["land_surface_temperature"], np.full((2, 2), 283.0)
    )

    component.step(
        dt=datetime(2000, 1, 1, 6, 0, 0) - start,
        time=start,
        coupler=_make_coupler(start),
    )
    assert isinstance(component.data["land_surface_temperature"], jax.Array)
    assert_allclose_compact(
        component.data["land_surface_temperature"],
        np.asarray([[281.0, 282.0], [283.0, 284.0]]),
    )
