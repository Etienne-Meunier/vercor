from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import pytest

from vercor.setups._time_helpers import (
    align_model_timestep,
    assign_model_timestep_alignment,
    runtime_forcing_index,
    run_logged_spinup,
    seed_grid_field_defaults,
)
import vercor.setups.external.camulator_forcing as camulator_forcing_module
from vercor.setups.external.camulator_forcing import initialize_camulator_forcing_cursor
from tests._coverage_support import make_test_grid
from vercor.components import DataComponent
from vercor.output import OutputConfig, PeriodOutput
from vercor.setup_config import JAXGCMConfig, Spinup
from vercor.settings import Settings


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _TimeIndex:
    def __init__(self, loc: int | slice) -> None:
        self.loc = loc

    def get_loc(self, value: object) -> int | slice:
        _ = value
        return self.loc


def test_align_model_timestep_returns_coupling_timestep_and_substeps() -> None:
    alignment = align_model_timestep(
        86400.0,
        timedelta(hours=6),
    )

    assert alignment.coupling_timestep == timedelta(days=1)
    assert alignment.model_substeps == 4


def test_align_model_timestep_rejects_non_divisible_model_step() -> None:
    with pytest.raises(
        ValueError,
        match=r"model_timestep .* must evenly divide coupling_timestep",
    ):
        align_model_timestep(
            3600.0,
            timedelta(minutes=45),
        )


def test_assign_model_timestep_alignment_sets_common_state_attributes() -> None:
    state = SimpleNamespace()

    alignment = assign_model_timestep_alignment(
        state,
        86400.0,
        timedelta(hours=6),
    )

    assert alignment.model_substeps == 4
    assert state.coupling_timestep == timedelta(days=1)
    assert state.model_timestep == timedelta(hours=6)
    assert state.model_substeps == 4


def test_runtime_forcing_index_uses_start_counter_and_model_substeps() -> None:
    assert runtime_forcing_index(start_ix=7, timestep_counter=3, model_substeps=4) == 19


def test_run_logged_spinup_logs_each_step_and_returns_callback_result() -> None:
    logger = _RecordingLogger()
    seen_steps: list[int] = []

    def step(step_number: int) -> int:
        seen_steps.append(step_number)
        return step_number * 10

    result = run_logged_spinup(
        steps=3,
        logger=logger,
        intro_message="Running spinup",
        step_message=lambda step, total: f"Step {step} / {total}",
        step=step,
    )

    assert result == 30
    assert seen_steps == [1, 2, 3]
    assert logger.infos == [
        "Running spinup",
        "Step 1 / 3",
        "Step 2 / 3",
        "Step 3 / 3",
    ]


def test_seed_grid_field_defaults_seeds_component_defaults_with_overrides() -> None:
    component = DataComponent.from_fields("ATM", make_test_grid())
    context = SimpleNamespace(settings=Settings())

    seed_grid_field_defaults(
        component,
        ("temperature", "humidity"),
        context,
        overrides={"temperature": 280.0},
    )

    assert set(component._data) == {"temperature", "humidity"}
    assert jnp.all(component._data["temperature"] == 280.0)
    assert jnp.all(component._data["humidity"] == 0.0)


def test_load_jcm_inputs_facade_returns_named_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vercor.setups as setups
    import vercor.setups.external.jax_gcm_tools as jax_gcm_tools

    coords = object()
    terrain = object()
    forcing = object()
    calls: dict[str, object] = {}

    def fake_load(
        *,
        resolution: int = 31,
        input_data_directory: Path | None = None,
    ) -> tuple[object, object, object]:
        calls["resolution"] = resolution
        calls["input_data_directory"] = input_data_directory
        return coords, terrain, forcing

    monkeypatch.setattr(
        jax_gcm_tools,
        "load_jcm_coords_terrain_forcing",
        fake_load,
    )

    inputs = setups.load_jcm_inputs(
        resolution=42,
        input_data_directory=tmp_path,
    )

    assert isinstance(inputs, setups.JCMInputs)
    assert inputs.coords is coords
    assert inputs.terrain is terrain
    assert inputs.forcing is forcing
    assert calls == {"resolution": 42, "input_data_directory": tmp_path}


def test_make_jcm_land_atmosphere_accepts_preloaded_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups.jcm_setup_helpers as helper

    coords = object()
    forcing = object()
    terrain = SimpleNamespace(fmask="original-mask")
    inputs = setups.JCMInputs(coords=coords, terrain=terrain, forcing=forcing)
    ocean_grid = make_test_grid(name="ocn-grid")
    land_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    land: Any = SimpleNamespace(grid=SimpleNamespace(binary_mask=land_mask))
    atmosphere: Any = object()
    calls: dict[str, Any] = {}

    def unexpected_load_jcm_inputs() -> object:
        pytest.fail("preloaded JCM inputs should avoid loading inputs again")

    def fake_make_jcm_land(
        received_coords: object,
        received_forcing: object,
        received_grid: object,
    ) -> Any:
        calls["land_args"] = (received_coords, received_forcing, received_grid)
        return land

    def fake_transposed_host_array(mask: object) -> str:
        calls["mask"] = mask
        return "patched-mask"

    def fake_make_jax_gcm(
        received_coords: object,
        received_terrain: object,
        **kwargs: object,
    ) -> object:
        calls["atmosphere_args"] = (received_coords, received_terrain, kwargs)
        return atmosphere

    def fake_load_jcm_factories() -> tuple[Any, Any]:
        return fake_make_jcm_land, fake_make_jax_gcm

    monkeypatch.setattr(
        helper,
        "load_jcm_inputs",
        unexpected_load_jcm_inputs,
    )
    monkeypatch.setattr(helper, "_load_jcm_factories", fake_load_jcm_factories)
    monkeypatch.setattr(helper, "transposed_host_array", fake_transposed_host_array)

    result = helper.make_jcm_land_atmosphere(
        ocean_grid,
        inputs=inputs,
        spinup=Spinup(enabled=False),
    )

    assert result.land is land
    assert result.atmosphere is atmosphere
    assert result.coords is coords
    assert result.terrain is terrain
    assert result.forcing is forcing
    assert terrain.fmask == "patched-mask"
    assert calls["mask"] is land_mask
    assert calls["land_args"] == (coords, forcing, ocean_grid)
    assert calls["atmosphere_args"] == (
        coords,
        terrain,
        {
            "config": JAXGCMConfig(
                forcing_data=forcing,
                spinup=Spinup(enabled=False),
                output=OutputConfig(period=PeriodOutput(frequency="month")),
                jitted=True,
            ),
        },
    )


def test_initialize_camulator_forcing_cursor_returns_index_and_warns_on_mismatch() -> (
    None
):
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1, 6)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(slice(7, 9))})

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
    )

    assert cursor.start_ix == 7
    assert cursor.init_str == "2000-01-01T06Z"
    assert cursor.init_datetime == forcing_start
    assert logger.infos == ["Starting integration at time index: 7"]
    assert len(logger.warnings) == 1
    assert "does not match" in logger.warnings[0]


def test_initialize_camulator_forcing_cursor_accepts_integer_index() -> None:
    logger = _RecordingLogger()
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(3)})

    cursor = initialize_camulator_forcing_cursor(
        conf={"predict": {"start_datetime": "2000-01-01 00:00:00"}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=datetime(2000, 1, 1),
        logger=logger,
    )

    assert cursor.start_ix == 3
    assert cursor.init_str == "2000-01-01T00Z"
    assert logger.warnings == []


def test_camulator_runtime_cursor_initializes_indexes_and_advances() -> None:
    logger = _RecordingLogger()
    forcing_start = datetime(2000, 1, 1)
    dynamic_ds = SimpleNamespace(indexes={"time": _TimeIndex(4)})
    cursor = camulator_forcing_module.CamulatorRuntimeCursor()

    result = cast(Any, cursor.initialize)(
        conf={"predict": {"start_datetime": forcing_start}},
        dynamic_ds=dynamic_ds,
        coupler_start_datetime=forcing_start,
        model_substeps=3,
        logger=logger,
    )

    assert result is None
    assert cursor.start_ix == 4
    assert cursor.init_datetime == forcing_start
    assert cursor.init_str == "2000-01-01T00Z"
    assert cursor.timestep_counter == 0
    assert cursor.current_index() == 4

    cursor.advance()
    assert cursor.timestep_counter == 1
    assert cursor.current_index() == 7


def test_make_jcm_land_atmosphere_patches_mask_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups.jcm_setup_helpers as helper

    coords = object()
    forcing = object()
    terrain = SimpleNamespace(fmask="original-mask")
    ocean_grid = make_test_grid(name="ocn-grid")
    land_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    land: Any = SimpleNamespace(grid=SimpleNamespace(binary_mask=land_mask))
    atmosphere: Any = object()
    calls: dict[str, Any] = {}

    def fake_load_jcm_inputs() -> Any:
        calls["load_inputs"] = True
        return helper.JCMInputs(coords=coords, terrain=terrain, forcing=forcing)

    def fake_make_jcm_land(
        received_coords: object,
        received_forcing: object,
        received_grid: object,
    ) -> Any:
        calls["land_args"] = (received_coords, received_forcing, received_grid)
        return land

    def fake_transposed_host_array(mask: object) -> str:
        calls["mask"] = mask
        return "patched-mask"

    def fake_make_jax_gcm(
        received_coords: object,
        received_terrain: object,
        **kwargs: object,
    ) -> object:
        calls["atmosphere_args"] = (received_coords, received_terrain, kwargs)
        return atmosphere

    def fake_load_jcm_factories() -> tuple[Any, Any]:
        return fake_make_jcm_land, fake_make_jax_gcm

    monkeypatch.setattr(helper, "load_jcm_inputs", fake_load_jcm_inputs)
    monkeypatch.setattr(helper, "_load_jcm_factories", fake_load_jcm_factories)
    monkeypatch.setattr(helper, "transposed_host_array", fake_transposed_host_array)

    result = helper.make_jcm_land_atmosphere(
        ocean_grid,
        custom_parameters={"surface_flux.vgust": 5.01},
        spinup=Spinup(enabled=False),
        jitted=False,
        output=OutputConfig(period=PeriodOutput(frequency="year")),
    )

    assert result.land is land
    assert result.atmosphere is atmosphere
    assert result.coords is coords
    assert result.terrain is terrain
    assert result.forcing is forcing
    assert terrain.fmask == "patched-mask"
    assert calls["load_inputs"] is True
    assert calls["mask"] is land_mask
    assert calls["land_args"] == (coords, forcing, ocean_grid)
    assert calls["atmosphere_args"] == (
        coords,
        terrain,
        {
            "config": JAXGCMConfig(
                custom_parameters={"surface_flux.vgust": 5.01},
                forcing_data=forcing,
                spinup=Spinup(enabled=False),
                output=OutputConfig(period=PeriodOutput(frequency="year")),
                jitted=False,
            ),
        },
    )


def test_build_jcm_land_atmosphere_components_is_removed() -> None:
    import vercor.setups.jcm_setup_helpers as helper

    assert not hasattr(helper, "build_jcm_land_atmosphere_components")
