from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.components.external.jax_gcm as jax_gcm_module
import vercor.components.external.veros_gcm as veros_gcm_module
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.settings import VercorSettings


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


@dataclass
class _PredictionDataset:
    dataset: xr.Dataset

    def to_xarray(self) -> xr.Dataset:
        return self.dataset


class _FakeForcing:
    def __init__(self) -> None:
        self.copy_calls: list[dict[str, Any]] = []

    def copy(self, **kwargs: Any) -> "_FakeForcing":
        self.copy_calls.append(kwargs)
        return self


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        _ = exc_type, exc, tb
        return False


class _FakeSettings(dict[str, Any]):
    def __init__(self, metadata: dict[str, Any], values: dict[str, Any]) -> None:
        super().__init__(values)
        object.__setattr__(self, "__metadata__", metadata)

    def unlock(self) -> _NullContext:
        return _NullContext()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "__metadata__":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeVariableStore(SimpleNamespace):
    def unlock(self) -> _NullContext:
        return _NullContext()


class _ConstructedVerosState:
    def __init__(
        self,
        var_meta: dict[str, Any],
        settings_meta: dict[str, Any],
        dimensions: dict[str, Any],
        plugin_interfaces: dict[str, Any] | None = None,
    ) -> None:
        self._var_meta = var_meta
        self._dimensions = dimensions
        self._plugin_interfaces = plugin_interfaces
        self.settings = _FakeSettings(settings_meta, {})
        self._variables: dict[str, Any] = {}
        self.timers: dict[str, Any] = {}
        self.profile_timers: dict[str, Any] = {}


def _make_coupler(
    *,
    dt_seconds: float,
    run_order: list[str],
    settings: VercorSettings | None = None,
) -> Any:
    return SimpleNamespace(
        clock=SimpleNamespace(dt_seconds=dt_seconds),
        logger=_RecordingLogger(),
        settings=settings or VercorSettings(),
        run_sequence=SimpleNamespace(order=run_order),
    )


def _make_fake_veros_state(surface_temperature: float = 10.0) -> Any:
    temp = np.full((8, 8, 1, 1), surface_temperature, dtype=float)
    variables = SimpleNamespace(temp=temp, tau=0)
    return SimpleNamespace(variables=variables)


def _make_flux_ready_veros_state() -> Any:
    tau = 0
    return SimpleNamespace(
        variables=SimpleNamespace(
            tau=tau,
            u=np.arange(36.0).reshape(6, 6, 1, 1),
            v=np.arange(36.0, 72.0).reshape(6, 6, 1, 1),
            temp=np.full((6, 6, 1, 1), 7.0, dtype=float),
            maskT=np.ones((6, 6, 1), dtype=float),
        )
    )


def test_asfloat_converts_tree_leaves_to_float_dtype() -> None:
    tree = {
        "a": jnp.asarray([1, 2], dtype=jnp.int32),
        "b": jnp.asarray([[3, 4]], dtype=jnp.int32),
    }

    converted = jax_gcm_module.asfloat(tree)

    assert jnp.issubdtype(converted["a"].dtype, jnp.floating)
    assert jnp.issubdtype(converted["b"].dtype, jnp.floating)
    assert_allclose_compact(converted["a"], np.asarray([1.0, 2.0]))


def test_cleanup_surface_temperature_fields_supports_jit_and_gradients() -> None:
    land_surface_temperature = jnp.asarray([[270.0, jnp.nan], [260.0, 240.0]])
    sea_surface_temperature = jnp.asarray([[jnp.nan, 281.0], [2.0, 3.0]])

    (
        clean_land_surface_temperature,
        clean_sea_surface_temperature,
        total_surface_temperature,
        cold_surface_cells,
    ) = jax.jit(jax_gcm_module._cleanup_surface_temperature_fields)(
        land_surface_temperature,
        sea_surface_temperature,
    )

    assert_allclose_compact(
        clean_land_surface_temperature,
        np.asarray([[270.0, 0.0], [260.0, 240.0]]),
    )
    assert_allclose_compact(
        clean_sea_surface_temperature,
        np.asarray([[0.0, 281.0], [2.0, 3.0]]),
    )
    assert_allclose_compact(
        total_surface_temperature,
        np.asarray([[270.0, 281.0], [262.0, 243.0]]),
    )
    assert_allclose_compact(
        cold_surface_cells,
        np.asarray([[False, False], [False, True]]),
    )

    gradient = jax.grad(
        lambda land: jnp.sum(
            jax_gcm_module._cleanup_surface_temperature_fields(
                land,
                jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            )[2]
        )
    )(jnp.asarray([[270.0, 271.0], [272.0, 273.0]]))
    assert_allclose_compact(gradient, np.ones((2, 2)))


def test_prepare_surface_temperature_forcing_supports_jit_and_fill_value() -> None:
    total_surface_temperature = jnp.asarray([[270.0, 281.0], [282.0, 567.0]])
    land_fraction_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])

    land_surface_temperature, sea_surface_temperature = jax.jit(
        jax_gcm_module._prepare_surface_temperature_forcing
    )(total_surface_temperature, land_fraction_mask)

    assert_allclose_compact(
        land_surface_temperature,
        np.asarray([[270.0, 288.15], [288.15, 567.0]]),
    )
    assert_allclose_compact(
        sea_surface_temperature,
        np.asarray([[288.15, 281.0], [282.0, 288.15]]),
    )


def test_map_jcm_output_fields_supports_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        jax_gcm_module,
        "compute_pressure_levels",
        lambda reference_pressure, top_pressure, sigma_levels, normalized_surface_pressure: jnp.asarray(
            [
                jnp.full((2, 2), 90000.0),
                jnp.full((2, 2), 80000.0),
            ]
        ),
    )
    monkeypatch.setattr(
        jax_gcm_module,
        "get_altitudes_sigma_levels",
        lambda temperature, pressure, specific_humidity: jnp.asarray(
            [
                jnp.full((2, 2), 50.0),
                jnp.full((2, 2), 150.0),
            ]
        ),
    )

    mapped_fields = jax.jit(jax_gcm_module._map_jcm_output_fields)(
        2.5e6,
        1.0e5,
        jnp.asarray([0.2, 1.0]),
        28.966,
        8314.47,
        1.0e5,
        0.286,
        jnp.full((2, 2, 2), 5.0),
        jnp.full((2, 2, 2), 2.0),
        jnp.full((2, 2), 40.0),
        jnp.full((2, 2), 30.0),
        jnp.asarray([[0.9, 1.0], [1.1, 1.2]]),
        jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        jnp.asarray(
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ]
        ),
        jnp.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
        jnp.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
    )

    assert_allclose_compact(
        mapped_fields["u_velocity"], np.asarray([[5.0, 7.0], [6.0, 8.0]])
    )
    assert_allclose_compact(
        mapped_fields["v_velocity"], np.asarray([[6.0, 8.0], [7.0, 9.0]])
    )
    assert_allclose_compact(
        mapped_fields["temperature"], np.asarray([[284.0, 286.0], [285.0, 287.0]])
    )
    assert_allclose_compact(
        mapped_fields["specific_humidity"],
        np.asarray([[0.05, 0.07], [0.06, 0.08]]),
    )
    assert_allclose_compact(mapped_fields["sensible_heat_flux"], np.full((2, 2), -10.0))
    assert_allclose_compact(
        mapped_fields["latent_heat_flux"],
        np.full((2, 2), -10000.0),
    )
    assert_allclose_compact(
        mapped_fields["pressure"],
        np.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    assert_allclose_compact(
        mapped_fields["density"],
        28.966
        / 8314.47
        * np.full((2, 2), 80000.0)
        / np.asarray([[284.0, 286.0], [285.0, 287.0]]),
    )
    assert_allclose_compact(
        mapped_fields["potential_temperature"],
        np.asarray([[284.0, 286.0], [285.0, 287.0]]) * (100000.0 / 80000.0) ** 0.286,
    )
    assert_allclose_compact(mapped_fields["model_level_height"], np.full((2, 2), 150.0))


def test_generate_step_function_non_jitted_averages_predictions() -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    component.save_interval = timedelta(days=2)
    component.coupling_timestep = timedelta(hours=12)

    calls: dict[str, Any] = {}

    class _FakeModel:
        def run_from_state(
            self,
            *,
            initial_state: Any,
            save_interval: float,
            total_time: float,
            forcing: Any,
        ) -> tuple[str, Any]:
            calls["run_from_state"] = (
                initial_state,
                save_interval,
                total_time,
                forcing,
            )
            predictions = SimpleNamespace(
                dynamics={"wind": jnp.asarray([[1.0, 3.0], [5.0, 7.0]])},
                physics={"temp": jnp.asarray([[2.0, 4.0], [6.0, 8.0]])},
            )
            return "next-modal-state", predictions

    component.model = _FakeModel()
    state = jax_gcm_module.JCMState(prog={}, phydata={}, metadata="initial-state")

    step_function = component._generate_step_function(jitted=False)
    next_state, predictions = step_function(state, "forcing")

    assert calls["run_from_state"] == ("initial-state", 2.0, 0.5, "forcing")
    assert next_state.metadata == "next-modal-state"
    assert_allclose_compact(next_state.prog["wind"], np.asarray([3.0, 5.0]))
    assert_allclose_compact(next_state.phydata["temp"], np.asarray([4.0, 6.0]))
    assert predictions.physics["temp"].shape == (2, 2)


def test_do_jcm_steps_updates_state_and_appends_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    prediction = SimpleNamespace(name="prediction")
    initial_state = jax_gcm_module.JCMState(
        prog={"u": jnp.asarray([0.0])},
        phydata={"t": jnp.asarray([0.0])},
        metadata="old-state",
    )
    updated_state = jax_gcm_module.JCMState(
        prog={"u": jnp.asarray([1.0])},
        phydata={"t": jnp.asarray([1.0])},
        metadata="new-state",
    )
    component._state = initial_state
    component.forcing = "forcing"
    component._predictions_list = []
    component._step_function = lambda state, forcing: (updated_state, prediction)

    averaged = SimpleNamespace(physics="physics-mean", dynamics="dynamics-mean")
    monkeypatch.setattr(jax_gcm_module, "stack_objects", lambda objs: objs)
    monkeypatch.setattr(jax_gcm_module, "unwrap_leading_dims", lambda obj: obj)
    monkeypatch.setattr(jax_gcm_module, "mean_leaf", lambda tree, axis: averaged)

    physics, dynamics = component.do_jcm_steps()

    assert component._state is updated_state
    assert component._predictions_list == [prediction]
    assert physics == "physics-mean"
    assert dynamics == "dynamics-mean"


def test_jax_gcm_initialize_validates_timestep_multiple() -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    component.spinup_time = timedelta(hours=6)
    component.model_timestep = timedelta(minutes=45)
    component.model = SimpleNamespace()
    component.grid = make_test_grid()

    with pytest.raises(ValueError, match="model_timestep"):
        component.initialize(_make_coupler(dt_seconds=3600.0, run_order=["ATM"]))


def test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    component.spinup_time = timedelta(hours=2)
    component.model_timestep = timedelta(hours=1)
    component.jitted = False
    component.do_spinup = True
    component.grid = make_test_grid()
    component.data = {}
    component.save_interval = timedelta(days=1)
    component.output_frequency = None
    component.forcing_data = "provided-forcing"
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
            vertical=SimpleNamespace(layers=2),
        ),
        primitive="primitive",
        _prepare_initial_modal_state=lambda: "modal-state",
    )

    physics_calls: dict[str, Any] = {}
    spinup_calls = {"count": 0}

    class _FakePhysicsData:
        @staticmethod
        def zeros(shape: tuple[int, int], layers: int) -> dict[str, Any]:
            physics_calls["zeros"] = (shape, layers)
            return {"shape": shape, "layers": layers}

    monkeypatch.setattr(jax_gcm_module, "PhysicsData", _FakePhysicsData)
    monkeypatch.setattr(
        jax_gcm_module,
        "dynamics_state_to_physics_state",
        lambda modal_state, primitive: {
            "modal_state": modal_state,
            "primitive": primitive,
        },
    )
    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: (lambda state, forcing: (state, "unused")),
    )

    def fake_do_jcm_steps() -> tuple[None, None]:
        spinup_calls["count"] += 1
        return (None, None)

    monkeypatch.setattr(component, "do_jcm_steps", fake_do_jcm_steps)

    coupler = _make_coupler(dt_seconds=3600.0, run_order=["OCN"])
    component.initialize(coupler)

    assert component.coupling_timestep == timedelta(hours=1)
    assert component.spinup_steps == 2
    assert physics_calls["zeros"] == ((2, 3), 2)
    assert component.forcing == "provided-forcing"
    assert component._predictions_list == []
    assert spinup_calls["count"] == 2
    assert component.data["sea_surface_temperature"].shape == component.grid.shape


def test_jax_gcm_initialize_builds_default_forcing_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    component.spinup_time = timedelta(hours=1)
    component.model_timestep = timedelta(hours=1)
    component.jitted = False
    component.do_spinup = False
    component.grid = make_test_grid()
    component.data = {}
    component.save_interval = timedelta(days=1)
    component.output_frequency = None
    component.forcing_data = None
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
            vertical=SimpleNamespace(layers=2),
        ),
        primitive="primitive",
        _prepare_initial_modal_state=lambda: "modal-state",
    )

    class _FakePhysicsData:
        @staticmethod
        def zeros(shape: tuple[int, int], layers: int) -> dict[str, Any]:
            return {"shape": shape, "layers": layers}

    forcing = _FakeForcing()

    monkeypatch.setattr(jax_gcm_module, "PhysicsData", _FakePhysicsData)
    monkeypatch.setattr(
        jax_gcm_module,
        "dynamics_state_to_physics_state",
        lambda modal_state, primitive: {
            "modal_state": modal_state,
            "primitive": primitive,
        },
    )
    monkeypatch.setattr(
        jax_gcm_module,
        "default_forcing",
        lambda horizontal: forcing,
    )
    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: (lambda state, forcing_data: (state, "unused")),
    )

    component.initialize(_make_coupler(dt_seconds=3600.0, run_order=["ATM"]))

    assert component.forcing is forcing
    assert forcing.copy_calls == [{"lfluxland": True}]


def test_jax_gcm_step_maps_outputs_and_respects_output_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    component.data = {
        "sea_surface_temperature": np.asarray([[np.nan, 281.0], [282.0, 283.0]]),
        "land_surface_temperature": np.asarray([[270.0, np.nan], [0.0, 284.0]]),
    }
    component.model = SimpleNamespace(
        terrain=SimpleNamespace(fmask=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    )
    component.forcing = _FakeForcing()
    component.sigma_levels = np.asarray([0.2, 1.0], dtype=float)
    component._predictions_list = []
    component.output_frequency = "day"

    written: dict[str, str] = {}

    p = SimpleNamespace(
        surface_flux=SimpleNamespace(
            shf=np.full((2, 2, 2), 5.0, dtype=float),
            evap=np.full((2, 2, 2), 2.0, dtype=float),
            rlds=np.full((2, 2), 40.0, dtype=float),
        ),
        shortwave_rad=SimpleNamespace(rsns=np.full((2, 2), 30.0, dtype=float)),
    )
    d = SimpleNamespace(
        u_wind=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        v_wind=np.asarray(
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ]
        ),
        temperature=np.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
        specific_humidity=np.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
        normalized_surface_pressure=np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=float),
    )

    monkeypatch.setattr(component, "do_jcm_steps", lambda: (p, d))
    monkeypatch.setattr(
        jax_gcm_module,
        "compute_pressure_levels",
        lambda reference_pressure, top_pressure, sigma_levels, normalized_surface_pressure: jnp.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    monkeypatch.setattr(
        jax_gcm_module,
        "get_altitudes_sigma_levels",
        lambda temperature, pressure, specific_humidity: jnp.asarray(
            [
                np.full((2, 2), 50.0),
                np.full((2, 2), 150.0),
            ]
        ),
    )
    monkeypatch.setattr(component, "_should_write_output", lambda time, dt: True)
    monkeypatch.setattr(
        component,
        "_write_output",
        lambda output: written.__setitem__("path", output),
    )

    coupler = _make_coupler(
        dt_seconds=3600.0, run_order=["ATM"], settings=VercorSettings()
    )
    component.step(timedelta(days=1), datetime(2000, 1, 2), coupler)

    forcing_call = component.forcing.copy_calls[-1]
    assert_allclose_compact(
        forcing_call["stl_am"],
        np.asarray([[270.0, 288.15], [288.15, 567.0]]),
    )
    assert_allclose_compact(
        forcing_call["sea_surface_temperature"],
        np.asarray([[288.15, 282.0], [281.0, 288.15]]),
    )
    assert_allclose_compact(
        component.data["total_surface_temperature"],
        np.asarray([[270.0, 281.0], [282.0, 567.0]]),
    )
    assert_allclose_compact(
        component.data["u_velocity"], np.asarray([[5.0, 7.0], [6.0, 8.0]])
    )
    assert_allclose_compact(
        component.data["v_velocity"], np.asarray([[6.0, 8.0], [7.0, 9.0]])
    )
    assert_allclose_compact(
        component.data["temperature"], np.asarray([[284.0, 286.0], [285.0, 287.0]])
    )
    assert_allclose_compact(
        component.data["specific_humidity"],
        np.asarray([[0.05, 0.07], [0.06, 0.08]]),
    )
    assert_allclose_compact(
        component.data["sensible_heat_flux"], np.full((2, 2), -10.0)
    )
    assert_allclose_compact(
        component.data["latent_heat_flux"],
        np.full((2, 2), -2.0 / 1e3 * coupler.settings.latvap * 2.0),
    )
    assert_allclose_compact(
        component.data["pressure"],
        np.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    assert_allclose_compact(
        component.data["density"],
        coupler.settings.mwdair
        / coupler.settings.rgas
        * np.full((2, 2), 80000.0)
        / np.asarray([[284.0, 286.0], [285.0, 287.0]]),
    )
    assert_allclose_compact(
        component.data["potential_temperature"],
        np.asarray([[284.0, 286.0], [285.0, 287.0]])
        * (coupler.settings.p0 / 80000.0) ** coupler.settings.cappa,
    )
    assert_allclose_compact(
        component.data["model_level_height"], np.full((2, 2), 150.0)
    )
    assert written["path"] == "jcm.averages.2000-01-02.nc"


def test_jax_gcm_write_output_persists_mean_dataset(tmp_path: Path) -> None:
    component = jax_gcm_module.JAXGCM.__new__(jax_gcm_module.JAXGCM)
    dataset = xr.Dataset(
        {
            "temperature": (
                ("time", "wvi_id", "hsg_level", "level", "lat", "lon"),
                np.arange(2.0).reshape(2, 1, 1, 1, 1, 1),
            )
        },
        coords={
            "time": np.asarray(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
            "wvi_id": [0],
            "hsg_level": [0],
            "level": [0],
            "lat": [0],
            "lon": [0],
        },
    )
    component._predictions_list = [_PredictionDataset(dataset=dataset)]

    output = tmp_path / "jcm_output.nc"
    component._write_output(str(output))

    with xr.open_dataset(output) as actual:
        assert actual["temperature"].shape == (1, 1, 1, 1, 1, 1)
        assert np.isclose(float(actual["temperature"].values.squeeze()), 0.5)
    assert component._predictions_list == []


def test_veros_compute_fluxes_zeroes_qnec_for_large_negative_dqfldt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_module.VerosGCM.__new__(veros_gcm_module.VerosGCM)
    component._veros_state = _make_flux_ready_veros_state()
    component.data = {
        "model_level_height": np.full((2, 2), 100.0),
        "u_velocity": np.full((2, 2), 2.0),
        "v_velocity": np.full((2, 2), 3.0),
        "potential_temperature": np.full((2, 2), 280.0),
        "specific_humidity": np.full((2, 2), 0.01),
        "density": np.full((2, 2), 1.2),
        "temperature": np.full((2, 2), 281.0),
        "net_shortwave_radiation_flux": np.full((2, 2), 10.0),
        "downward_longwave_radiation_flux": np.full((2, 2), 20.0),
    }

    captured: dict[str, np.ndarray] = {}

    def fake_new_flux_atm_ocn(
        settings: VercorSettings,
        mask: np.ndarray,
        model_level_height: np.ndarray,
        u_velocity: np.ndarray,
        v_velocity: np.ndarray,
        potential_temperature: np.ndarray,
        specific_humidity: np.ndarray,
        density: np.ndarray,
        temperature: np.ndarray,
        u_tgrid: np.ndarray,
        v_tgrid: np.ndarray,
        surface_temperature: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        _ = (
            settings,
            model_level_height,
            u_velocity,
            v_velocity,
            potential_temperature,
            specific_humidity,
            density,
            temperature,
        )
        captured["mask"] = mask
        captured["u_tgrid"] = u_tgrid
        captured["v_tgrid"] = v_tgrid
        captured["surface_temperature"] = surface_temperature
        return (
            np.full((2, 2), -1.0),
            np.full((2, 2), -2.0),
            np.full((2, 2), -3.0),
            np.full((2, 2), 4.0),
            np.full((2, 2), 5.0),
            np.full((2, 2), 6.0),
            np.full((2, 2), 7.0),
            np.full((2, 2), 8.0),
            np.full((2, 2), 9.0),
            np.full((2, 2), 10.0),
            np.full((2, 2), 11.0),
            np.full((2, 2), 12.0),
            np.asarray([[-1e10, -1e11], [0.5, -2.0]]),
        )

    monkeypatch.setattr(veros_gcm_module, "new_flux_atmOcn", fake_new_flux_atm_ocn)

    taux, tauy, qnet, qnec = veros_gcm_module.compute_fluxes(
        component, VercorSettings()
    )

    assert_allclose_compact(captured["mask"], np.ones((2, 2)))
    assert captured["u_tgrid"].shape == (2, 2)
    assert captured["v_tgrid"].shape == (2, 2)
    assert_allclose_compact(captured["surface_temperature"], np.full((2, 2), 280.15))
    assert_allclose_compact(taux, np.full((2, 2), 5.0))
    assert_allclose_compact(tauy, np.full((2, 2), 6.0))
    assert_allclose_compact(qnet, np.full((2, 2), 24.0))
    assert_allclose_compact(qnec, np.asarray([[0.0, 0.0], [-0.5, 2.0]]))


def test_custom_global_four_degree_set_diagnostics_populates_outputs() -> None:
    state = SimpleNamespace(
        settings=SimpleNamespace(dt_tracer=1200.0),
        diagnostics={
            "snapshot": SimpleNamespace(output_frequency=None),
            "overturning": SimpleNamespace(
                output_frequency=None, sampling_frequency=None
            ),
            "energy": SimpleNamespace(output_frequency=None, sampling_frequency=None),
            "averages": SimpleNamespace(
                output_variables=None,
                output_frequency=None,
                sampling_frequency=None,
            ),
        },
    )

    component = object.__new__(veros_gcm_module.CustomGlobalFourDegree)
    routine = veros_gcm_module.CustomGlobalFourDegree.set_diagnostics.func.__self__
    routine.function(component, state)

    assert state.diagnostics["snapshot"].output_frequency == 365 * 86400.0
    assert state.diagnostics["overturning"].sampling_frequency == 1200.0
    assert state.diagnostics["energy"].sampling_frequency == 86400
    assert state.diagnostics["averages"].output_frequency == 365 * 86400.0
    assert state.diagnostics["averages"].sampling_frequency == 86400
    assert state.diagnostics["averages"].output_variables == [
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "surface_taux",
        "surface_tauy",
        "psi",
        "qnet",
        "qnec",
    ]


def test_veros_copy_state_jitted_path_deep_copies_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(veros_gcm_module, "VerosState", _ConstructedVerosState)

    state = SimpleNamespace(
        _dimensions={"xt": [1, 2]},
        settings=_FakeSettings({"meta": {"precision": "x64"}}, {"dt_tracer": 600.0}),
        _plugin_interfaces={"plugins": ["tracer"]},
        _var_meta={"temp": {"units": "K"}},
        _variables={"temp": [1.0, 2.0]},
        timers={"step": [1.0]},
        profile_timers={"profile": [2.0]},
    )

    copied = veros_gcm_module.copy_state(state, jitted=True)

    assert copied is not state
    assert copied._dimensions == state._dimensions
    assert copied._dimensions is not state._dimensions
    assert copied._plugin_interfaces == state._plugin_interfaces
    assert copied._plugin_interfaces is not state._plugin_interfaces
    assert copied._var_meta == state._var_meta
    assert copied._var_meta is not state._var_meta
    assert copied._variables == state._variables
    assert copied._variables is not state._variables
    assert copied.timers == state.timers
    assert copied.timers is not state.timers
    assert copied.profile_timers == state.profile_timers
    assert copied.profile_timers is not state.profile_timers
    assert copied.settings["dt_tracer"] == 600.0
    assert copied.settings.__metadata__ == state.settings.__metadata__
    assert copied.settings.__metadata__ is not state.settings.__metadata__


def test_veros_pure_runs_step_on_copied_state(monkeypatch: pytest.MonkeyPatch) -> None:
    original_state = SimpleNamespace(counter=1)
    copied_state = SimpleNamespace(counter=1)

    def fake_copy_state(state: Any, jitted: bool = True) -> Any:
        assert state is original_state
        assert jitted is False
        return copied_state

    monkeypatch.setattr(veros_gcm_module, "copy_state", fake_copy_state)

    def fake_step(state: Any) -> None:
        state.counter += 1

    result = veros_gcm_module.pure(original_state, jitted=False, step=fake_step)

    assert result is copied_state
    assert copied_state.counter == 2
    assert original_state.counter == 1


def test_veros_update_veros_interior_supports_jit_and_gradients() -> None:
    array = jnp.zeros((8, 8, 1), dtype=jnp.float64)
    interior = jnp.arange(16.0, dtype=jnp.float64).reshape(4, 4, 1)

    updated = jax.jit(veros_gcm_module._update_veros_interior)(array, interior)

    assert_allclose_compact(updated[2:-2, 2:-2, :], interior)
    assert np.count_nonzero(np.asarray(updated[:2, :, :])) == 0
    assert np.count_nonzero(np.asarray(updated[-2:, :, :])) == 0
    assert np.count_nonzero(np.asarray(updated[:, :2, :])) == 0
    assert np.count_nonzero(np.asarray(updated[:, -2:, :])) == 0

    gradient = jax.grad(
        lambda payload: jnp.sum(veros_gcm_module._update_veros_interior(array, payload))
    )(interior)
    assert_allclose_compact(gradient, np.ones((4, 4, 1)))


def test_veros_prepare_surface_forcing_fields_shapes_nan_cleanup_and_qnec_gate() -> (
    None
):
    taux = jnp.asarray([[1.0, jnp.nan], [3.0, 4.0]])
    tauy = jnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    qnet = jnp.asarray([[9.0, 10.0], [11.0, jnp.nan]])
    qnec = jnp.asarray([[12.0, 13.0], [14.0, 15.0]])

    prepared = jax.jit(veros_gcm_module._prepare_surface_forcing_fields)(
        taux, tauy, qnet, qnec, False
    )
    taux_out, tauy_out, qnet_out, qnec_out = prepared

    assert taux_out.shape == (2, 2, 1)
    assert tauy_out.shape == (2, 2, 1)
    assert qnet_out.shape == (2, 2, 1)
    assert qnec_out.shape == (2, 2, 1)
    assert_allclose_compact(taux_out, np.asarray([[[1.0], [3.0]], [[0.0], [4.0]]]))
    assert_allclose_compact(qnet_out, np.asarray([[[9.0], [11.0]], [[10.0], [0.0]]]))
    assert_allclose_compact(qnec_out, np.zeros((2, 2, 1)))

    restored = veros_gcm_module._prepare_surface_forcing_fields(
        taux, tauy, qnet, qnec, True
    )[3]
    assert_allclose_compact(
        restored,
        np.asarray([[[12.0], [14.0]], [[13.0], [15.0]]]),
    )


def test_veros_set_variable_updates_only_interior_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = _FakeVariableStore(temp=np.zeros((8, 8, 1), dtype=float))
    state = SimpleNamespace(variables=variables)

    monkeypatch.setattr(
        veros_gcm_module, "copy_state", lambda tree, jitted=True: deepcopy(tree)
    )

    updated = veros_gcm_module.set_variable(
        state,
        "temp",
        np.full((4, 4, 1), 9.0),
        jitted=False,
    )

    assert_allclose_compact(
        updated.variables.temp[2:-2, 2:-2, :], np.full((4, 4, 1), 9.0)
    )
    assert np.count_nonzero(updated.variables.temp[:2, :, :]) == 0
    assert np.count_nonzero(updated.variables.temp[-2:, :, :]) == 0
    assert np.count_nonzero(updated.variables.temp[:, :2, :]) == 0
    assert np.count_nonzero(updated.variables.temp[:, -2:, :]) == 0


def test_veros_initialize_validates_timestep_multiple() -> None:
    component = veros_gcm_module.VerosGCM.__new__(veros_gcm_module.VerosGCM)
    component.dt_tracer = 7.0

    with pytest.raises(ValueError, match="dt_tracer"):
        component.initialize(_make_coupler(dt_seconds=20.0, run_order=["OCN"]))


def test_veros_initialize_can_spin_up_and_extract_surface_temperature() -> None:
    component = veros_gcm_module.VerosGCM.__new__(veros_gcm_module.VerosGCM)
    component.dt_tracer = 10.0
    component.do_spinup = True
    component.spinup_time = timedelta(seconds=20.0)
    component.spinup_steps = 2
    component._veros_state = _make_fake_veros_state(surface_temperature=10.0)
    component.data = {}

    step_calls = {"count": 0}

    def fake_step_function(state: Any) -> Any:
        step_calls["count"] += 1
        return state

    component._step_function = fake_step_function

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component.initialize(coupler)

    assert component.model_substeps == 2
    assert step_calls["count"] == 2
    assert_allclose_compact(
        component.data["sea_surface_temperature"],
        np.full((4, 4), 283.15),
    )


@pytest.mark.parametrize(
    ("restore_to_climatology", "expected_qnec"),
    [
        (False, np.zeros((2, 2, 1))),
        (True, np.asarray([[[3.0], [5.0]], [[4.0], [6.0]]])),
    ],
)
def test_veros_step_sets_forcing_fields_and_refreshes_sst(
    monkeypatch: pytest.MonkeyPatch,
    restore_to_climatology: bool,
    expected_qnec: np.ndarray,
) -> None:
    component = veros_gcm_module.VerosGCM.__new__(veros_gcm_module.VerosGCM)
    component.restore_to_climatology = restore_to_climatology
    component.model_substeps = 2
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.data = {}

    set_calls: list[tuple[str, np.ndarray]] = []

    def fake_set_variable(
        state: Any, variable_name: str, variable_value: np.ndarray, jitted: bool = True
    ) -> Any:
        _ = jitted
        set_calls.append((variable_name, np.asarray(variable_value)))
        return state

    def fake_step_function(state: Any) -> Any:
        state.variables.temp = np.full((8, 8, 1, 1), 15.0, dtype=float)
        return state

    monkeypatch.setattr(
        veros_gcm_module,
        "compute_fluxes",
        lambda component_state, settings: (
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, 12.0]]),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        ),
    )
    monkeypatch.setattr(veros_gcm_module, "set_variable", fake_set_variable)
    component._step_function = fake_step_function

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component.step(timedelta(seconds=20.0), datetime(2000, 1, 1), coupler)

    expected_names = ["taux", "tauy", "qnet", "qnec"]
    assert [name for name, _ in set_calls] == expected_names
    assert_allclose_compact(
        set_calls[0][1], np.asarray([[[1.0], [3.0]], [[2.0], [4.0]]])
    )
    assert_allclose_compact(set_calls[3][1], expected_qnec)
    assert_allclose_compact(
        component.data["sea_surface_temperature"],
        np.full((4, 4), 288.15),
    )


def test_veros_step_nan_cleans_forcing_fields_before_set_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_module.VerosGCM.__new__(veros_gcm_module.VerosGCM)
    component.restore_to_climatology = True
    component.model_substeps = 0
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.data = {}

    set_calls: list[tuple[str, np.ndarray]] = []

    def fake_set_variable(
        state: Any, variable_name: str, variable_value: np.ndarray, jitted: bool = True
    ) -> Any:
        _ = jitted
        set_calls.append((variable_name, np.asarray(variable_value)))
        return state

    monkeypatch.setattr(
        veros_gcm_module,
        "compute_fluxes",
        lambda component_state, settings: (
            np.asarray([[1.0, np.nan], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [np.nan, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, np.nan]]),
            np.asarray([[12.0, 13.0], [14.0, np.nan]]),
        ),
    )
    monkeypatch.setattr(veros_gcm_module, "set_variable", fake_set_variable)
    component._step_function = lambda state: state

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component.step(timedelta(seconds=20.0), datetime(2000, 1, 1), coupler)

    assert [name for name, _ in set_calls] == ["taux", "tauy", "qnet", "qnec"]
    assert_allclose_compact(
        set_calls[0][1], np.asarray([[[1.0], [3.0]], [[0.0], [4.0]]])
    )
    assert_allclose_compact(
        set_calls[1][1], np.asarray([[[5.0], [0.0]], [[6.0], [8.0]]])
    )
    assert_allclose_compact(
        set_calls[2][1], np.asarray([[[9.0], [11.0]], [[10.0], [0.0]]])
    )
    assert_allclose_compact(
        set_calls[3][1], np.asarray([[[12.0], [14.0]], [[13.0], [0.0]]])
    )
