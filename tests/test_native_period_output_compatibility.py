"""Bundled native output paths use ordinary providers and core coordination."""

from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_bundled_factories_install_ordinary_output_providers() -> None:
    jax_gcm = _source("vercor/setups/_external/jax_gcm.py")
    veros = _source("vercor/setups/_external/veros_gcm.py")
    camulator = _source("vercor/setups/_external/camulator.py")

    assert "jax_gcm_output_provider(state)" in jax_gcm
    assert "veros_output_provider(state)" in veros
    assert "camulator_output_provider(state)" in camulator
    assert "variables=(" not in veros.split("veros_output_provider(state)", 1)[0][-200:]
    for source in (jax_gcm, veros, camulator):
        assert "_period_output_handled_by_step" not in source
        assert "_period_output_schema_factory" not in source


def test_native_steps_do_not_own_output_cadence_or_writes() -> None:
    jax_runtime = _source("vercor/setups/_external/jax_gcm_runtime.py")
    veros_runtime = _source("vercor/setups/_external/veros_runtime.py")
    camulator_runtime = _source("vercor/setups/_external/camulator_runtime.py")

    for source in (jax_runtime, veros_runtime, camulator_runtime):
        assert "should_write_period_output" not in source
        assert "write_period" not in source
        assert "record_period" not in source
    assert "period_output" not in jax_runtime


def test_native_output_modules_return_output_frames() -> None:
    jax_output = _source("vercor/setups/_external/jax_gcm_output.py")
    veros_output = _source("vercor/setups/_external/veros_output.py")
    camulator_output = _source("vercor/setups/_external/camulator_output.py")

    for source in (jax_output, veros_output, camulator_output):
        assert "def sample(self, context: OutputContext) -> OutputFrame:" in source
        assert "return OutputFrame(" in source


def test_camulator_v0_4_adapter_does_not_own_output_paths_or_increment_writes() -> None:
    output_source = _source("vercor/setups/_external/camulator_output.py")
    state_source = _source("vercor/setups/_external/camulator_gcm_state.py")
    config_source = _source("vercor/setups/config.py")

    for removed in (
        "camulator_average_output_path",
        "write_camulator_netcdf_increment",
        "write_camulator_prediction_output",
    ):
        assert f"def {removed}(" not in output_source
        assert f'"{removed}"' not in output_source
    assert 'conf["predict"]["save_forecast"]' not in state_source
    assert "output_subfolder_name" not in state_source
    assert "output_subfolder_name" not in config_source
