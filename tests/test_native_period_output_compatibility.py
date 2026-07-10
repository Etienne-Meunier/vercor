from __future__ import annotations

from datetime import datetime
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import Component, ComponentSpec, HostComponent
from vercor.coupler import Coupler
from vercor.output import OutputConfig, PeriodOutput
import vercor.output._session as output_session_module
from vercor.output._session import build_period_output_plan
from vercor.runtime import RuntimeOptions
from vercor.setups import CAMulatorConfig, VerosConfig
import vercor.setups._external.camulator as camulator_module
import vercor.setups._external.camulator_gcm_state as camulator_gcm_state_module
import vercor.setups._external.camulator_output as camulator_output_module
import vercor.setups._external.camulator_runtime as camulator_runtime_module
import vercor.setups._external.veros_gcm as veros_gcm_module


def _host_coupler(component: Any) -> Coupler:
    return Coupler(
        Clock(datetime(2000, 1, 1), 86_400.0, 1),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(execution="host"),
    )


def test_veros_native_period_variables_prepare_and_run_without_generic_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_writes: list[datetime] = []

    class FakeVerosState:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs
            self.grid = make_test_grid(name="veros-native-output")

        def initialize(self, component: Any, context: Any) -> None:
            _ = component, context

    def step_veros_runtime(
        state: FakeVerosState,
        fields: Any,
        context: Any,
        payload: Any,
    ) -> dict[str, Any]:
        _ = state, fields, payload
        native_writes.append(context.time)
        return {"sea_surface_temperature": jnp.full((2, 2), 284.0)}

    fake_veros = ModuleType("veros")
    setattr(fake_veros, "runtime_settings", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "veros", fake_veros)
    monkeypatch.setattr(
        veros_gcm_module,
        "_load_veros_implementation",
        lambda: (
            SimpleNamespace(
                VEROS_INPUT_FIELD_NAMES=(),
                veros_default_fields=lambda: {"sea_surface_temperature": 283.15},
            ),
            SimpleNamespace(write_veros_snapshot_output=lambda *args: None),
            SimpleNamespace(step_veros_runtime=step_veros_runtime),
            FakeVerosState,
        ),
    )

    component = veros_gcm_module.make_veros_gcm(
        config=VerosConfig(
            output=OutputConfig(
                period=PeriodOutput(
                    frequency="day",
                    variables=("native_veros_temperature",),
                )
            )
        )
    )

    result = _host_coupler(component).run()

    assert result.component("OCN").field("sea_surface_temperature").shape == (2, 2)
    assert native_writes == [datetime(2000, 1, 1)]


def test_camulator_native_period_mode_does_not_build_duplicate_generic_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_writes: list[datetime] = []

    class FakeCAMulatorState:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs
            self.grid = make_test_grid(name="camulator-native-output")

        def initialize(self, component: Any, context: Any) -> None:
            _ = component, context

    def step_camulator_runtime(
        state: FakeCAMulatorState,
        fields: Any,
        context: Any,
        payload: Any,
    ) -> dict[str, Any]:
        _ = state, fields, payload
        native_writes.append(context.time)
        return {}

    def fail_generic_schema(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        pytest.fail("CAMulator native period output must not build a generic schema")

    monkeypatch.setattr(
        camulator_gcm_state_module,
        "CAMulatorGCMSetupState",
        FakeCAMulatorState,
    )
    monkeypatch.setattr(
        camulator_runtime_module,
        "step_camulator_runtime",
        step_camulator_runtime,
    )
    monkeypatch.setattr(
        camulator_output_module,
        "write_camulator_snapshot_output",
        lambda *args: None,
    )
    monkeypatch.setattr(
        output_session_module,
        "_generic_period_output_schema",
        fail_generic_schema,
    )

    component = camulator_module.make_camulator_gcm(
        config=CAMulatorConfig(
            config_path="unused.yaml",
            device="cpu",
            output=OutputConfig(period=PeriodOutput(frequency="day")),
        )
    )

    _host_coupler(component).run()

    assert native_writes == [datetime(2000, 1, 1)]


def test_mixed_period_plan_keeps_generic_schema_and_skips_native_host_owner() -> None:
    grid = make_test_grid(name="mixed-period-output")
    native = HostComponent.from_step(
        name="native",
        grid=grid,
        step=lambda fields: {"runtime_native": fields["runtime_native"]},
        spec=ComponentSpec(
            outputs=("runtime_native",),
            defaults={"runtime_native": 1.0},
            output=OutputConfig(
                period=PeriodOutput(
                    frequency="day",
                    variables=("model_native_temperature",),
                )
            ),
        ),
    )
    setattr(native, "_period_output_handled_by_step", True)
    generic = Component.from_step(
        name="generic",
        grid=grid,
        step=lambda fields: {"temperature": fields["temperature"]},
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 280.0},
            output=OutputConfig(period=PeriodOutput(frequency="day")),
        ),
    )
    clock = Clock(datetime(2000, 1, 1), 86_400.0, 1)
    coupler = Coupler(
        clock,
        components=(native, generic),
        run_order=("native", "generic"),
        runtime=RuntimeOptions(execution="host"),
    )

    plan = build_period_output_plan(
        {"native": native, "generic": generic},
        coupler.initial_state(),
        clock,
    )

    assert tuple(schema.component_name for schema in plan.schemas) == ("generic",)
