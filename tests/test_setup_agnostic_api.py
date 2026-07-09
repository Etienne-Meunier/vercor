from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import vercor
import vercor.setups._jcm as jcm_setup_module
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor import (
    Clock,
    Component,
    ComponentSpec,
    Coupler,
    DataComponent,
    Exchange,
    HostComponent,
    LifecycleHooks,
    RuntimeOptions,
    StepContext,
    SurfaceMaskPolicy,
)
from vercor.exceptions import ComponentError, CouplerError
from vercor.output import OutputConfig, PeriodOutput
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup


def _clock(steps: int = 1) -> Clock:
    return Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=steps)


@pytest.mark.fast_always
def test_surface_mask_policy_is_public_core_configuration() -> None:
    policy = SurfaceMaskPolicy(mode="disabled")

    assert policy.mode == "disabled"
    assert policy.atmosphere == "ATM"
    assert "SurfaceMaskPolicy" in vercor.__all__
    assert vercor.SurfaceMaskPolicy is SurfaceMaskPolicy


@pytest.mark.fast_always
def test_custom_named_components_can_exchange_custom_fields_without_surface_masks() -> (
    None
):
    grid = make_test_grid(name="custom-grid")
    source = DataComponent.from_fields(
        "SRC",
        grid,
        {"custom_flux": 1.0},
    )

    def step(fields: dict[str, Any], context: StepContext) -> dict[str, Any]:
        return {
            "custom_flux": fields["custom_flux"] + context.step,
        }

    target = Component.from_step(
        "DST",
        grid,
        step,
        spec=ComponentSpec(
            inputs=("custom_flux",),
            outputs=("custom_flux",),
            defaults={"custom_flux": 0.0},
        ),
    )
    coupler = Coupler(
        clock=_clock(steps=3),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("custom_flux",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(surface_masks=None),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("DST").field("custom_flux"),
        np.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_exchanged_fields_must_be_declared_by_receiving_component() -> None:
    grid = make_test_grid(name="undeclared-grid")
    source = DataComponent.from_fields(
        "SRC",
        grid,
        {"custom_flux": 1.0},
    )
    target = Component.from_step(
        "DST",
        grid,
        lambda fields: {"other": fields["other"]},
        spec=ComponentSpec(
            inputs=("other",),
            outputs=("other",),
            defaults={"other": 0.0},
        ),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("custom_flux",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(surface_masks=None),
    )

    with pytest.raises(ComponentError, match="custom_flux.*DST.*declare"):
        coupler.initial_state()


@pytest.mark.fast_always
def test_required_surface_mask_policy_preserves_missing_role_errors() -> None:
    grid = make_test_grid(name="required-policy-grid")
    source = DataComponent.from_fields("SRC", grid, {"temperature": 1.0})
    target = DataComponent.from_fields(
        "DST",
        grid,
        {"temperature": 0.0},
        spec=ComponentSpec(inputs=("temperature",)),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("temperature",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(surface_masks=SurfaceMaskPolicy(mode="required")),
    )

    with pytest.raises(CouplerError, match="role component 'LND'"):
        coupler.initial_state()


@pytest.mark.fast_always
def test_step_context_step_increments_in_scanned_runtime() -> None:
    grid = make_test_grid(name="scanned-step-grid")

    component = Component.from_step(
        "MODEL",
        grid,
        lambda fields, context: {
            "temperature": jnp.full_like(fields["temperature"], context.step)
        },
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 0.0},
        ),
    )
    coupler = Coupler(
        clock=_clock(steps=3),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(surface_masks=None),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        np.full(grid.shape, 2.0),
    )


@pytest.mark.fast_always
def test_step_context_step_increments_in_host_runtime() -> None:
    grid = make_test_grid(name="host-step-grid")

    component = HostComponent.from_step(
        "HOST",
        grid,
        lambda fields, context: {
            "temperature": jnp.full_like(fields["temperature"], context.step)
        },
        spec=ComponentSpec(
            outputs=("temperature",),
            defaults={"temperature": 0.0},
        ),
    )
    coupler = Coupler(
        clock=_clock(steps=3),
        components=(component,),
        run_order=("HOST",),
        runtime=RuntimeOptions(surface_masks=None),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("HOST").field("temperature"),
        np.full(grid.shape, 2.0),
    )


@pytest.mark.fast_always
def test_no_exchange_components_run_initialize_hooks_before_state_creation() -> None:
    grid = make_test_grid(name="no-exchange-init-grid")
    events: list[tuple[str, tuple[str, ...]]] = []

    def initialize(component: DataComponent, context: vercor.SetupContext) -> None:
        events.append((component.name, tuple(context.run_order)))
        component.seed_field("temperature", 280.0)

    component = DataComponent.from_fields(
        "ONLY",
        grid,
        spec=ComponentSpec(lifecycle=LifecycleHooks(initialize=initialize)),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(component,),
        run_order=("ONLY",),
        runtime=RuntimeOptions(surface_masks=None),
    )

    state = coupler.initial_state()

    assert events == [("ONLY", ("ONLY",))]
    assert_allclose_compact(
        state.component("ONLY").field("temperature"),
        np.full(grid.shape, 280.0),
    )


@pytest.mark.fast_always
def test_make_jcm_land_atmosphere_accepts_jax_gcm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocn_grid = make_test_grid(name="ocn")
    jcm_grid = make_test_grid(name="jcm", binary_mask=np.ones((2, 2)))
    coords = SimpleNamespace(horizontal=SimpleNamespace())
    terrain = SimpleNamespace(fmask=None)
    forcing = object()
    captured_config: dict[str, JAXGCMConfig] = {}

    def fake_load_inputs() -> jcm_setup_module.JCMInputs:
        return jcm_setup_module.JCMInputs(
            coords=coords,
            terrain=terrain,
            forcing=forcing,
        )

    def fake_make_jcm_land(
        loaded_coords: object,
        loaded_forcing: object,
        loaded_ocn_grid: object,
        *,
        name: str = "LND",
    ) -> DataComponent:
        assert loaded_coords is coords
        assert loaded_forcing is forcing
        assert loaded_ocn_grid is ocn_grid
        return DataComponent.from_fields(
            name,
            jcm_grid,
            {"land_surface_temperature": 280.0},
        )

    def fake_make_jax_gcm(
        loaded_coords: object,
        loaded_terrain: object,
        *,
        config: JAXGCMConfig | None = None,
    ) -> Component:
        assert loaded_coords is coords
        assert loaded_terrain is terrain
        assert config is not None
        captured_config["value"] = config
        return Component.from_step(
            config.name,
            jcm_grid,
            lambda fields: {},
            spec=ComponentSpec(outputs=("temperature",)),
        )

    monkeypatch.setattr(jcm_setup_module, "load_jcm_inputs", fake_load_inputs)
    monkeypatch.setattr(
        jcm_setup_module,
        "_load_jcm_factories",
        lambda: (fake_make_jcm_land, fake_make_jax_gcm),
    )

    config = JAXGCMConfig(
        name="CUSTOM_ATM",
        custom_parameters={"surface_flux.vgust": 5.01},
        spinup=Spinup(enabled=False),
        output=OutputConfig(period=PeriodOutput(frequency="day")),
        jitted=False,
    )
    setup = jcm_setup_module.make_jcm_land_atmosphere(
        ocn_grid,
        config=JCMLandAtmosphereConfig(atmosphere=config),
    )

    assert setup.atmosphere.name == "CUSTOM_ATM"
    assert captured_config["value"].name == config.name
    assert captured_config["value"].custom_parameters == config.custom_parameters
    assert captured_config["value"].forcing_data is forcing
    assert captured_config["value"].spinup == config.spinup
    assert captured_config["value"].output == config.output
    assert captured_config["value"].jitted == config.jitted
