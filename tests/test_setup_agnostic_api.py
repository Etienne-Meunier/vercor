from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import vercor
import vercor.topology as topology_module
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
)
from vercor.exceptions import ComponentError, CouplerError
from vercor.output import OutputConfig, PeriodOutput
from vercor.regridding import conservative
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup
from vercor.topology import SurfaceMaskPolicy


def _clock(steps: int = 1) -> Clock:
    return Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=steps)


@pytest.mark.fast_always
def test_surface_mask_policy_is_public_core_configuration() -> None:
    policy = SurfaceMaskPolicy(mode="disabled")

    assert policy.mode == "disabled"
    assert policy.atmosphere == "ATM"
    assert "SurfaceMaskPolicy" in topology_module.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__
    assert not hasattr(vercor, "SurfaceMaskPolicy")


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
        runtime=RuntimeOptions(topology=None),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("DST").field("custom_flux"),
        np.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_duplicate_exchange_topology_key_requires_merged_fields_or_distinct_factory() -> (
    None
):
    grid = make_test_grid(name="duplicate-topology-key")
    source = DataComponent.from_fields(
        "SRC",
        grid,
        {"temperature": 280.0, "humidity": 0.5},
    )
    target = DataComponent.from_fields(
        "DST",
        grid,
        {"temperature": 0.0, "humidity": 0.0},
        spec=ComponentSpec(inputs=("temperature", "humidity")),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(source, target),
        exchanges=(
            Exchange("SRC", "DST", ("temperature",)),
            Exchange("SRC", "DST", ("humidity",)),
        ),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=None),
    )

    with pytest.raises(
        CouplerError,
        match="Duplicate exchange topology key.*merge field declarations.*distinct regrid factories",
    ):
        coupler.initial_state()


@pytest.mark.fast_always
def test_exchange_fan_in_rejects_scalar_conflicts_independent_of_order_and_regrid() -> (
    None
):
    grid = make_test_grid(name="fan-in-scalar")
    source_a = DataComponent.from_fields("SRC_A", grid, {"flux": 1.0})
    source_b = DataComponent.from_fields("SRC_B", grid, {"flux": 2.0})
    target = DataComponent.from_fields(
        "DST",
        grid,
        {"flux": 0.0},
        spec=ComponentSpec(inputs=("flux",)),
    )
    exchanges = (
        Exchange("SRC_A", "DST", ("flux",), label="alpha route"),
        Exchange(
            "SRC_B",
            "DST",
            ("flux",),
            regrid=conservative,
            label="omega route",
        ),
    )

    messages = []
    for declared in (exchanges, tuple(reversed(exchanges))):
        coupler = Coupler(
            clock=_clock(),
            components=(source_a, source_b, target),
            exchanges=declared,
            run_order=("SRC_A", "SRC_B", "DST"),
            runtime=RuntimeOptions(topology=None),
        )
        with pytest.raises(CouplerError) as error:
            coupler.initial_state()
        messages.append(str(error.value))

    assert messages[0] == messages[1]
    assert "DST" in messages[0]
    assert "flux" in messages[0]
    assert "alpha route" in messages[0]
    assert "omega route" in messages[0]
    assert "distinct field names" in messages[0]
    assert "aggregator component" in messages[0]


@pytest.mark.fast_always
def test_exchange_fan_in_flattens_vector_declarations() -> None:
    grid = make_test_grid(name="fan-in-vector")
    source_a = DataComponent.from_fields("SRC_A", grid, {"u": 1.0, "v": 2.0})
    source_b = DataComponent.from_fields("SRC_B", grid, {"v": 3.0})
    target = DataComponent.from_fields(
        "DST",
        grid,
        {"u": 0.0, "v": 0.0},
        spec=ComponentSpec(inputs=("u", "v")),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(source_a, source_b, target),
        exchanges=(
            Exchange(
                "SRC_A",
                "DST",
                (vercor.vector("u", "v"),),
                label="vector route",
            ),
            Exchange("SRC_B", "DST", ("v",), label="scalar route"),
        ),
        run_order=("SRC_A", "SRC_B", "DST"),
        runtime=RuntimeOptions(topology=None),
    )

    with pytest.raises(
        CouplerError,
        match="DST.*v.*scalar route.*vector route.*distinct field names.*aggregator",
    ):
        coupler.initial_state()


@pytest.mark.fast_always
def test_component_can_receive_step_and_send_the_same_field() -> None:
    grid = make_test_grid(name="feedback-field")
    source = DataComponent.from_fields("SRC", grid, {"signal": 2.0})
    middle = Component.from_step(
        "MID",
        grid,
        lambda fields: {"signal": fields["signal"] + 1.0},
        spec=ComponentSpec(
            inputs=("signal",),
            outputs=("signal",),
            defaults={"signal": 0.0},
        ),
    )
    target = DataComponent.from_fields(
        "DST",
        grid,
        {"signal": 0.0},
        spec=ComponentSpec(inputs=("signal",)),
    )
    coupler = Coupler(
        clock=_clock(),
        components=(source, middle, target),
        exchanges=(
            Exchange("SRC", "MID", ("signal",)),
            Exchange("MID", "DST", ("signal",)),
        ),
        run_order=("SRC", "MID", "DST"),
        runtime=RuntimeOptions(topology=None),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("MID").field("signal"),
        np.full(grid.shape, 3.0),
    )
    assert_allclose_compact(
        final_state.component("DST").field("signal"),
        np.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_slab_driver_has_one_bilinear_ocean_to_seaice_temperature_route() -> None:
    source = Path("examples/run_slab_driver.py").read_text(encoding="utf-8")

    assert source.count("fields=OCEAN_TO_SEAICE_SURFACE_FIELDS") == 1
    exchange_block = source.split(
        "fields=OCEAN_TO_SEAICE_SURFACE_FIELDS",
        maxsplit=1,
    )[
        1
    ].split("),", maxsplit=1)[0]
    assert "regrid=bilinear" in exchange_block


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
        runtime=RuntimeOptions(topology=None),
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
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy(mode="required")),
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
        runtime=RuntimeOptions(topology=None),
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
        runtime=RuntimeOptions(topology=None),
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
        runtime=RuntimeOptions(topology=None),
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
