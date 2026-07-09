from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime
import importlib
from typing import Any

import jax.numpy as jnp
import pytest

import vercor
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor import Clock, ComponentSpec, Coupler, Exchange, StepContext


@pytest.mark.fast_always
def test_runtime_options_own_core_runtime_configuration() -> None:
    from vercor.config import RuntimeOptions, SurfaceMaskPolicy

    runtime = RuntimeOptions(
        surface_masks=SurfaceMaskPolicy(mode="disabled"),
        year_in_seconds=360.0,
    )

    assert runtime.surface_masks == SurfaceMaskPolicy(mode="disabled")
    assert runtime.dtype.enable_x64 is False
    assert runtime.execution == "auto"
    assert runtime.year_in_seconds == 360.0
    assert vercor.RuntimeOptions is RuntimeOptions
    assert vercor.SurfaceMaskPolicy is SurfaceMaskPolicy
    assert "RuntimeOptions" in vercor.__all__
    assert "SurfaceMaskPolicy" in vercor.__all__

    with pytest.raises(ModuleNotFoundError, match="vercor.setup_config"):
        importlib.import_module("vercor.setup_config")


@pytest.mark.fast_always
def test_component_spec_freezes_mapping_inputs_and_exposes_import_policy() -> None:
    from vercor.components import FieldImportPolicy

    defaults: dict[str, object] = {"temperature": 280.0}
    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("heat_flux",),
        defaults=defaults,
        import_policy=FieldImportPolicy(time_interpolation=True),
    )
    defaults["temperature"] = 999.0

    assert spec.inputs == ("temperature",)
    assert spec.defaults["temperature"] == 280.0
    assert spec.import_policy.time_interpolation is True
    assert spec.import_policy.daily_selection is False
    with pytest.raises(TypeError):
        spec.defaults["temperature"] = 281.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.import_policy.time_interpolation = False  # type: ignore[misc]


@pytest.mark.fast_always
def test_structural_component_like_runs_without_private_component_internals() -> None:
    from vercor.config import RuntimeOptions

    class PlainComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-component")
            self.spec = ComponentSpec(
                inputs=("heat_flux",),
                outputs=("temperature",),
                defaults={"temperature": 280.0, "heat_flux": 0.0},
            )

        def initial_fields(self) -> Mapping[str, Any]:
            return {"temperature": 280.0, "heat_flux": 1.5}

        def initialize(self, context: vercor.SetupContext) -> None:
            assert tuple(context.run_order) == ("MODEL",)

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = payload
            return {
                "temperature": fields["temperature"]
                + fields["heat_flux"]
                * jnp.asarray(context.dt_seconds)
                * jnp.asarray(context.step + 1)
            }

    component = PlainComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=2.0, steps=2),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(surface_masks=None),
    )

    final_state = coupler.run()

    assert not hasattr(component, "_data")
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 289.0),
    )


@pytest.mark.fast_always
def test_coupler_spec_builds_coupler_from_plain_recipe() -> None:
    from vercor.config import RuntimeOptions
    from vercor.recipes import CouplerSpec

    grid = make_test_grid(name="coupler-spec")
    forcing = vercor.DataComponent.from_fields("SRC", grid, {"flux": 2.0})
    model = vercor.Component.from_step(
        "DST",
        grid,
        lambda fields: {"flux": fields["flux"] + 1.0},
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("flux",),
            defaults={"flux": 0.0},
        ),
    )
    recipe = CouplerSpec(
        components=(forcing, model),
        exchanges=(Exchange("SRC", "DST", ("flux",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(surface_masks=None),
    )

    coupler = recipe.build(Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))

    assert isinstance(coupler, Coupler)
    assert coupler.runtime.surface_masks is None
    assert_allclose_compact(
        coupler.run().component("DST").field("flux"),
        jnp.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_runtime_options_accept_custom_execution_backend() -> None:
    from vercor.config import RuntimeOptions

    class RecordingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            state: vercor.RunState,
            *,
            context: object,
        ) -> vercor.RunState:
            self.calls += 1
            assert hasattr(context, "run_order")
            return state.replace_fields(
                "MODEL",
                {"temperature": jnp.full(grid.shape, 301.0)},
            )

    grid = make_test_grid(name="custom-backend")
    backend = RecordingBackend()
    component = vercor.DataComponent.from_fields(
        "MODEL",
        grid,
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(surface_masks=None, execution=backend),
    )

    final_state = coupler.run()

    assert backend.calls == 1
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(grid.shape, 301.0),
    )
