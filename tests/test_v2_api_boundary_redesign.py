from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

import vercor
import vercor.runtime as runtime
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact


def _clock(steps: int = 1) -> vercor.Clock:
    return vercor.Clock(
        start=datetime(2000, 1, 1),
        dt_seconds=60.0,
        steps=steps,
    )


@pytest.mark.fast_always
def test_runtime_module_owns_public_runtime_contracts() -> None:
    options = runtime.RuntimeOptions()

    assert runtime.__all__ == [
        "ComponentState",
        "DTypePolicy",
        "ExecutionBackend",
        "ExecutionContext",
        "ExecutionMode",
        "RunState",
        "RuntimeDriver",
        "RuntimeOptions",
        "SurfaceMaskPolicy",
    ]
    assert vercor.RuntimeOptions is runtime.RuntimeOptions
    assert vercor.ExecutionContext is runtime.ExecutionContext
    assert vercor.RuntimeDriver is runtime.RuntimeDriver
    assert vercor.RunState is runtime.RunState
    assert vercor.ComponentState is runtime.ComponentState
    assert options.surface_masks is None
    assert options.model_year_seconds == 365 * 86400.0
    assert "year_in_seconds" not in signature(runtime.RuntimeOptions).parameters
    assert "RuntimeRunContext" not in str(signature(runtime.ExecutionBackend.run))


@pytest.mark.fast_always
def test_custom_execution_backend_receives_public_context_and_driver() -> None:
    grid = make_test_grid(name="v2-custom-backend")

    class StepOnceBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[vercor.ExecutionContext, object]] = []

        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> vercor.RunState:
            self.calls.append((context, driver))
            assert context.run_order == ("MODEL",)
            assert context.options.execution is self
            return driver.step_component(state, "MODEL", step=0)

    backend = StepOnceBackend()
    component = vercor.Component.from_step(
        "MODEL",
        grid,
        lambda fields, context: {"temperature": fields["temperature"] + 1.0},
        spec=vercor.ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            defaults={"temperature": 280.0},
        ),
    )
    coupler = vercor.Coupler(
        _clock(),
        components=(component,),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(execution=backend),
    )

    final_state = coupler.run()

    assert len(backend.calls) == 1
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_structural_component_can_request_host_runtime_through_spec() -> None:
    class PlainHostComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="v2-plain-host")
            self.spec = vercor.ComponentSpec(
                inputs=("temperature",),
                outputs=("temperature",),
                defaults={"temperature": 280.0},
                execution="host",
            )

        def initial_fields(self) -> Mapping[str, Any]:
            return {}

        def initialize(self, context: vercor.SetupContext) -> None:
            assert context.run_order == ("MODEL",)

        def step(
            self,
            fields: Mapping[str, Any],
            context: vercor.StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = payload
            return {
                "temperature": fields["temperature"] + jnp.asarray(context.step) + 1.0
            }

    component = PlainHostComponent()
    coupler = vercor.Coupler(
        _clock(steps=2),
        components=(component,),
        run_order=("MODEL",),
    )

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 283.0),
    )


@pytest.mark.fast_always
def test_data_import_policy_belongs_to_data_component_not_component_spec() -> None:
    policy = vercor.FieldImportPolicy(time_interpolation=True)
    grid = make_test_grid(name="v2-import-policy")
    spec = vercor.ComponentSpec(outputs=("temperature",))

    component = vercor.DataComponent.from_fields(
        "OBS",
        grid,
        fields={"temperature": 280.0},
        spec=spec,
        import_policy=policy,
    )

    assert not hasattr(spec, "import_policy")
    assert component.import_policy is policy


@pytest.mark.fast_always
def test_coupler_components_exposes_component_info_not_internal_adapters() -> None:
    class PlainComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="v2-component-info")
            self.spec = vercor.ComponentSpec(outputs=("temperature",))

        def initial_fields(self) -> Mapping[str, Any]:
            return {"temperature": 280.0}

        def initialize(self, context: vercor.SetupContext) -> None:
            _ = context

        def step(
            self,
            fields: Mapping[str, Any],
            context: vercor.StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = fields, context, payload
            return {}

    coupler = vercor.Coupler(
        _clock(),
        components=(PlainComponent(),),
        run_order=("MODEL",),
    )

    info = coupler.components["MODEL"]

    assert isinstance(info, vercor.ComponentInfo)
    assert info.name == "MODEL"
    assert info.grid.name == "v2-component-info"
    assert info.spec.outputs == ("temperature",)
    assert not isinstance(info, vercor.Component)
    with pytest.raises(TypeError):
        coupler.components["OTHER"] = info  # type: ignore[index]


@pytest.mark.fast_always
def test_snapshot_writer_receives_component_info(tmp_path: Path) -> None:
    grid = make_test_grid(name="v2-snapshot-component-info")
    seen: list[vercor.ComponentInfo] = []

    def writer(context: vercor.SnapshotContext) -> None:
        seen.append(context.component)

    component = vercor.Component.from_step(
        "MODEL",
        grid,
        lambda fields: {"temperature": fields["temperature"]},
        spec=vercor.ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            defaults={"temperature": 280.0},
            output=vercor.OutputConfig(snapshot_writer=writer),
        ),
    )
    coupler = vercor.Coupler(
        _clock(),
        components=(component,),
        run_order=("MODEL",),
    )

    coupler.write_outputs(coupler.run(), output_dir=tmp_path)

    assert seen == [
        vercor.ComponentInfo(name="MODEL", grid=grid, spec=component.spec),
    ]


@pytest.mark.fast_always
def test_coupling_module_owns_generic_coupler_recipe() -> None:
    import vercor.coupling as coupling
    import vercor.recipes as recipes

    grid = make_test_grid(name="v2-coupler-spec")
    component = vercor.DataComponent.from_fields(
        "MODEL",
        grid,
        {"temperature": 280.0},
    )
    recipe = coupling.CouplerSpec(
        components=(component,),
        run_order=("MODEL",),
    )

    coupler = recipe.build(_clock())

    assert coupling.Coupler is vercor.Coupler
    assert coupling.Exchange is vercor.Exchange
    assert recipes.CouplerSpec is coupling.CouplerSpec
    assert "CouplerSpec" not in recipes.__all__
    assert isinstance(coupler, vercor.Coupler)
