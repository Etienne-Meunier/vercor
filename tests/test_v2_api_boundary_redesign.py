from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import vercor
import vercor.runtime as runtime
import vercor.topology as topology
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.exceptions import CouplerError


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
    ]
    assert vercor.RuntimeOptions is runtime.RuntimeOptions
    assert vercor.ExecutionContext is runtime.ExecutionContext
    assert vercor.RuntimeDriver is runtime.RuntimeDriver
    assert vercor.RunState is runtime.RunState
    assert vercor.ComponentState is runtime.ComponentState
    assert options.topology is None
    assert options.model_year_seconds == 365 * 86400.0
    assert "year_in_seconds" not in signature(runtime.RuntimeOptions).parameters
    assert "surface_masks" not in signature(runtime.RuntimeOptions).parameters
    assert not hasattr(runtime, "SurfaceMaskPolicy")
    assert not hasattr(vercor, "SurfaceMaskPolicy")
    assert "RuntimeRunContext" not in str(signature(runtime.ExecutionBackend.run))


@pytest.mark.fast_always
def test_topology_module_owns_public_topology_contracts() -> None:
    policy = topology.SurfaceMaskPolicy(mode="disabled")
    patch = topology.ExchangeTopologyPatch(
        fractional_masks={("SRC", "DST", "custom"): jnp.asarray(1.0)}
    )

    assert topology.__all__ == [
        "ExchangeKey",
        "ExchangeTopologyPatch",
        "SurfaceMaskPolicy",
        "TopologyContext",
        "TopologyPolicy",
    ]
    assert policy.mode == "disabled"
    assert patch.fractional_masks[("SRC", "DST", "custom")].shape == ()
    assert "TopologyPolicy" not in vercor.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__


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
def test_custom_backend_runs_complete_host_exchange_order_from_supplied_state() -> None:
    grid = make_test_grid(name="v2-custom-host-backend")
    observed_steps: list[tuple[str, int, object]] = []

    def step_source(
        fields: Mapping[str, Any],
        context: vercor.StepContext,
    ) -> Mapping[str, Any]:
        observed_steps.append(("SRC", int(context.step), context.time))
        return {"flux": fields["flux"] + 1.0}

    def step_target(
        fields: Mapping[str, Any],
        context: vercor.StepContext,
    ) -> Mapping[str, Any]:
        observed_steps.append(("DST", int(context.step), context.time))
        return {"total": fields["total"] + fields["flux"]}

    class SequentialBackend:
        def __init__(self) -> None:
            self.received_state: vercor.RunState | None = None
            self.driver_calls: list[tuple[int, str]] = []

        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> vercor.RunState:
            self.received_state = state
            for step, _, _ in context.clock.iter():
                for component in context.run_order:
                    self.driver_calls.append((step, component))
                    state = driver.step_component(state, component, step=step)
            return state

    source = vercor.HostComponent.from_step(
        "SRC",
        grid,
        step_source,
        spec=vercor.ComponentSpec(
            inputs=("flux",),
            outputs=("flux",),
            defaults={"flux": 1.0},
        ),
    )
    target = vercor.HostComponent.from_step(
        "DST",
        grid,
        step_target,
        spec=vercor.ComponentSpec(
            inputs=("flux", "total"),
            outputs=("total",),
            defaults={"total": 0.0},
        ),
    )
    backend = SequentialBackend()
    coupler = vercor.Coupler(
        _clock(steps=2),
        components=(source, target),
        exchanges=(vercor.Exchange("SRC", "DST", ("flux",)),),
        run_order=("SRC", "DST"),
        runtime=vercor.RuntimeOptions(execution=backend),
    )
    initial_state = coupler.initial_state().replace_fields(
        "SRC",
        {"flux": jnp.full(grid.shape, 10.0)},
    )

    final_state = coupler.run(initial_state)

    assert backend.received_state is initial_state
    assert backend.driver_calls == [
        (0, "SRC"),
        (0, "DST"),
        (1, "SRC"),
        (1, "DST"),
    ]
    assert observed_steps == [
        ("SRC", 0, datetime(2000, 1, 1, 0, 0)),
        ("DST", 0, datetime(2000, 1, 1, 0, 0)),
        ("SRC", 1, datetime(2000, 1, 1, 0, 1)),
        ("DST", 1, datetime(2000, 1, 1, 0, 1)),
    ]
    assert_allclose_compact(
        final_state.component("SRC").field("flux"),
        jnp.full(grid.shape, 12.0),
    )
    assert_allclose_compact(
        final_state.component("DST").field("total"),
        jnp.full(grid.shape, 23.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("returned", "actual_type"),
    (
        pytest.param(None, "NoneType", id="none"),
        pytest.param({"state": "invalid"}, "dict", id="mapping"),
        pytest.param(object(), "object", id="object"),
    ),
)
def test_custom_backend_rejects_non_run_state_return(
    returned: object,
    actual_type: str,
) -> None:
    class InvalidReturnBackend:
        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> Any:
            _ = state, context, driver
            return returned

    grid = make_test_grid(name=f"v2-invalid-backend-{actual_type}")
    coupler = vercor.Coupler(
        _clock(),
        components=(vercor.DataComponent.from_fields("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(execution=InvalidReturnBackend()),
    )

    with pytest.raises(
        CouplerError,
        match=rf"InvalidReturnBackend.*return.*RunState.*{actual_type}",
    ):
        coupler.run()


class _ReturnForeignStateBackend:
    def __init__(self, state: vercor.RunState) -> None:
        self.state = state

    def run(
        self,
        state: vercor.RunState,
        *,
        context: vercor.ExecutionContext,
        driver: vercor.RuntimeDriver,
    ) -> vercor.RunState:
        _ = state, context, driver
        return self.state


def _data_state(
    *components: vercor.DataComponent,
    run_order: tuple[str, ...],
) -> vercor.RunState:
    return vercor.Coupler(
        _clock(),
        components=components,
        run_order=run_order,
        runtime=vercor.RuntimeOptions(topology=None),
    ).initial_state()


@pytest.mark.fast_always
def test_custom_backend_accepts_structurally_compatible_foreign_run_state() -> None:
    grid = make_test_grid(name="compatible-foreign-state")
    foreign_state = _data_state(
        vercor.DataComponent.from_fields("MODEL", grid, {"value": 9.0}),
        run_order=("MODEL",),
    )
    coupler = vercor.Coupler(
        _clock(),
        components=(vercor.DataComponent.from_fields("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(
            topology=None,
            execution=_ReturnForeignStateBackend(foreign_state),
        ),
    )

    result = coupler.run()

    assert result is foreign_state
    assert_allclose_compact(
        result.component("MODEL").field("value"),
        jnp.full(grid.shape, 9.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "case",
    (
        pytest.param("missing", id="missing-component"),
        pytest.param("extra", id="extra-component"),
        pytest.param("extra-field", id="extra-field"),
        pytest.param("shape", id="incompatible-field-shape"),
    ),
)
def test_custom_backend_validates_returned_run_state_schema(case: str) -> None:
    grid = make_test_grid(name=f"custom-backend-schema-{case}")
    if case == "missing":
        foreign_state = _data_state(
            vercor.DataComponent.from_fields("OTHER", grid, {"value": 1.0}),
            run_order=("OTHER",),
        )
        message = "missing.*MODEL"
    elif case == "extra":
        foreign_state = _data_state(
            vercor.DataComponent.from_fields("MODEL", grid, {"value": 1.0}),
            vercor.DataComponent.from_fields("EXTRA", grid, {"value": 2.0}),
            run_order=("MODEL",),
        )
        message = "extra.*EXTRA"
    elif case == "extra-field":
        foreign_state = _data_state(
            vercor.DataComponent.from_fields(
                "MODEL",
                grid,
                {"value": 1.0, "extra_field": 2.0},
            ),
            run_order=("MODEL",),
        )
        message = "MODEL.*fields.*extra_field"
    else:
        wide_grid = make_test_grid(
            name="custom-backend-schema-wide",
            longitude=np.asarray([0.0, 1.0, 2.0]),
        )
        foreign_state = _data_state(
            vercor.DataComponent.from_fields("MODEL", wide_grid, {"value": 1.0}),
            run_order=("MODEL",),
        )
        message = r"value.*MODEL.*shape \(2, 3\).*expected.*grid shape \(2, 2\)"

    coupler = vercor.Coupler(
        _clock(),
        components=(vercor.DataComponent.from_fields("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(
            topology=None,
            execution=_ReturnForeignStateBackend(foreign_state),
        ),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("case", "step", "message"),
    (
        pytest.param("state", 0, "state.*RunState.*object", id="state"),
        pytest.param(
            "component",
            0,
            "component.*UNKNOWN.*prepared.*MODEL",
            id="unknown-component",
        ),
        pytest.param("step", True, "step.*boolean", id="boolean"),
        pytest.param("step", 0.5, "step.*integer.*0.5", id="fractional"),
        pytest.param(
            "step",
            jnp.asarray([0, 1]),
            r"step.*scalar.*shape \(2,\)",
            id="non-scalar",
        ),
        pytest.param("step", -1, r"step -1.*\[0, 2\)", id="negative"),
        pytest.param("step", 2, r"step 2.*\[0, 2\)", id="equal-to-steps"),
        pytest.param("step", 3, r"step 3.*\[0, 2\)", id="beyond-range"),
    ),
)
def test_runtime_driver_rejects_invalid_dispatch_before_component_step(
    case: str,
    step: object,
    message: str,
) -> None:
    class InvalidDriverCallBackend:
        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> vercor.RunState:
            _ = context
            if case == "state":
                return driver.step_component(object(), "MODEL", step=0)  # type: ignore[arg-type]
            component = "UNKNOWN" if case == "component" else "MODEL"
            return driver.step_component(
                state,
                component,
                step=step,  # type: ignore[arg-type]
            )

    grid = make_test_grid(name=f"v2-driver-{case}")
    coupler = vercor.Coupler(
        _clock(steps=2),
        components=(vercor.DataComponent.from_fields("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(execution=InvalidDriverCallBackend()),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()


@pytest.mark.fast_always
def test_runtime_driver_accepts_integer_jax_scalar_and_uses_requested_time() -> None:
    grid = make_test_grid(name="v2-driver-jax-step")
    observed_contexts: list[vercor.StepContext] = []

    def step_model(
        fields: Mapping[str, Any],
        context: vercor.StepContext,
    ) -> Mapping[str, Any]:
        observed_contexts.append(context)
        return {"value": fields["value"] + 1.0}

    class JAXScalarStepBackend:
        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> vercor.RunState:
            _ = context
            return driver.step_component(
                state,
                "MODEL",
                step=jnp.asarray(1, dtype=jnp.int32),
            )

    component = vercor.HostComponent.from_step(
        "MODEL",
        grid,
        step_model,
        spec=vercor.ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            defaults={"value": 1.0},
        ),
    )
    coupler = vercor.Coupler(
        _clock(steps=3),
        components=(component,),
        run_order=("MODEL",),
        runtime=vercor.RuntimeOptions(execution=JAXScalarStepBackend()),
    )

    final_state = coupler.run()

    assert len(observed_contexts) == 1
    assert int(observed_contexts[0].step) == 1
    assert observed_contexts[0].time == datetime(2000, 1, 1, 0, 1)
    assert_allclose_compact(
        final_state.component("MODEL").field("value"),
        jnp.full(grid.shape, 2.0),
    )


@pytest.mark.fast_always
def test_runtime_backends_own_loops_without_importing_runner() -> None:
    package_root = Path(vercor.__file__).parent
    backend_source = (package_root / "_runtime" / "backends.py").read_text()
    runner_source = (package_root / "_runtime" / "runner.py").read_text()

    assert "vercor._runtime.runner" not in backend_source
    assert "class _JAXScannedBackend" not in backend_source
    assert "class _HostLoopBackend" not in backend_source
    for implementation in (
        "run_compiled_scanned_runtime",
        "run_scanned_runtime",
        "run_host_runtime",
    ):
        assert f"def {implementation}" in backend_source
        assert f"def {implementation}" not in runner_source


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
    assert vercor.CouplerSpec is coupling.CouplerSpec
    assert not hasattr(recipes, "CouplerSpec")
    assert "CouplerSpec" not in recipes.__all__
    assert isinstance(coupler, vercor.Coupler)
