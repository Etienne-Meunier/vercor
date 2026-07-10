from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime
import importlib
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

import vercor
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor import Clock, ComponentSpec, Coupler, Exchange, StepContext
from vercor.runtime import RuntimeOptions
from vercor.topology import SurfaceMaskPolicy


@pytest.mark.fast_always
def test_runtime_options_own_core_runtime_configuration() -> None:
    runtime = RuntimeOptions(
        topology=SurfaceMaskPolicy(mode="disabled"),
        model_year_seconds=360.0,
    )

    assert runtime.topology == SurfaceMaskPolicy(mode="disabled")
    assert runtime.dtype.enable_x64 is False
    assert runtime.execution == "auto"
    assert runtime.model_year_seconds == 360.0
    assert vercor.RuntimeOptions is RuntimeOptions
    assert "RuntimeOptions" in vercor.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__
    assert not hasattr(vercor, "SurfaceMaskPolicy")

    with pytest.raises(ModuleNotFoundError, match="vercor.config"):
        importlib.import_module("vercor.config")
    with pytest.raises(ModuleNotFoundError, match="vercor.setup_config"):
        importlib.import_module("vercor.setup_config")


@pytest.mark.fast_always
def test_component_spec_freezes_mapping_inputs_and_exposes_execution_policy() -> None:
    defaults: dict[str, object] = {"temperature": 280.0}
    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("heat_flux",),
        defaults=defaults,
        execution="host",
    )
    defaults["temperature"] = 999.0

    assert spec.inputs == ("temperature",)
    assert spec.defaults["temperature"] == 280.0
    assert spec.execution == "host"
    assert not hasattr(spec, "import_policy")
    with pytest.raises(TypeError):
        spec.defaults["temperature"] = 281.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.execution = "jax"  # type: ignore[misc]

    grid = make_test_grid(name="execution-precedence")
    capable_host = vercor.Component.from_step(
        "CAPABLE_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="host"),
    )
    enforced_host = vercor.HostComponent.from_step(
        "ENFORCED_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="jax"),
    )

    assert capable_host.spec.execution == "host"
    assert enforced_host.spec.execution == "host"


@pytest.mark.fast_always
def test_structural_component_like_runs_without_private_component_internals() -> None:
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
        runtime=RuntimeOptions(topology=None),
    )

    final_state = coupler.run()

    assert not hasattr(component, "_data")
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 289.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"name": ""}, "name.*non-empty string"),
        ({"name": "   "}, "name.*non-empty string"),
        ({"name": 7}, "name.*non-empty string"),
        ({"grid": object()}, "grid.*RectilinearGrid"),
        ({"spec": object()}, "spec.*ComponentSpec"),
        ({"initial_fields": None}, "initial_fields.*callable"),
        ({"initialize": None}, "initialize.*callable"),
        ({"step": None}, "step.*callable"),
        ({"initial_fields": lambda: ()}, "initial_fields.*mapping"),
    ),
)
def test_structural_component_validation_is_actionable(
    override: dict[str, object],
    message: str,
) -> None:
    candidate = SimpleNamespace(
        name="MODEL",
        grid=make_test_grid(name="invalid-plain-component"),
        spec=ComponentSpec(),
        initial_fields=lambda: {},
        initialize=lambda context: None,
        step=lambda fields, context, payload=None: {},
    )
    for attribute, value in override.items():
        setattr(candidate, attribute, value)

    with pytest.raises(vercor.ComponentError, match=message):
        Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(candidate,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(topology=None),
        )


@pytest.mark.fast_always
def test_structural_lifecycle_hooks_receive_original_object_in_order() -> None:
    events: list[str] = []
    hook_owners: list[object] = []

    def initialize_hook(owner: object, context: vercor.SetupContext) -> None:
        component = owner
        hook_owners.append(owner)
        events.append("lifecycle-initialize")
        assert component.fields["temperature"] == 281.0  # type: ignore[attr-defined]
        component.fields["temperature"] = 282.0  # type: ignore[attr-defined]
        assert context.run_order == ("MODEL",)

    def create_payload_hook(owner: object) -> dict[str, str]:
        hook_owners.append(owner)
        events.append("create-payload")
        return {"owner": owner.name}  # type: ignore[attr-defined]

    def prefill_hook(
        owner: object,
        context: vercor.PrefillContext,
    ) -> None:
        hook_owners.append(owner)
        events.append("prefill")
        assert float(jnp.mean(context.fields["temperature"])) == 282.0

    def validate_hook(
        owner: object,
        context: vercor.ValidationContext,
    ) -> None:
        hook_owners.append(owner)
        events.append("validate")
        assert isinstance(context.state, vercor.ComponentState)
        assert context.payload == {"owner": "MODEL"}

    class PlainLifecycleComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-lifecycle")
            self.fields = {"temperature": 280.0}
            self.spec = ComponentSpec(
                outputs=("temperature",),
                lifecycle=vercor.LifecycleHooks(
                    initialize=initialize_hook,
                    create_payload=create_payload_hook,
                    prefill=prefill_hook,
                    validate=validate_hook,
                ),
            )

        def initial_fields(self) -> Mapping[str, Any]:
            return self.fields

        def initialize(self, context: vercor.SetupContext) -> None:
            events.append("user-initialize")
            self.fields["temperature"] = 281.0
            assert context.run_order == ("MODEL",)

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = context, payload
            return {"temperature": fields["temperature"]}

    component = PlainLifecycleComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )

    state = coupler.initial_state()

    assert_allclose_compact(
        state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 282.0),
    )
    assert events == [
        "user-initialize",
        "lifecycle-initialize",
        "prefill",
        "create-payload",
        "validate",
    ]
    assert hook_owners == [component, component, component, component]


@pytest.mark.fast_always
def test_coupler_spec_builds_coupler_from_plain_recipe() -> None:
    from vercor.coupling import CouplerSpec

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
        runtime=RuntimeOptions(topology=None),
    )

    coupler = recipe.build(Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))

    assert isinstance(coupler, Coupler)
    assert coupler.runtime.topology is None
    assert_allclose_compact(
        coupler.run().component("DST").field("flux"),
        jnp.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_runtime_options_accept_custom_execution_backend() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            state: vercor.RunState,
            *,
            context: vercor.ExecutionContext,
            driver: vercor.RuntimeDriver,
        ) -> vercor.RunState:
            _ = driver
            self.calls += 1
            assert context.run_order == ("MODEL",)
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
        runtime=RuntimeOptions(topology=None, execution=backend),
    )

    final_state = coupler.run()

    assert backend.calls == 1
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(grid.shape, 301.0),
    )
