from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime
import importlib
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

import vercor
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor import Clock, Coupler, Exchange
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    PrefillContext,
    PrefillResult,
    SetupContext,
    SetupResult,
    StepContext,
    ValidationContext,
)
from vercor.exceptions import CouplerError
from vercor.runtime import ExecutionContext, RuntimeDriver, RuntimeOptions
from vercor.state import ComponentState, RunState
from vercor.topology import ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext


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


class _RecordingTopologyPolicy:
    def __init__(
        self,
        *,
        applies: bool,
        patch: ExchangeTopologyPatch | None = None,
    ) -> None:
        self._applies = applies
        self._patch = ExchangeTopologyPatch() if patch is None else patch
        self.events: list[tuple[str, TopologyContext]] = []

    def applies(self, context: TopologyContext) -> bool:
        self.events.append(("applies", context))
        return self._applies

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        self.events.append(("build", context))
        return self._patch


def _topology_policy_coupler(policy: _RecordingTopologyPolicy) -> Coupler:
    grid = make_test_grid(name="topology-policy")
    source = DataComponent(
        "SRC",
        grid,
        {"custom_flux": 1.0},
    )
    target = DataComponent(
        "DST",
        grid,
        {"custom_flux": 0.0},
        spec=ComponentSpec(inputs=("custom_flux",)),
    )
    return Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("custom_flux",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=policy),
    )


@pytest.mark.fast_always
def test_custom_topology_policy_uses_applies_then_build_and_patches_maps() -> None:
    key = ("SRC", "DST", "bilinear")
    skipped = _RecordingTopologyPolicy(applies=False)
    skipped_coupler = _topology_policy_coupler(skipped)

    skipped_coupler.initial_state()

    assert [event for event, _ in skipped.events] == ["applies"]
    assert skipped_coupler._prepared is not None
    assert_allclose_compact(
        skipped_coupler._prepared.topology_maps.fractional_masks[key],
        jnp.ones((2, 2)),
    )

    applied = _RecordingTopologyPolicy(
        applies=True,
        patch=ExchangeTopologyPatch(
            fractional_masks={key: jnp.full((2, 2), 0.25)},
        ),
    )
    applied_coupler = _topology_policy_coupler(applied)

    applied_coupler.initial_state()

    assert [event for event, _ in applied.events] == ["applies", "build"]
    assert applied.events[0][1] is applied.events[1][1]
    assert applied_coupler._prepared is not None
    assert_allclose_compact(
        applied_coupler._prepared.topology_maps.fractional_masks[key],
        jnp.full((2, 2), 0.25),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("patch", "message"),
    (
        (
            ExchangeTopologyPatch(
                binary_masks={("UNKNOWN", "DST", "bilinear"): jnp.ones((2, 2))}
            ),
            "UNKNOWN.*configured topology key",
        ),
        (
            ExchangeTopologyPatch(
                fractional_masks={("SRC", "DST", "bilinear"): jnp.ones((1, 2))}
            ),
            r"\('SRC', 'DST', 'bilinear'\).*shape \(1, 2\).*expected \(2, 2\)",
        ),
    ),
)
def test_topology_policy_patch_rejects_unknown_keys_and_wrong_shapes(
    patch: ExchangeTopologyPatch,
    message: str,
) -> None:
    coupler = _topology_policy_coupler(
        _RecordingTopologyPolicy(applies=True, patch=patch)
    )

    with pytest.raises(CouplerError, match=message):
        coupler.initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("mask_kind", "value", "message"),
    (
        pytest.param(
            "binary",
            np.full((2, 2), "land"),
            "binary.*concrete numeric or bool",
            id="binary-nonnumeric",
        ),
        pytest.param(
            "fractional",
            np.full((2, 2), object(), dtype=object),
            "fractional.*concrete numeric or bool",
            id="fractional-nonconcrete-object",
        ),
        pytest.param(
            "binary",
            np.asarray([[0.0, 1.0], [np.nan, 0.0]]),
            "binary.*finite",
            id="binary-nonfinite",
        ),
        pytest.param(
            "fractional",
            np.asarray([[0.0, 1.0], [np.inf, 0.5]]),
            "fractional.*finite",
            id="fractional-nonfinite",
        ),
        pytest.param(
            "binary",
            np.asarray([[0.0, 1.0], [0.5, 0.0]]),
            r"binary.*\{0, 1\}",
            id="binary-nonbinary",
        ),
        pytest.param(
            "fractional",
            np.asarray([[0.0, 1.0], [-0.01, 0.5]]),
            r"fractional.*\[0, 1\]",
            id="fractional-below-range",
        ),
        pytest.param(
            "fractional",
            np.asarray([[0.0, 1.01], [0.25, 0.5]]),
            r"fractional.*\[0, 1\]",
            id="fractional-above-range",
        ),
    ),
)
def test_topology_policy_patch_rejects_invalid_mask_values(
    mask_kind: str,
    value: Any,
    message: str,
) -> None:
    key = ("SRC", "DST", "bilinear")
    patch = ExchangeTopologyPatch(
        binary_masks={key: value} if mask_kind == "binary" else {},
        fractional_masks={key: value} if mask_kind == "fractional" else {},
    )
    coupler = _topology_policy_coupler(
        _RecordingTopologyPolicy(applies=True, patch=patch)
    )

    with pytest.raises(CouplerError, match=message):
        coupler.initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "binary_mask",
    (
        pytest.param(
            np.asarray([[False, True], [True, False]]),
            id="bool",
        ),
        pytest.param(
            np.asarray([[0, 1], [1, 0]], dtype=np.int32),
            id="zero-one",
        ),
    ),
)
def test_topology_policy_patch_accepts_valid_binary_and_fractional_masks(
    binary_mask: np.ndarray,
) -> None:
    key = ("SRC", "DST", "bilinear")
    fractional_mask = np.asarray([[0.0, 0.25], [0.75, 1.0]])
    coupler = _topology_policy_coupler(
        _RecordingTopologyPolicy(
            applies=True,
            patch=ExchangeTopologyPatch(
                binary_masks={key: binary_mask},
                fractional_masks={key: fractional_mask},
            ),
        )
    )

    coupler.initial_state()

    assert coupler._prepared is not None
    assert_allclose_compact(
        coupler._prepared.topology_maps.binary_masks[key],
        binary_mask,
    )
    assert_allclose_compact(
        coupler._prepared.topology_maps.fractional_masks[key],
        fractional_mask,
    )


@pytest.mark.fast_always
def test_component_spec_freezes_mapping_inputs_and_exposes_execution_policy() -> None:
    initial_fields: dict[str, object] = {"temperature": 280.0}
    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("heat_flux",),
        initial_fields=initial_fields,
        execution="host",
    )
    initial_fields["temperature"] = 999.0

    assert spec.inputs == ("temperature",)
    assert spec.initial_fields["temperature"] == 280.0
    assert spec.execution == "host"
    assert not hasattr(spec, "import_policy")
    with pytest.raises(TypeError):
        spec.initial_fields["temperature"] = 281.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.execution = "jax"  # type: ignore[misc]

    grid = make_test_grid(name="execution-precedence")
    capable_host = CallableComponent(
        "CAPABLE_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="host"),
    )
    explicit_jax = CallableComponent(
        "ENFORCED_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="jax"),
    )

    assert capable_host.spec.execution == "host"
    assert explicit_jax.spec.execution == "jax"


@pytest.mark.fast_always
def test_structural_component_like_runs_without_private_component_internals() -> None:
    class PlainComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-component")

            def setup(owner: object, context: SetupContext) -> None:
                assert owner is self
                assert tuple(context.run_order) == ("MODEL",)

            self.spec = ComponentSpec(
                inputs=("heat_flux",),
                outputs=("temperature",),
                initial_fields={"temperature": 280.0, "heat_flux": 1.5},
                lifecycle=LifecycleHooks(setup=setup),
            )

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
        ({"step": None}, "step.*callable"),
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
        step=lambda fields, context, payload=None: {},
    )
    for attribute, value in override.items():
        setattr(candidate, attribute, value)

    with pytest.raises(CouplerError, match=message):
        Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(candidate,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(topology=None),
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("name", "name.*non-empty string"),
        ("grid", "grid.*RectilinearGrid"),
        ("spec", "spec.*ComponentSpec"),
        ("step", "step.*callable"),
    ),
)
def test_vercor_component_validation_uses_the_structural_contract_path(
    mutation: str,
    message: str,
) -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="invalid-vercor-component"),
        lambda fields: fields,
    )
    if mutation == "name":
        component.name = "   "
    elif mutation == "grid":
        component.grid = cast(Any, object())
    elif mutation == "spec":
        component.spec = cast(Any, object())
    else:
        setattr(component, mutation, None)

    with pytest.raises(CouplerError, match=message):
        Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(component,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(topology=None),
        )


@pytest.mark.fast_always
def test_component_binding_is_stable_when_setup_mutates_original_owner() -> None:
    class InvalidatingComponent:
        def __init__(self, name: str, grid: Any) -> None:
            self.name = name
            self.grid = grid

            def setup(owner: object, context: SetupContext) -> None:
                _ = context
                owner.name = "CHANGED"  # type: ignore[attr-defined]

            self.spec = ComponentSpec(lifecycle=LifecycleHooks(setup=setup))

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = context, payload
            return fields

    component = InvalidatingComponent(
        "MODEL",
        make_test_grid(name="initialize-contract-mutation"),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )

    state = coupler.initial_state()

    assert component.name == "CHANGED"
    assert state.component_names == ("MODEL",)
    assert coupler._prepared is not None
    assert coupler._prepared.components["MODEL"].name == "MODEL"


@pytest.mark.fast_always
def test_structural_lifecycle_hooks_receive_original_object_in_order() -> None:
    events: list[str] = []
    hook_owners: list[object] = []

    def setup_hook(owner: object, context: SetupContext) -> SetupResult:
        hook_owners.append(owner)
        events.append("setup")
        assert context.run_order == ("MODEL",)
        return SetupResult(
            fields={"temperature": 282.0},
            payload={"owner": owner.name},  # type: ignore[attr-defined]
        )

    def prefill_hook(
        owner: object,
        context: PrefillContext,
    ) -> PrefillResult:
        hook_owners.append(owner)
        events.append("prefill")
        assert float(jnp.mean(context.fields["temperature"])) == 282.0
        return PrefillResult()

    def validate_hook(
        owner: object,
        context: ValidationContext,
    ) -> None:
        hook_owners.append(owner)
        events.append("validate")
        assert isinstance(context.state, ComponentState)
        assert context.payload == {"owner": "MODEL"}

    class PlainLifecycleComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-lifecycle")
            self.spec = ComponentSpec(
                outputs=("temperature",),
                initial_fields={"temperature": 280.0},
                lifecycle=LifecycleHooks(
                    setup=setup_hook,
                    prefill=prefill_hook,
                    validate=validate_hook,
                ),
            )

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
        "setup",
        "prefill",
        "validate",
    ]
    assert hook_owners == [component, component, component]


@pytest.mark.fast_always
def test_constructor_builds_coupler_from_plain_recipe() -> None:
    grid = make_test_grid(name="coupler-spec")
    forcing = DataComponent("SRC", grid, {"flux": 2.0})
    model = CallableComponent(
        "DST",
        grid,
        lambda fields: {"flux": fields["flux"] + 1.0},
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("flux",),
            initial_fields={"flux": 0.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(forcing, model),
        exchanges=(Exchange("SRC", "DST", ("flux",)),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=None),
    )

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
            state: RunState,
            *,
            context: ExecutionContext,
            driver: RuntimeDriver,
        ) -> RunState:
            _ = driver
            self.calls += 1
            assert context.run_order == ("MODEL",)
            return state.replace_fields(
                "MODEL",
                {"temperature": jnp.full(grid.shape, 301.0)},
            )

    grid = make_test_grid(name="custom-backend")
    backend = RecordingBackend()
    component = DataComponent(
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
