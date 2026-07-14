from __future__ import annotations

from datetime import datetime
import importlib
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import jax
import jax.numpy as jnp
import pytest

import vercor
import vercor.output
import vercor.setups
from tests._coverage_support import make_test_grid
from vercor import Clock, Coupler, RectilinearGrid
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    PrefillContext,
    PrefillResult,
    SetupResult,
    StepResult,
    ValidationContext,
)
from vercor.dtypes import DTypePolicy
from vercor.output import (
    OutputConfig,
    OutputVariable,
    PeriodOutput,
    SnapshotContext,
)
from vercor.regridding import bilinear
from vercor.state import ComponentState, RunState


@pytest.mark.fast_always
def test_grid_constructors_live_on_rectilinear_grid_class() -> None:
    grid = RectilinearGrid.uniform(
        "class-grid",
        nlon=2,
        nlat=2,
        longitude=(0.0, 90.0),
        latitude=(-45.0, 45.0),
    )
    explicit = RectilinearGrid.from_coordinates(
        "explicit-grid",
        longitude=jnp.asarray([0.0, 90.0]),
        latitude=jnp.asarray([-45.0, 45.0]),
    )

    assert grid.shape == (2, 2)
    assert explicit.shape == (2, 2)
    grid_type = getattr(vercor, "RectilinearGrid")
    with pytest.raises(TypeError):
        grid_type(
            "positional-grid",
            jnp.asarray([0.0, 90.0]),
            jnp.asarray([-45.0, 45.0]),
        )
    assert "grid_from_coordinates" not in vercor.__all__
    assert "uniform_rectilinear_grid" not in vercor.__all__
    assert not hasattr(vercor, "grid_from_coordinates")
    assert not hasattr(vercor, "uniform_rectilinear_grid")


@pytest.mark.fast_always
def test_run_state_exposes_component_state_view_not_runtime_state() -> None:
    component = DataComponent(
        "ATM",
        make_test_grid(name="v1-state"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    state = coupler.initial_state()
    view = state.component("ATM")

    assert isinstance(view, ComponentState)
    assert view.field("temperature").shape == component.grid.shape
    assert view.field("temperature", scope="state").shape == component.grid.shape
    assert tuple(view.fields()) == ("temperature",)
    assert isinstance(state.components()["ATM"], ComponentState)
    assert not hasattr(state, "get_component_state")
    assert not hasattr(view, "data")
    assert not hasattr(vercor, "ComponentView")
    assert "ComponentState" not in vercor.__all__
    assert ComponentState.__module__ == "vercor.state"


@pytest.mark.fast_always
def test_coupler_uses_initial_state_name() -> None:
    component = DataComponent(
        "ATM",
        make_test_grid(name="v1-coupler"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    assert isinstance(coupler.initial_state(), RunState)
    assert isinstance(coupler.initial_state(prefill_missing=True), RunState)
    with pytest.raises(TypeError):
        coupler.initial_state(prefill=True)  # type: ignore[call-arg]
    assert not hasattr(Coupler, "state")
    assert isinstance(coupler.initial_state().component("ATM"), ComponentState)


@pytest.mark.fast_always
def test_component_setup_storage_is_not_publicly_mutable() -> None:
    component = DataComponent(
        "ATM",
        make_test_grid(name="v1-component"),
        fields={"temperature": 280.0},
    )

    assert tuple(component.spec.initial_fields) == ("temperature",)
    with pytest.raises(TypeError):
        component.spec.initial_fields["humidity"] = 0.5  # type: ignore[index]
    assert not hasattr(component, "data")
    assert not hasattr(component, "setup_metadata")
    assert not hasattr(component, "seed_field")


@pytest.mark.fast_always
def test_regridder_public_grid_name_is_target_only() -> None:
    grid = make_test_grid(name="v1-regridder")
    regridder = bilinear(grid, grid)
    scalar = jnp.ones(grid.shape)

    assert regridder.target_grid is grid
    assert regridder.regrid(scalar) is scalar
    assert not hasattr(regridder, "destination_grid")
    assert not callable(regridder)


@pytest.mark.fast_always
def test_output_public_api_is_spec_not_mutable_adapter() -> None:
    calls: list[tuple[object, ...]] = []

    def writer(context: SnapshotContext) -> None:
        calls.append((context.state, context.output_path, context.time, context.logger))

    assert hasattr(vercor.output, "OutputConfig")
    output = vercor.output.OutputConfig(snapshot_writer=writer)
    period = vercor.output.PeriodOutput()

    assert output.snapshot_writer is writer
    assert output.period is None
    assert period.frequency == "step"
    assert OutputVariable(dims=("time",), values=jnp.asarray([1.0])).dims == ("time",)
    assert vercor.output.OutputConfig.__module__ == "vercor.output"
    assert vercor.output.PeriodOutput.__module__ == "vercor.output"
    assert vercor.output.__all__ == [
        "OutputConfig",
        "OutputFrequency",
        "OutputVariable",
        "PeriodOutput",
        "SnapshotContext",
        "SnapshotWriter",
    ]
    for removed_name in (
        "ComponentOutput",
        "ComponentOutputAdapter",
        "register_component_snapshot_writer",
    ):
        assert removed_name not in vercor.output.__all__
        assert not hasattr(vercor.output, removed_name)
    with pytest.raises(ModuleNotFoundError, match="vercor.config"):
        importlib.import_module("vercor.config")
    with pytest.raises(ModuleNotFoundError, match="vercor.setup_config"):
        importlib.import_module("vercor.setup_config")


@pytest.mark.fast_always
def test_component_constructors_accept_component_spec_only() -> None:
    grid = make_test_grid(name="v1-spec-only")

    component = CallableComponent(
        "OCN",
        grid,
        lambda fields: {"sea_surface_temperature": fields["temperature"]},
        spec=ComponentSpec(
            inputs=("temperature",),
            outputs=("sea_surface_temperature",),
            initial_fields={
                "temperature": 280.0,
                "sea_surface_temperature": 280.0,
            },
        ),
    )
    forcing = DataComponent(
        "ATM",
        grid,
        {"temperature": 280.0},
        spec=ComponentSpec(outputs=("temperature",)),
    )

    assert component.spec.inputs == ("temperature",)
    assert component.spec.outputs == ("sea_surface_temperature",)
    assert forcing.spec.outputs == ("temperature",)

    with pytest.raises(TypeError, match="inputs"):
        CallableComponent(  # type: ignore[call-arg]
            "OLD",
            grid,
            lambda fields: {},
            inputs=("temperature",),
        )
    with pytest.raises(TypeError, match="outputs"):
        DataComponent(  # type: ignore[call-arg]
            "OLD",
            grid,
            fields={"temperature": 280.0},
            outputs=("temperature",),
        )


@pytest.mark.fast_always
def test_component_step_return_contract_is_public_in_its_owner_package() -> None:
    components_module = importlib.import_module("vercor.components")
    contracts_module = importlib.import_module("vercor.components.contracts")

    assert not hasattr(components_module, "ComponentStepReturn")
    assert not hasattr(contracts_module, "ComponentStepReturn")
    assert "ComponentStepReturn" not in vercor.__all__
    assert not hasattr(vercor, "ComponentStepReturn")

    for step_method in (
        components_module.Component.step,
        components_module.CallableComponent.step,
        components_module.DataComponent.step,
    ):
        return_annotation = str(signature(step_method).return_annotation)
        assert "Mapping" in return_annotation
        assert "StepResult" in return_annotation

    step_annotation = (
        signature(components_module.CallableComponent).parameters["step"].annotation
    )
    assert "Mapping" in str(step_annotation)
    assert "StepResult" in str(step_annotation)


@pytest.mark.fast_always
def test_public_component_contracts_do_not_expose_runtime_implementation_types() -> (
    None
):
    components_module = importlib.import_module("vercor.components")
    contracts_module = importlib.import_module("vercor.components.contracts")

    public_signatures = (
        str(signature(components_module.Component.step)),
        str(signature(components_module.CallableComponent.step)),
        str(signature(components_module.DataComponent.step)),
        str(signature(contracts_module.ValidationContext)),
        str(signature(components_module.ComponentSpec)),
    )
    for public_signature in public_signatures:
        for private_type in (
            "_ComponentStepReturn",
            "ExchangeContract",
            "ComponentRuntimeState",
            "FieldStore",
        ):
            assert private_type not in public_signature

    assert (
        signature(contracts_module.ValidationContext).parameters["state"].annotation
        == "ComponentState"
    )
    assert get_type_hints(components_module.ComponentSpec)["output"] is OutputConfig


@pytest.mark.fast_always
def test_validation_context_runtime_type_hints_resolve_public_state() -> None:
    contracts_module = importlib.import_module("vercor.components.contracts")

    type_hints = get_type_hints(contracts_module.ValidationContext)

    assert type_hints["state"] is ComponentState


@pytest.mark.fast_always
def test_public_component_step_type_hints_resolve_at_runtime() -> None:
    components_module = importlib.import_module("vercor.components")

    for component_type in (
        components_module.Component,
        components_module.CallableComponent,
        components_module.DataComponent,
    ):
        type_hints = get_type_hints(component_type.step)
        assert type_hints["context"] is components_module.StepContext
        assert "StepResult" in str(type_hints["return"])


@pytest.mark.fast_always
def test_data_component_rejects_active_step_factory() -> None:
    assert not hasattr(DataComponent, "from_step")
    assert not hasattr(DataComponent, "from_fields")
    assert CallableComponent is not DataComponent


@pytest.mark.fast_always
def test_typing_aliases_have_explicit_owner_modules_without_root_aliases() -> None:
    dtypes_module = importlib.import_module("vercor.dtypes")
    logging_module = importlib.import_module("vercor.jax_logging")
    types_module = importlib.import_module("vercor.types")

    assert types_module.__all__ == ["RuntimeArray"]
    assert "PrecisionPolicy" in dtypes_module.__all__
    assert "LoggerLike" in logging_module.__all__
    for name in ("RuntimeArray", "PrecisionPolicy", "LoggerLike"):
        assert name not in vercor.__all__
        assert not hasattr(vercor, name)


@pytest.mark.fast_always
def test_component_spec_replaces_field_hooks_and_output_specs() -> None:
    grid = make_test_grid(name="component-spec-redesign")
    events: list[str] = []

    def prefill(
        component: Component,
        context: PrefillContext,
    ) -> PrefillResult:
        events.append(f"prefill:{component.name}:{context.receives}:{context.sends}")
        return PrefillResult(fields={"humidity": jnp.full(component.grid.shape, 0.5)})

    def writer(context: SnapshotContext) -> None:
        events.append(f"snapshot:{context.component.name}:{context.output_path.name}")

    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("sea_surface_temperature",),
        initial_fields={"temperature": 280.0, "sea_surface_temperature": 281.0},
        lifecycle=LifecycleHooks(prefill=prefill),
        output=OutputConfig(snapshot_writer=writer),
    )
    component = CallableComponent(
        "OCN",
        grid,
        lambda fields: {"sea_surface_temperature": fields["temperature"]},
        spec=spec,
    )

    assert component.spec is spec
    assert component.spec.output.snapshot_writer is writer
    assert spec.inputs == ("temperature",)
    assert spec.outputs == ("sea_surface_temperature",)
    assert spec.lifecycle.prefill is prefill
    assert "ComponentSpec" not in vercor.__all__
    assert "LifecycleHooks" not in vercor.__all__
    assert "OutputConfig" not in vercor.__all__
    assert ComponentSpec.__module__ == "vercor.components.contracts"
    assert OutputConfig.__module__ == "vercor.output"
    assert "FieldSpec" not in vercor.__all__
    assert "ComponentHooks" not in vercor.__all__
    assert "OutputSpec" not in vercor.__all__
    assert not hasattr(vercor, "FieldSpec")
    assert not hasattr(vercor, "ComponentHooks")
    assert not hasattr(vercor, "OutputSpec")
    assert not hasattr(component, "field_spec")

    with pytest.raises(TypeError, match="hooks"):
        ComponentSpec(hooks=LifecycleHooks())  # type: ignore[call-arg]


@pytest.mark.fast_always
def test_state_views_use_domain_scopes_not_runtime_store_names() -> None:
    component = DataComponent(
        "ATM",
        make_test_grid(name="scope-state"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    state = coupler.initial_state()
    view = state.component("ATM")
    updated = state.replace_fields(
        "ATM", {"temperature": jnp.full(component.grid.shape, 281.0)}
    )

    assert view.field("temperature", scope="state").shape == component.grid.shape
    assert view.field("temperature", scope="any").shape == component.grid.shape
    assert tuple(view.fields(scope="state")) == ("temperature",)
    assert (
        float(jnp.mean(updated.component("ATM").field("temperature", scope="state")))
        == 281.0
    )
    with pytest.raises(TypeError, match="store"):
        view.field("temperature", store="data")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="scope"):
        state.replace_fields(
            "ATM",
            {"temperature": jnp.full(component.grid.shape, 282.0)},
            scope="state",  # type: ignore[call-arg]
        )
    assert not hasattr(state, "with_fields")


@pytest.mark.fast_always
def test_output_and_setup_configs_use_final_names() -> None:
    period = PeriodOutput(frequency="month", variables=("temp",))
    output = OutputConfig(period=period)
    spinup = vercor.setups.Spinup(enabled=True)
    veros = vercor.setups.VerosConfig(spinup=spinup, output=output)
    jax_gcm = vercor.setups.JAXGCMConfig(spinup=spinup, output=output)
    camulator = vercor.setups.CAMulatorConfig(
        config_path="config.yml", spinup=spinup, output=output
    )

    assert PeriodOutput().frequency == "step"
    assert OutputConfig().period is None
    assert output.period is period
    assert spinup.duration.days == 2
    assert veros.output.period is period
    assert period.frequency == "month"
    assert jax_gcm.jitted is True
    assert camulator.device == "cuda"
    for removed_name in ("Spinup", "SpinupConfig", "PeriodOutputConfig"):
        assert removed_name not in vercor.__all__
        assert not hasattr(vercor, removed_name)
    for setup_name in ("Spinup", "VerosConfig", "JAXGCMConfig", "CAMulatorConfig"):
        assert setup_name in vercor.setups.__all__
        assert hasattr(vercor.setups, setup_name)


@pytest.mark.fast_always
def test_snapshot_writer_receives_public_context(tmp_path: Path) -> None:
    grid = make_test_grid(name="v1-snapshot-context")
    contexts: list[SnapshotContext] = []

    def writer(context: SnapshotContext) -> None:
        contexts.append(context)

    component = CallableComponent(
        "ATM",
        grid,
        lambda fields, context, payload: StepResult(
            fields={"temperature": fields["temperature"]},
            payload=payload,
        ),
        spec=ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
            lifecycle=LifecycleHooks(
                setup=lambda owner, context: SetupResult(payload=jnp.asarray(7.0))
            ),
            output=OutputConfig(snapshot_writer=writer),
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    final_state = coupler.run()
    coupler.write_outputs(final_state, output_dir=tmp_path)

    assert len(contexts) == 1
    assert contexts[0].component.name == component.name
    assert contexts[0].component.grid.name == component.grid.name
    assert contexts[0].component.grid.shape == component.grid.shape
    assert contexts[0].component.spec is component.spec
    assert isinstance(contexts[0].state, ComponentState)
    assert contexts[0].payload is not None
    assert float(contexts[0].payload) == 7.0
    assert contexts[0].output_path == tmp_path / "atm.snapshot.nc"
    assert not contexts[0].state.__class__.__name__.startswith("Runtime")


@pytest.mark.fast_always
def test_lifecycle_hooks_use_typed_contexts_and_results() -> None:
    grid = make_test_grid(name="v1-hook-context")
    events: list[str] = []

    def prefill(
        component: Component,
        context: PrefillContext,
    ) -> PrefillResult:
        events.append(f"prefill:{component.name}:{context.receives}:{context.sends}")
        return PrefillResult(fields={"humidity": jnp.full(component.grid.shape, 0.5)})

    def validate(
        component: Component,
        context: ValidationContext,
    ) -> None:
        events.append(
            f"validate:{component.name}:{'humidity' in context.state.fields()}"
        )

    component = DataComponent(
        "OBS",
        grid,
        spec=ComponentSpec(
            outputs=("humidity",),
            lifecycle=LifecycleHooks(prefill=prefill, validate=validate),
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("OBS",),
    )

    state = coupler.initial_state()

    assert state.component("OBS").field("humidity").shape == grid.shape
    assert events == ["prefill:OBS:():()", "validate:OBS:True"]


@pytest.mark.fast_always
def test_setup_and_dtype_config_objects_are_public() -> None:
    output = PeriodOutput(frequency="month", variables=("temp", "salt"))
    spinup = vercor.setups.Spinup(enabled=True)

    assert output.frequency == "month"
    assert output.variables == ("temp", "salt")
    assert spinup.duration.days == 2
    assert DTypePolicy(enable_x64=True).enable_x64
    assert PeriodOutput is vercor.output.PeriodOutput
    assert "PeriodOutput" not in vercor.__all__
    assert "Spinup" not in vercor.__all__
    assert "Spinup" in vercor.setups.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__
    assert "RuntimeOptions" in vercor.__all__
    assert "DTypePolicy" not in vercor.__all__
    assert "OutputConfig" not in vercor.__all__
    assert "setups" not in vercor.__all__
    assert "ComponentOutput" not in vercor.__all__


@pytest.mark.fast_always
def test_private_grid_and_exchange_modules_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="vercor._grid"):
        importlib.import_module("vercor._grid")
    with pytest.raises(ModuleNotFoundError, match="vercor._exchange"):
        importlib.import_module("vercor._exchange")


@pytest.mark.fast_always
def test_active_docs_do_not_advertise_removed_transition_apis() -> None:
    active_docs = (
        Path("DESIGN.md").read_text(encoding="utf-8")
        + "\n"
        + Path("DEPENDENCIES.md").read_text(encoding="utf-8")
    )
    stale_markers = (
        "ComponentView",
        "`Coupler` exposes `state()`",
        "`run()`, `state()`",
        "callable scalar/vector behavior for sta" + "ged compat" + "ibility",
        "`Coupler.initialize()`",
        "`Component.setup_metadata`",
        "`Component.data`",
    )

    for marker in stale_markers:
        assert marker not in active_docs


@pytest.mark.fast_always
def test_run_state_remains_a_jax_pytree() -> None:
    component = DataComponent(
        "ATM",
        make_test_grid(name="v1-pytree"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )
    state = coupler.initial_state()

    leaves = jax.tree_util.tree_leaves(state)

    assert leaves
    assert state.component("ATM").field("temperature").shape == component.grid.shape
