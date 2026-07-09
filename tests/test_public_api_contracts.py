from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import vercor
import vercor.output
import vercor.setup_config
from tests._coverage_support import make_test_grid
from vercor import Clock, Coupler, DataComponent, RectilinearGrid
from vercor.output import OutputVariable
from vercor.regridding import bilinear


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
    component = DataComponent.from_fields(
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

    assert isinstance(view, vercor.ComponentState)
    assert view.field("temperature").shape == component.grid.shape
    assert view.field("temperature", scope="state").shape == component.grid.shape
    assert tuple(view.fields()) == ("temperature",)
    assert isinstance(state.components()["ATM"], vercor.ComponentState)
    assert not hasattr(state, "get_component_state")
    assert not hasattr(view, "data")
    assert not hasattr(vercor, "ComponentView")
    assert "ComponentState" in vercor.__all__


@pytest.mark.fast_always
def test_coupler_uses_initial_state_name() -> None:
    component = DataComponent.from_fields(
        "ATM",
        make_test_grid(name="v1-coupler"),
        fields={"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    assert isinstance(coupler.initial_state(), vercor.RunState)
    assert isinstance(coupler.initial_state(prefill_missing=True), vercor.RunState)
    with pytest.raises(TypeError):
        coupler.initial_state(prefill=True)  # type: ignore[call-arg]
    assert not hasattr(Coupler, "state")
    assert isinstance(coupler.initial_state().component("ATM"), vercor.ComponentState)


@pytest.mark.fast_always
def test_component_setup_storage_is_not_publicly_mutable() -> None:
    component = DataComponent.from_fields(
        "ATM",
        make_test_grid(name="v1-component"),
        fields={"temperature": 280.0},
    )

    component.seed_field("humidity", 0.5)

    assert component.field_names == ("temperature", "humidity")
    assert not hasattr(component, "data")
    assert not hasattr(component, "setup_metadata")


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

    def writer(context: vercor.SnapshotContext) -> None:
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
    assert not hasattr(vercor.setup_config, "OutputConfig")


@pytest.mark.fast_always
def test_component_constructors_accept_component_spec_only() -> None:
    grid = make_test_grid(name="v1-spec-only")

    component = vercor.Component.from_step(
        "OCN",
        grid,
        lambda fields: {"sea_surface_temperature": fields["temperature"]},
        spec=vercor.ComponentSpec(
            inputs=("temperature",),
            outputs=("sea_surface_temperature",),
            defaults={"temperature": 280.0, "sea_surface_temperature": 280.0},
        ),
    )
    forcing = DataComponent.from_fields(
        "ATM",
        grid,
        {"temperature": 280.0},
        spec=vercor.ComponentSpec(outputs=("temperature",)),
    )

    assert component.spec.inputs == ("temperature",)
    assert component.spec.outputs == ("sea_surface_temperature",)
    assert forcing.spec.outputs == ("temperature",)

    with pytest.raises(TypeError, match="inputs"):
        vercor.Component.from_step(  # type: ignore[call-arg]
            "OLD",
            grid,
            lambda fields: {},
            inputs=("temperature",),
        )
    with pytest.raises(TypeError, match="outputs"):
        DataComponent.from_fields(  # type: ignore[call-arg]
            "OLD",
            grid,
            fields={"temperature": 280.0},
            outputs=("temperature",),
        )


@pytest.mark.fast_always
def test_component_spec_replaces_field_hooks_and_output_specs() -> None:
    grid = make_test_grid(name="component-spec-redesign")
    events: list[str] = []

    def prefill(
        component: vercor.Component,
        context: vercor.PrefillContext,
    ) -> vercor.PrefillResult:
        events.append(f"prefill:{component.name}:{context.receives}:{context.sends}")
        return vercor.PrefillResult(
            fields={"humidity": jnp.full(component.grid.shape, 0.5)}
        )

    def writer(context: vercor.SnapshotContext) -> None:
        events.append(f"snapshot:{context.component.name}:{context.output_path.name}")

    spec = vercor.ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("sea_surface_temperature",),
        defaults={"temperature": 280.0, "sea_surface_temperature": 281.0},
        lifecycle=vercor.LifecycleHooks(prefill=prefill),
        output=vercor.OutputConfig(snapshot_writer=writer),
    )
    component = vercor.Component.from_step(
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
    assert "ComponentSpec" in vercor.__all__
    assert "LifecycleHooks" in vercor.__all__
    assert "OutputConfig" in vercor.__all__
    assert "FieldSpec" not in vercor.__all__
    assert "ComponentHooks" not in vercor.__all__
    assert "OutputSpec" not in vercor.__all__
    assert not hasattr(vercor, "FieldSpec")
    assert not hasattr(vercor, "ComponentHooks")
    assert not hasattr(vercor, "OutputSpec")
    assert not hasattr(component, "field_spec")

    with pytest.raises(TypeError, match="hooks"):
        vercor.ComponentSpec(hooks=vercor.LifecycleHooks())  # type: ignore[call-arg]


@pytest.mark.fast_always
def test_state_views_use_domain_scopes_not_runtime_store_names() -> None:
    component = DataComponent.from_fields(
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
    period = vercor.PeriodOutput(frequency="month", variables=("temp",))
    output = vercor.OutputConfig(period=period)
    spinup = vercor.Spinup(enabled=True)
    veros = vercor.VerosConfig(spinup=spinup, output=output)
    jax_gcm = vercor.JAXGCMConfig(spinup=spinup, output=output)
    camulator = vercor.CAMulatorConfig(
        config_path="config.yml", spinup=spinup, output=output
    )

    assert vercor.PeriodOutput().frequency == "step"
    assert vercor.OutputConfig().period is None
    assert output.period is period
    assert spinup.duration.days == 2
    assert veros.output.period is period
    assert period.frequency == "month"
    assert jax_gcm.jitted is True
    assert camulator.device == "cuda"
    for removed_name in ("SpinupConfig", "PeriodOutputConfig"):
        assert removed_name not in vercor.__all__
        assert not hasattr(vercor, removed_name)


@pytest.mark.fast_always
def test_snapshot_writer_receives_public_context(tmp_path: Path) -> None:
    grid = make_test_grid(name="v1-snapshot-context")
    contexts: list[vercor.SnapshotContext] = []

    def writer(context: vercor.SnapshotContext) -> None:
        contexts.append(context)

    component = vercor.Component.from_step(
        "ATM",
        grid,
        lambda fields, context, payload: vercor.StepResult(
            fields={"temperature": fields["temperature"]},
            payload=payload,
        ),
        spec=vercor.ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            defaults={"temperature": 280.0},
            output=vercor.OutputConfig(snapshot_writer=writer),
        ),
        payload=jnp.asarray(7.0),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    final_state = coupler.run()
    coupler.write_outputs(final_state, output_dir=tmp_path)

    assert len(contexts) == 1
    assert contexts[0].component is component
    assert isinstance(contexts[0].state, vercor.ComponentState)
    assert contexts[0].payload is not None
    assert float(contexts[0].payload) == 7.0
    assert contexts[0].output_path == tmp_path / "atm.snapshot.nc"
    assert not contexts[0].state.__class__.__name__.startswith("Runtime")


@pytest.mark.fast_always
def test_lifecycle_hooks_use_typed_contexts_and_results() -> None:
    grid = make_test_grid(name="v1-hook-context")
    events: list[str] = []

    def prefill(
        component: vercor.Component,
        context: vercor.PrefillContext,
    ) -> vercor.PrefillResult:
        events.append(f"prefill:{component.name}:{context.receives}:{context.sends}")
        return vercor.PrefillResult(
            fields={"humidity": jnp.full(component.grid.shape, 0.5)}
        )

    def validate(
        component: vercor.Component,
        context: vercor.ValidationContext,
    ) -> None:
        events.append(
            f"validate:{component.name}:{'humidity' in context.state.fields()}"
        )

    component = DataComponent.from_fields(
        "OBS",
        grid,
        spec=vercor.ComponentSpec(
            lifecycle=vercor.LifecycleHooks(prefill=prefill, validate=validate),
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
    output = vercor.PeriodOutput(frequency="month", variables=("temp", "salt"))
    spinup = vercor.Spinup(enabled=True)

    assert output.frequency == "month"
    assert output.variables == ("temp", "salt")
    assert spinup.duration.days == 2
    assert vercor.DTypePolicy(enable_x64=True).enable_x64
    assert vercor.PeriodOutput is vercor.output.PeriodOutput
    assert "PeriodOutput" in vercor.__all__
    assert "Spinup" in vercor.__all__
    assert "SurfaceMaskPolicy" in vercor.__all__
    assert "DTypePolicy" in vercor.__all__
    assert "OutputConfig" in vercor.__all__
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
    component = DataComponent.from_fields(
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
