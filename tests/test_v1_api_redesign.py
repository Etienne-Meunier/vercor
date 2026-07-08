from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import vercor
import vercor.output
from tests._coverage_support import make_test_grid
from vercor import Clock, Coupler, DataComponent, RectilinearGrid
from vercor.output import OutputVariable
from vercor.regridding import bilinear


@pytest.mark.fast_always
def test_v1_grid_constructors_live_on_rectilinear_grid_class() -> None:
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
    assert "grid_from_coordinates" not in vercor.__all__
    assert "uniform_rectilinear_grid" not in vercor.__all__
    assert not hasattr(vercor, "grid_from_coordinates")
    assert not hasattr(vercor, "uniform_rectilinear_grid")


@pytest.mark.fast_always
def test_v1_run_state_exposes_component_state_view_not_runtime_state() -> None:
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
    assert view.field("temperature", store="data").shape == component.grid.shape
    assert tuple(view.fields()) == ("temperature",)
    assert isinstance(state.components()["ATM"], vercor.ComponentState)
    assert not hasattr(state, "get_component_state")
    assert not hasattr(view, "data")
    assert not hasattr(vercor, "ComponentView")
    assert "ComponentState" in vercor.__all__


@pytest.mark.fast_always
def test_v1_coupler_uses_initial_state_name() -> None:
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
    assert not hasattr(Coupler, "state")
    assert isinstance(
        coupler.view(coupler.initial_state(), "ATM"), vercor.ComponentState
    )


@pytest.mark.fast_always
def test_v1_component_setup_storage_is_not_publicly_mutable() -> None:
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
def test_v1_regridder_public_grid_name_is_target_only() -> None:
    grid = make_test_grid(name="v1-regridder")
    regridder = bilinear(grid, grid)
    scalar = jnp.ones(grid.shape)

    assert regridder.target_grid is grid
    assert regridder.regrid(scalar) is scalar
    assert not hasattr(regridder, "destination_grid")
    assert not callable(regridder)


@pytest.mark.fast_always
def test_v1_output_public_api_is_spec_not_mutable_adapter() -> None:
    calls: list[tuple[object, ...]] = []

    def writer(
        component_state: object, output_dir: object, time: object, logger: object
    ) -> None:
        calls.append((component_state, output_dir, time, logger))

    assert hasattr(vercor.output, "ComponentOutput")
    output = vercor.output.ComponentOutput(snapshot_writer=writer)

    assert output.snapshot_writer is writer
    assert OutputVariable(dims=("time",), values=jnp.asarray([1.0])).dims == ("time",)
    assert "ComponentOutput" in vercor.output.__all__
    assert "ComponentOutputAdapter" not in vercor.output.__all__
    assert "register_component_snapshot_writer" not in vercor.output.__all__
    assert not hasattr(vercor.output, "ComponentOutputAdapter")
    assert not hasattr(vercor.output, "register_component_snapshot_writer")


@pytest.mark.fast_always
def test_v1_private_grid_and_exchange_shims_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match="vercor._grid"):
        importlib.import_module("vercor._grid")
    with pytest.raises(ModuleNotFoundError, match="vercor._exchange"):
        importlib.import_module("vercor._exchange")


@pytest.mark.fast_always
def test_v1_active_docs_do_not_advertise_removed_transition_apis() -> None:
    active_docs = (
        Path("DESIGN.md").read_text(encoding="utf-8")
        + "\n"
        + Path("DEPENDENCIES.md").read_text(encoding="utf-8")
    )
    stale_markers = (
        "ComponentView",
        "`Coupler` exposes `state()`",
        "`run()`, `state()`",
        "callable scalar/vector behavior for staged compatibility",
        "`Coupler.initialize()`",
        "`Component.setup_metadata`",
        "`Component.data`",
    )

    for marker in stale_markers:
        assert marker not in active_docs


@pytest.mark.fast_always
def test_v1_run_state_remains_a_jax_pytree() -> None:
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
