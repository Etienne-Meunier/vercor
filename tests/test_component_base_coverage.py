from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.components as components_module
import vercor.components.base as base_module
from tests._coverage_support import DummyComponent, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.runtime.contexts import RuntimeStepContext
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.forcing_data import ComponentForcingData
from vercor.output import write_runtime_component_view_to_netcdf
from vercor.run_sequence import RunSequence
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeComponentState,
    RuntimeFieldStore,
)
from vercor.runtime.components import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
    create_runtime_component_state,
    receive_runtime_fields,
    send_runtime_fields,
    validate_component_runtime_contract_fields,
)
from vercor.runtime.time import scalar_runtime_step_info
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings


class _RuntimeOnlyComponent(base_module.Component):
    def step_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        data = component_state.data.set(
            "temperature",
            component_state.data.get("temperature") + context.dt_seconds,
        )
        return component_state.with_data(data)


class _MissingSetupComponent(base_module.Component):
    def __init__(self) -> None:
        pass

    def step_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        _ = context
        return component_state


class _HostStepOnlyComponent(base_module.HostRuntimeComponent):
    def step_host_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        _ = context
        return component_state


@pytest.mark.fast_always
def test_active_component_requires_explicit_runtime_step() -> None:
    class MissingRuntimeStep(base_module.Component):
        pass

    with pytest.raises(TypeError, match="step_runtime_state"):
        MissingRuntimeStep(name="ATM", grid=make_test_grid())  # type: ignore[abstract]


@pytest.mark.fast_always
def test_host_runtime_component_requires_explicit_host_step() -> None:
    class MissingHostStep(base_module.HostRuntimeComponent):
        pass

    with pytest.raises(TypeError, match="step_host_runtime_state"):
        MissingHostStep(name="ATM", grid=make_test_grid())  # type: ignore[abstract]


@pytest.mark.fast_always
def test_data_component_uses_explicit_noop_runtime_step() -> None:
    class StaticForcingComponent(base_module.DataComponent):
        pass

    grid = make_test_grid(name="data")
    component = StaticForcingComponent(name="OCN", grid=grid)
    component.data["sea_surface_temperature"] = jnp.full(grid.shape, 280.0)
    contract = RuntimeComponentContract(exports=("sea_surface_temperature",))
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=60.0, settings=VercorSettings()),
    )

    assert stepped is state
    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.outgoing.get("sea_surface_temperature"),
        np.full(grid.shape, 280.0),
    )


@pytest.mark.fast_always
def test_component_setup_validation_reports_missing_required_attributes() -> None:
    component = _MissingSetupComponent()
    contract = RuntimeComponentContract(exports=("temperature",))

    with pytest.raises(
        ComponentError,
        match="missing required setup attribute.*name.*grid.*data.*settings",
    ):
        create_runtime_component_state(component, contract=contract)


@pytest.mark.fast_always
def test_host_component_rejects_scanned_runtime_with_clear_error() -> None:
    grid = make_test_grid(name="host")
    component = _HostStepOnlyComponent(name="ATM", grid=grid)
    component.data["temperature"] = jnp.ones(grid.shape)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = RunSequence(order=["ATM"])
    state = coupler.create_runtime_state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        coupler._run_scanned_runtime(state)


@pytest.mark.fast_always
def test_removed_component_api_stays_absent() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    assert not hasattr(components_module, "Shared")
    assert not hasattr(components_module, "TimedNamedArray")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")
    assert not hasattr(base_module, "Shared")
    assert not hasattr(base_module, "TimedNamedArray")
    assert not hasattr(base_module, "write_shared_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(base_module, "ComponentForcingData")
    assert not hasattr(components_module, "ComponentForcingData")
    assert not hasattr(components_module, "Atmosphere")
    assert not hasattr(components_module, "Ocean")
    assert not hasattr(components_module, "SeaIce")
    assert not hasattr(components_module, "Land")
    assert not hasattr(components_module, "ERA5Atmosphere")
    assert not hasattr(components_module, "ERA5Ocean")
    assert not hasattr(components_module, "ERAInterimOcean")
    assert not hasattr(components_module, "ERA5Land")
    assert not hasattr(components_module, "JCMLand")
    assert not hasattr(components_module, "JAXGCM")
    assert not hasattr(components_module, "VerosGCM")
    assert not hasattr(components_module, "CAMulatorGCM")
    assert not hasattr(components_module, "CAMulatorLand")
    assert not hasattr(components_module, "write_runtime_component_to_netcdf")
    assert not hasattr(components_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(component, "incoming_fields")
    assert not hasattr(component, "outgoing_fields")
    assert not hasattr(component, "commit_runtime_state")
    assert not hasattr(component, "merge_incoming_outgoing_fields")
    assert not hasattr(component, "get")
    assert not hasattr(component, "step")
    assert not hasattr(component, "to_runtime_component_state")
    assert not hasattr(component, "receive_runtime_fields")
    assert not hasattr(component, "send_runtime_fields")
    assert not hasattr(component, "check_not_empty_import_export_lists")
    assert not hasattr(component, "check_valid_exchange_field_names")
    assert not hasattr(component, "_validate_runtime_grid_data_field")
    assert not hasattr(component, "_sync_data_from_runtime_state")


def test_runtime_state_creation_receive_and_send() -> None:
    grid = make_test_grid(name="atm")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    component.data["sensible_heat_flux"] = jnp.full(grid.shape, 2.0)

    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    assert set(state.incoming.field_names) == {"temperature"}
    assert set(state.outgoing.field_names) == {"sensible_heat_flux"}
    assert isinstance(state.incoming.get("temperature"), jax.Array)

    incoming = state.incoming.set("temperature", jnp.full(grid.shape, 5.0))
    state = receive_runtime_fields(
        state.with_incoming(incoming),
        contract,
    )
    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 5.0))

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(
            dt_seconds=3.0,
            settings=VercorSettings(),
        ),
    )
    assert_allclose_compact(stepped.data.get("temperature"), np.full(grid.shape, 8.0))

    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.outgoing.get("sensible_heat_flux"),
        np.full(grid.shape, 2.0),
    )


def test_component_validation_and_runtime_receive_delegate() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    with pytest.raises(ComponentError, match="no fields to import"):
        check_not_empty_import_export_lists(component, RuntimeComponentContract())

    import_only = RuntimeComponentContract(imports=("temperature",))
    with pytest.raises(ComponentError, match="no fields to export"):
        check_not_empty_import_export_lists(component, import_only)

    overlapping = RuntimeComponentContract(
        imports=("temperature",),
        exports=("temperature",),
    )
    with pytest.raises(ComponentError, match="overlapping fields"):
        check_not_empty_import_export_lists(component, overlapping)

    invalid = RuntimeComponentContract(
        imports=("temperature",),
        exports=("not_supported",),
    )
    with pytest.raises(ComponentError, match="not a recognized exchange variable"):
        check_valid_exchange_field_names(component, invalid)

    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    state = state.with_incoming(
        state.incoming.set("temperature", np.ones(component.grid.shape))
    )
    received = receive_runtime_fields(state, contract)
    assert_allclose_compact(
        received.data.get("temperature"), np.ones(component.grid.shape)
    )


def test_runtime_validation_uses_component_grid_shape_without_shape_argument() -> None:
    grid = make_test_grid(
        name="atm",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    valid_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                "temperature": jnp.ones(grid.shape),
                "sensible_heat_flux": jnp.zeros(grid.shape),
            }
        ),
        incoming=RuntimeFieldStore.from_mapping({"temperature": jnp.ones(grid.shape)}),
        outgoing=RuntimeFieldStore.from_mapping(
            {"sensible_heat_flux": jnp.zeros(grid.shape)}
        ),
    )

    validate_component_runtime_contract_fields(component, valid_state, contract)
    component.validate_runtime_state(valid_state, contract)

    bad_state = valid_state.with_incoming(
        RuntimeFieldStore.from_mapping({"temperature": jnp.ones((1, 3))})
    )
    with pytest.raises(
        CouplerError,
        match=r"has shape \(1, 3\), expected \(2, 3\)",
    ):
        validate_component_runtime_contract_fields(component, bad_state, contract)


def test_send_runtime_fields_updates_outgoing_store() -> None:
    grid = make_test_grid()
    component = DummyComponent(name="ATM", grid=grid)
    timestamp = datetime(2000, 1, 1)
    contract = RuntimeComponentContract(exports=("temperature",))
    component.data["temperature"] = jnp.full(grid.shape, 1.0)

    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.full(grid.shape, 1.0),
    )
    assert isinstance(component_state.outgoing.get("temperature"), jax.Array)

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    monthly = jnp.zeros((*grid.shape, 12), dtype=jnp.float64)
    monthly = monthly.at[:, :, 0].set(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    component.settings.apply_time_interpolation = True
    component.settings.get_field_time_slice = False
    component.data["temperature"] = monthly
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        scalar_runtime_step_info(
            timestamp, runtime_coupler.clock, runtime_coupler.settings
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.asarray(monthly[:, :, 0]).T,
    )

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1)
    )
    daily = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, *grid.shape))
    component.settings.apply_time_interpolation = False
    component.settings.get_field_time_slice = True
    component.data["temperature"] = daily
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        scalar_runtime_step_info(
            runtime_coupler.clock.start,
            runtime_coupler.clock,
            runtime_coupler.settings,
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.asarray(daily[2]),
    )


def test_component_forcing_data_read_and_runtime_write_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forcing.nc"
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(path)

    reader = ComponentForcingData()
    reader.DATA_FILES = {"sample": str(path)}

    normal_read = reader._read_forcing("foo", "sample")
    flipped_read = reader._read_forcing("foo", "sample", flip_y=True)

    assert isinstance(normal_read, jax.Array)
    assert isinstance(flipped_read, jax.Array)
    assert_allclose_compact(normal_read, source.T)
    assert_allclose_compact(
        flipped_read,
        np.flip(source.T, axis=1),
    )

    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        reader._read_forcing("foo", "missing")

    with pytest.raises(KeyError, match="Provided 'where' key 'sample'"):
        reader._read_forcing("bar", "sample")

    broken = tmp_path / "broken.nc"
    broken.write_text("not-a-netcdf-file", encoding="utf-8")
    reader.DATA_FILES["broken"] = str(broken)

    with pytest.raises(RuntimeError, match="Error reading variable 'foo'"):
        reader._read_forcing("foo", "broken")

    state = RuntimeComponentState(
        data=RuntimeFieldStore.empty(),
        incoming=RuntimeFieldStore.from_mapping(
            {"temperature": jnp.asarray([[10.0, 11.0], [12.0, 13.0]])}
        ),
        outgoing=RuntimeFieldStore.from_mapping(
            {"humidity": jnp.asarray([[0.1, 0.2], [0.3, 0.4]])}
        ),
    )
    output = tmp_path / "runtime.nc"

    write_runtime_component_view_to_netcdf(
        RuntimeComponentView.from_component_state("ATM", make_test_grid(), state),
        output,
        masks={"fmask_OCN_ATM_bilinear": jnp.ones((2, 2))},
    )

    with xr.open_dataset(output) as dataset:
        assert_allclose_compact(
            dataset["incoming_temperature"].values,
            state.incoming.get("temperature"),
        )
        assert_allclose_compact(
            dataset["outgoing_humidity"].values,
            state.outgoing.get("humidity"),
        )
        assert_allclose_compact(dataset["latitude"].values, np.asarray([-1.0, 1.0]))
        assert_allclose_compact(dataset["longitude"].values, np.asarray([0.0, 1.0]))
        assert dataset["incoming_temperature"].attrs["component"] == "ATM"
        assert dataset["incoming_temperature"].attrs["runtime_store"] == "incoming"
        assert "fmask_OCN_ATM_bilinear" in dataset

    view_output = tmp_path / "runtime-view.nc"
    write_runtime_component_view_to_netcdf(
        RuntimeComponentView.from_component_state(
            "ATM",
            make_test_grid(),
            state,
        ),
        view_output,
    )
    with xr.open_dataset(view_output) as dataset:
        assert dataset["outgoing_humidity"].attrs["component"] == "ATM"
