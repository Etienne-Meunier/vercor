from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.components.base as base_module
from tests._coverage_support import CoverageCouplerStub, DummyComponent, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components.base import (
    ComponentForcingData,
    Shared,
    TimedNamedArray,
    write_shared_to_netcdf,
)
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError


@pytest.mark.fast_always
def test_timed_named_array_and_shared_accessors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timestamp = datetime(2001, 2, 3, 4, 5, 6)
    data = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    timed = TimedNamedArray(data=data, timestamp=timestamp, component_name="ATM")
    jax_timed = TimedNamedArray(
        data=jnp.asarray(data), timestamp=timestamp, component_name="ATM"
    )

    assert_allclose_compact(np.asarray(timed), data)
    assert_allclose_compact(np.asarray(jax_timed), data)
    assert isinstance(jax_timed.data, jax.Array)
    assert "ATM" in str(timed)
    assert "shape=(2, 2)" in repr(timed)

    shared = Shared()
    assert shared.is_empty
    shared.temperature = (data, timestamp, "ATM")
    shared["humidity"] = TimedNamedArray(
        data=jnp.asarray([[0.5, 0.6], [0.7, 0.8]]),
        timestamp=timestamp,
        component_name="OCN",
    )

    assert shared.field_names == ["temperature", "humidity"]
    assert_allclose_compact(shared.fields()["temperature"], data)
    assert shared.timestamps()["temperature"] == timestamp
    assert shared.component_names()["humidity"] == "OCN"
    assert "temperature(ATM)" in str(shared)
    assert "humidity=" in repr(shared)
    assert shared.temperature.component_name == "ATM"
    assert isinstance(shared.humidity.data, jax.Array)

    assert shared["missing"] is None
    assert "has no item 'missing'" in capsys.readouterr().out

    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        _ = shared.missing


def test_shared_rejects_invalid_assignments() -> None:
    shared = Shared()

    with pytest.raises(ValueError, match="tuple of length 3"):
        shared.temperature = (np.asarray([1.0]), datetime(2000, 1, 1))

    with pytest.raises(TypeError, match="second element must be a datetime"):
        shared.temperature = (np.asarray([1.0]), "2000-01-01", "ATM")

    with pytest.raises(TypeError, match="provide a tuple"):
        shared.temperature = np.asarray([1.0])


def test_component_get_import_receive_merge_and_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = make_test_grid(name="atm")
    component = DummyComponent(name="ATM", grid=grid)
    coupler = CoverageCouplerStub()
    timestamp = coupler.clock.start

    incoming = Shared()
    incoming["temperature"] = (jnp.ones(grid.shape), timestamp, "OCN")
    component.import_fields(incoming)
    component._fields2import = ["temperature"]
    component.receive_fields(timestamp)

    assert_allclose_compact(component.get("temperature"), np.ones(grid.shape))
    assert isinstance(component.get("temperature"), jax.Array)

    component.data["specific_humidity"] = jnp.full(grid.shape, 0.5)
    assert_allclose_compact(
        component.get("specific_humidity"), np.full(grid.shape, 0.5)
    )
    assert isinstance(component.get("specific_humidity"), jax.Array)

    component.outgoing_fields["latent_heat_flux"] = (
        jnp.full(grid.shape, 2.0),
        timestamp,
        "ATM",
    )
    assert_allclose_compact(component.get("latent_heat_flux"), np.full(grid.shape, 2.0))
    assert isinstance(component.get("latent_heat_flux"), jax.Array)

    merged = component.merge_incoming_outgoing_fields()
    assert set(merged.field_names) == {"temperature", "latent_heat_flux"}

    component.outgoing_fields["temperature"] = (np.zeros(grid.shape), timestamp, "ATM")
    with pytest.raises(ComponentError, match="found in both incoming and outgoing"):
        component.get("temperature")

    component.outgoing_fields._fields.pop("temperature")

    captured: dict[str, Any] = {}

    def fake_write(shared: Shared, out_grid: Any, filename: Path) -> None:
        captured["shared"] = shared
        captured["grid"] = out_grid
        captured["filename"] = filename

    monkeypatch.setattr(base_module, "write_shared_to_netcdf", fake_write)

    component.finalize(cast(Any, coupler), Path("snapshot"))

    assert captured["grid"] is grid
    assert captured["filename"] == Path("atm_snapshot.nc")
    assert "mask_for_atm" in captured["shared"].field_names
    assert coupler.appended_components == ["ATM"]


def test_component_validation_and_runtime_receive_delegate() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    with pytest.raises(ComponentError, match="no fields to import"):
        component.check_not_empty_import_export_lists()

    component._fields2import = ["temperature"]
    with pytest.raises(ComponentError, match="no fields to export"):
        component.check_not_empty_import_export_lists()

    component._fields2export = ["temperature"]
    with pytest.raises(ComponentError, match="overlapping fields"):
        component.check_not_empty_import_export_lists()

    component._fields2export = ["not_supported"]
    with pytest.raises(ComponentError, match="not a recognized exchange variable"):
        component.check_valid_exchange_field_names()

    timestamp = datetime(2000, 1, 1, 0, 0, 0)
    component._fields2import = ["temperature"]
    component.incoming_fields["temperature"] = (
        np.ones(component.grid.shape),
        datetime(2000, 1, 1, 1, 0, 0),
        "OCN",
    )
    component.receive_fields(timestamp)
    assert_allclose_compact(
        component.data["temperature"], np.ones(component.grid.shape)
    )

    with pytest.raises(
        ComponentError, match="not found in incoming, outgoing or internal"
    ):
        component.get("missing")


def test_send_fields_delegates_to_runtime_sender() -> None:
    grid = make_test_grid()
    component = DummyComponent(name="ATM", grid=grid)
    coupler = CoverageCouplerStub()
    timestamp = coupler.clock.start
    component._fields2export = ["temperature"]
    component.data["temperature"] = jnp.full(grid.shape, 1.0)

    component.send_fields(timestamp, cast(Any, coupler))
    assert_allclose_compact(
        component.outgoing_fields.temperature.data,
        np.full(grid.shape, 1.0),
    )
    assert isinstance(component.outgoing_fields.temperature.data, jax.Array)

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    monthly = jnp.zeros((*grid.shape, 12), dtype=jnp.float64)
    monthly = monthly.at[:, :, 0].set(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    component.settings.apply_time_interpolation = True
    component.settings.get_field_time_slice = False
    component.data["temperature"] = monthly
    component.send_fields(timestamp, cast(Any, runtime_coupler))
    assert_allclose_compact(
        component.outgoing_fields.temperature.data,
        np.asarray(monthly[:, :, 0]).T,
    )
    assert isinstance(component.outgoing_fields.temperature.data, jax.Array)

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1)
    )
    daily = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, *grid.shape))
    component.settings.apply_time_interpolation = False
    component.settings.get_field_time_slice = True
    component.data["temperature"] = daily
    component.send_fields(runtime_coupler.clock.start, cast(Any, runtime_coupler))
    assert_allclose_compact(
        component.outgoing_fields.temperature.data,
        np.asarray(daily[2]),
    )
    assert isinstance(component.outgoing_fields.temperature.data, jax.Array)


def test_component_forcing_data_read_and_write_round_trip(tmp_path: Path) -> None:
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

    shared = Shared()
    timestamp = datetime(2000, 1, 1, 12, 0, 0)
    shared["temperature"] = (
        jnp.asarray([[10.0, 11.0], [12.0, 13.0]]),
        timestamp,
        "ATM",
    )
    output = tmp_path / "shared.nc"

    write_shared_to_netcdf(shared, make_test_grid(), output)

    with xr.open_dataset(output) as dataset:
        assert_allclose_compact(dataset["temperature"].values, shared.temperature.data)
        assert_allclose_compact(dataset["latitude"].values, np.asarray([-1.0, 1.0]))
        assert_allclose_compact(dataset["longitude"].values, np.asarray([0.0, 1.0]))
        assert dataset["temperature"].attrs["component"] == "ATM"
        assert dataset["temperature"].attrs["timestamp"] == timestamp.isoformat()
